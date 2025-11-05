"""
RAG Agent with REST API
Integrates ChromaDB vector store with Claude for question answering,
and connects to MCP server for additional tools.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import voyageai
from anthropic import Anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to answer")
    max_results: int = Field(5, description="Maximum number of documents to retrieve", ge=1, le=20)
    use_mcp_tools: bool = Field(True, description="Whether to allow MCP tool usage")
    stream: bool = Field(False, description="Whether to stream the response")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[Dict[str, Any]]
    mcp_tool_calls: Optional[List[Dict[str, Any]]] = None
    model: str
    usage: Optional[Dict[str, int]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_documents: int
    mcp_server_connected: bool
    model: str


# RAG Agent Components
class VectorStoreRetriever:
    """Retrieves relevant documents from ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.client.get_collection(name="rag_documents")
            logger.info(f"Loaded collection with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            raise
    
    def retrieve(
        self,
        query_embedding: List[float],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return documents


class MCPClient:
    """Client for communicating with MCP server."""
    
    def __init__(self, mcp_base_url: str = "http://127.0.0.1:8080"):
        self.base_url = mcp_base_url
        self.sse_url = f"{mcp_base_url}/sse"
        self.info_url = f"{mcp_base_url}/info"
        self.tools = []
        self._initialized = False
        self._message_id = 0
    
    async def initialize(self) -> bool:
        """Initialize connection and fetch available tools."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get server info
                response = await client.get(self.info_url)
                if response.status_code == 200:
                    info = response.json()
                    self.tools = info.get('tools', [])
                    self._initialized = True
                    logger.info(f"MCP Client initialized with {len(self.tools)} tools")
                    return True
                else:
                    logger.error(f"Failed to get MCP server info: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error initializing MCP client: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not self._initialized:
            await self.initialize()
        
        self._message_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": str(self._message_id),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.sse_url,
                    json=message,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    # Parse SSE response
                    content = response.text
                    # Extract JSON from SSE data format
                    if content.startswith("data: "):
                        json_str = content[6:].strip()
                        result = json.loads(json_str)
                        if "result" in result:
                            return result["result"]
                        elif "error" in result:
                            logger.error(f"MCP tool error: {result['error']}")
                            return {"error": result["error"]}
                    return json.loads(content)
                else:
                    logger.error(f"MCP tool call failed: {response.status_code}")
                    return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            return {"error": str(e)}
    
    def get_tools_for_claude(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Claude tool format."""
        claude_tools = []
        for tool in self.tools:
            claude_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["inputSchema"]
            }
            claude_tools.append(claude_tool)
        return claude_tools
    
    async def health_check(self) -> bool:
        """Check if MCP server is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


class RAGAgent:
    """Main RAG agent that orchestrates retrieval and generation."""
    
    def __init__(
        self,
        anthropic_api_key: str,
        voyage_api_key: str,
        vector_store_dir: str = "./chroma_db",
        mcp_base_url: str = "http://127.0.0.1:8080",
        model: str = "claude-3-5-sonnet-20241022"
    ):
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.model = model
        
        # Initialize components
        self.retriever = VectorStoreRetriever(vector_store_dir)
        self.mcp_client = MCPClient(mcp_base_url)
        
        logger.info("RAG Agent initialized")
    
    async def initialize(self):
        """Initialize async components."""
        await self.mcp_client.initialize()
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query."""
        result = self.voyage_client.embed(
            texts=[query],
            model="voyage-3",
            input_type="query"
        )
        return result.embeddings[0]
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('filename', 'Unknown')
            content = doc['content']
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context."""
        return f"""You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately and comprehensively.

If the context doesn't contain enough information to answer the question fully, 
say so and provide what information you can.

Always cite your sources by referencing the source numbers in brackets (e.g., [Source 1]).

CONTEXT:
{context}
"""
    
    async def _handle_tool_use(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Handle tool use by calling MCP server."""
        logger.info(f"Calling MCP tool: {tool_name} with input: {tool_input}")
        result = await self.mcp_client.call_tool(tool_name, tool_input)
        
        if "error" in result:
            return f"Error calling tool: {result['error']}"
        
        # Extract text content from result
        if "content" in result:
            content_items = result["content"]
            text_parts = [
                item["text"] for item in content_items 
                if item.get("type") == "text"
            ]
            return "\n".join(text_parts)
        
        return json.dumps(result)
    
    async def query(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ) -> Dict[str, Any]:
        """Process a query and return an answer."""
        # Generate query embedding
        logger.info(f"Processing query: {question}")
        query_embedding = self._generate_query_embedding(question)
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(query_embedding, max_results)
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Build context and system prompt
        context = self._build_context(documents)
        system_prompt = self._build_system_prompt(context)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Prepare tools if MCP is enabled
        tools = None
        if use_mcp_tools and self.mcp_client._initialized:
            tools = self.mcp_client.get_tools_for_claude()
        
        # Call Claude
        mcp_tool_calls = []
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else None
            )
            
            # Handle tool use if present
            while response.stop_reason == "tool_use":
                # Process tool calls
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id
                        
                        logger.info(f"Claude requested tool: {tool_name}")
                        
                        # Call the tool
                        tool_result = await self._handle_tool_use(tool_name, tool_input)
                        
                        mcp_tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": tool_result
                        })
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result
                        })
                
                # Continue conversation with tool results
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Get next response
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools if tools else None
                )
            
            # Extract final answer
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text
            
            return {
                "answer": answer,
                "sources": documents,
                "mcp_tool_calls": mcp_tool_calls if mcp_tool_calls else None,
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise
    
    async def query_stream(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ):
        """Process a query and stream the answer."""
        # Generate query embedding
        logger.info(f"Processing streaming query: {question}")
        query_embedding = self._generate_query_embedding(question)
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(query_embedding, max_results)
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Build context and system prompt
        context = self._build_context(documents)
        system_prompt = self._build_system_prompt(context)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Prepare tools if MCP is enabled
        tools = None
        if use_mcp_tools and self.mcp_client._initialized:
            tools = self.mcp_client.get_tools_for_claude()
        
        # Stream response
        try:
            async with self.anthropic_client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else None
            ) as stream:
                # First, send sources
                yield json.dumps({
                    "type": "sources",
                    "data": documents
                }) + "\n"
                
                # Then stream the answer
                async for text in stream.text_stream:
                    yield json.dumps({
                        "type": "content",
                        "data": text
                    }) + "\n"
                
                # Finally, send usage stats
                message = await stream.get_final_message()
                yield json.dumps({
                    "type": "usage",
                    "data": {
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens
                    }
                }) + "\n"
                
        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"


# FastAPI Application
app = FastAPI(
    title="RAG Agent API",
    description="REST API for RAG-based question answering with MCP tool integration",
    version="1.0.0"
)

# Global agent instance
agent: Optional[RAGAgent] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG agent on startup."""
    global agent
    
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    voyage_api_key = os.getenv('VOYAGE_API_KEY')
    
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    if not voyage_api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    
    vector_store_dir = os.getenv('VECTOR_STORE_DIR', './chroma_db')
    mcp_base_url = os.getenv('MCP_BASE_URL', 'http://127.0.0.1:8080')
    model = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    
    agent = RAGAgent(
        anthropic_api_key=anthropic_api_key,
        voyage_api_key=voyage_api_key,
        vector_store_dir=vector_store_dir,
        mcp_base_url=mcp_base_url,
        model=model
    )
    
    await agent.initialize()
    logger.info("RAG Agent API started successfully")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG agent with a question.
    
    Returns an answer based on the knowledge base and optionally uses MCP tools.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.query(
            question=request.question,
            max_results=request.max_results,
            use_mcp_tools=request.use_mcp_tools
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Query the RAG agent with streaming response.
    
    Returns a stream of JSON objects with answer chunks.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream mode must be enabled")
    
    try:
        return StreamingResponse(
            agent.query_stream(
                question=request.question,
                max_results=request.max_results,
                use_mcp_tools=request.use_mcp_tools
            ),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        logger.error(f"Error processing streaming query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        mcp_connected = await agent.mcp_client.health_check()
        doc_count = agent.retriever.collection.count()
        
        return HealthResponse(
            status="healthy",
            vector_store_documents=doc_count,
            mcp_server_connected=mcp_connected,
            model=agent.model
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "tools": agent.mcp_client.tools,
        "connected": await agent.mcp_client.health_check()
    }


@app.get("/sources")
async def list_sources():
    """List all available sources in the vector store."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Get all documents
        results = agent.retriever.collection.get()
        
        # Extract unique sources
        sources = {}
        for metadata in results['metadatas']:
            filename = metadata.get('filename', 'Unknown')
            if filename not in sources:
                sources[filename] = {
                    'filename': filename,
                    'file_type': metadata.get('file_type', 'unknown'),
                    'chunks': 0
                }
            sources[filename]['chunks'] += 1
        
        return {
            "total_documents": len(results['ids']),
            "sources": list(sources.values())
        }
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the RAG Agent API server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RAG Agent REST API'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--vector-store-dir',
        default='./chroma_db',
        help='Vector store directory (default: ./chroma_db)'
    )
    parser.add_argument(
        '--mcp-url',
        default='http://127.0.0.1:8080',
        help='MCP server base URL (default: http://127.0.0.1:8080)'
    )
    parser.add_argument(
        '--model',
        default='claude-sonnet-4-5-20250929',
        help='Claude model to use'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['VECTOR_STORE_DIR'] = args.vector_store_dir
    os.environ['MCP_BASE_URL'] = args.mcp_url
    os.environ['CLAUDE_MODEL'] = args.model
    
    # Update log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("RAG Agent API Server")
    logger.info("=" * 60)
    logger.info(f"API endpoint: http://{args.host}:{args.port}")
    logger.info(f"Documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"Vector store: {args.vector_store_dir}")
    logger.info(f"MCP server: {args.mcp_url}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


if __name__ == '__main__':
    main()