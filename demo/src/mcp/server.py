"""
MCP Server using Server-Sent Events (SSE)
Provides a single tool to return the current time.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """Base MCP message structure."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPTimeServer:
    """MCP Server that provides current time functionality."""
    
    def __init__(self):
        self.server_info = {
            "name": "time-server",
            "version": "1.0.0"
        }
        
        self.capabilities = {
            "tools": {}
        }
        
        self.tools = [
            {
                "name": "get_current_time",
                "description": "Returns the current date and time in various formats",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (e.g., 'UTC', 'America/New_York'). Defaults to system local time.",
                            "default": "local"
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format: 'iso', 'unix', 'human'. Defaults to 'iso'.",
                            "enum": ["iso", "unix", "human"],
                            "default": "iso"
                        }
                    }
                }
            }
        ]
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        logger.info(f"Initialize request from client: {params.get('clientInfo', {})}")
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities,
            "serverInfo": self.server_info
        }
    
    def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        logger.info("Listing available tools")
        return {"tools": self.tools}
    
    def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Tool called: {tool_name} with arguments: {arguments}")
        
        if tool_name == "get_current_time":
            return self._get_current_time(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _get_current_time(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get current time in specified format."""
        timezone = arguments.get("timezone", "local")
        format_type = arguments.get("format", "iso")
        
        now = datetime.now()
        
        # Format the time based on requested format
        if format_type == "iso":
            time_str = now.isoformat()
        elif format_type == "unix":
            time_str = str(int(now.timestamp()))
        elif format_type == "human":
            time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
        else:
            time_str = now.isoformat()
        
        result = {
            "content": [
                {
                    "type": "text",
                    "text": f"Current time ({timezone}): {time_str}"
                }
            ]
        }
        
        logger.info(f"Returning time: {time_str}")
        return result
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming MCP message and return response."""
        try:
            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")
            
            logger.info(f"Handling method: {method}")
            
            # Route to appropriate handler
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_list_tools(params)
            elif method == "tools/call":
                result = self.handle_call_tool(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Build response
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }


# Global server instance
mcp_server = MCPTimeServer()


async def sse_endpoint(request: Request) -> StreamingResponse:
    """
    SSE endpoint for MCP protocol.
    Handles bidirectional JSON-RPC communication over SSE.
    """
    
    async def event_stream():
        """Generate SSE events."""
        try:
            # Read the request body (client's messages)
            body = await request.body()
            
            if body:
                # Parse incoming message
                try:
                    message = json.loads(body.decode('utf-8'))
                    logger.info(f"Received message: {json.dumps(message, indent=2)}")
                    
                    # Process message
                    response = await mcp_server.handle_message(message)
                    
                    # Send response as SSE event
                    event_data = json.dumps(response)
                    yield f"data: {event_data}\n\n"
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
            
            # Keep connection alive
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in event stream: {e}", exc_info=True)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "server": mcp_server.server_info,
        "timestamp": datetime.now().isoformat()
    })


async def server_info(request: Request) -> JSONResponse:
    """Return server information."""
    return JSONResponse({
        "serverInfo": mcp_server.server_info,
        "capabilities": mcp_server.capabilities,
        "tools": mcp_server.tools
    })


# Create Starlette application
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", sse_endpoint, methods=["GET", "POST"]),
        Route("/health", health_check, methods=["GET"]),
        Route("/info", server_info, methods=["GET"]),
    ]
)


def main():
    """Run the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MCP Time Server using SSE'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Update log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting MCP Time Server on {args.host}:{args.port}")
    logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")
    logger.info(f"Server info: http://{args.host}:{args.port}/info")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


if __name__ == '__main__':
    main()