"""
RAG Agent REST Client
Client for interacting with the RAG Agent API.
Supports both regular and streaming queries.
"""

import os
import json
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import argparse

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout


console = Console()


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[Dict[str, Any]]
    mcp_tool_calls: Optional[List[Dict[str, Any]]] = None
    model: str = ""
    usage: Optional[Dict[str, int]] = None


class RAGClient:
    """Client for the RAG Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        """
        Initialize the RAG client.
        
        Args:
            base_url: Base URL of the RAG Agent API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def query(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ) -> QueryResult:
        """
        Send a query to the RAG agent.
        
        Args:
            question: The question to ask
            max_results: Maximum number of documents to retrieve
            use_mcp_tools: Whether to allow MCP tool usage
            
        Returns:
            QueryResult object with the answer and metadata
        """
        url = f"{self.base_url}/query"
        payload = {
            "question": question,
            "max_results": max_results,
            "use_mcp_tools": use_mcp_tools,
            "stream": False
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return QueryResult(
                    answer=data["answer"],
                    sources=data["sources"],
                    mcp_tool_calls=data.get("mcp_tool_calls"),
                    model=data.get("model", ""),
                    usage=data.get("usage")
                )
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def query_stream(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ):
        """
        Send a streaming query to the RAG agent.
        
        Args:
            question: The question to ask
            max_results: Maximum number of documents to retrieve
            use_mcp_tools: Whether to allow MCP tool usage
            
        Yields:
            Dictionary with type and data for each stream event
        """
        url = f"{self.base_url}/query/stream"
        payload = {
            "question": question,
            "max_results": max_results,
            "use_mcp_tools": use_mcp_tools,
            "stream": True
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield data
                            except json.JSONDecodeError:
                                console.print(f"[yellow]Warning: Invalid JSON: {line}[/yellow]")
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the RAG agent."""
        url = f"{self.base_url}/health"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def list_mcp_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        url = f"{self.base_url}/mcp/tools"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def list_sources(self) -> Dict[str, Any]:
        """List all sources in the vector store."""
        url = f"{self.base_url}/sources"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise


class RAGClientCLI:
    """Command-line interface for the RAG client."""
    
    def __init__(self, client: RAGClient):
        self.client = client
    
    def display_health(self):
        """Display health check information."""
        console.print("\n[bold cyan]Health Check[/bold cyan]")
        console.print("=" * 60)
        
        try:
            health = self.client.health_check()
            
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Status", health["status"])
            table.add_row("Documents", str(health["vector_store_documents"]))
            table.add_row("MCP Connected", "✓" if health["mcp_server_connected"] else "✗")
            table.add_row("Model", health["model"])
            
            console.print(table)
            console.print()
            
        except Exception as e:
            console.print(f"[red]Health check failed: {e}[/red]\n")
    
    def display_sources(self):
        """Display available sources."""
        console.print("\n[bold cyan]Available Sources[/bold cyan]")
        console.print("=" * 60)
        
        try:
            data = self.client.list_sources()
            
            console.print(f"Total documents: [green]{data['total_documents']}[/green]\n")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Filename", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Chunks", justify="right", style="green")
            
            for source in data["sources"]:
                table.add_row(
                    source["filename"],
                    source["file_type"],
                    str(source["chunks"])
                )
            
            console.print(table)
            console.print()
            
        except Exception as e:
            console.print(f"[red]Failed to list sources: {e}[/red]\n")
    
    def display_mcp_tools(self):
        """Display available MCP tools."""
        console.print("\n[bold cyan]MCP Tools[/bold cyan]")
        console.print("=" * 60)
        
        try:
            data = self.client.list_mcp_tools()
            
            connected = "✓" if data["connected"] else "✗"
            console.print(f"MCP Server Connected: {connected}\n")
            
            if data["tools"]:
                for tool in data["tools"]:
                    console.print(f"[bold green]{tool['name']}[/bold green]")
                    console.print(f"  {tool['description']}")
                    console.print()
            else:
                console.print("[yellow]No MCP tools available[/yellow]")
            
            console.print()
            
        except Exception as e:
            console.print(f"[red]Failed to list MCP tools: {e}[/red]\n")
    
    def display_sources_panel(self, sources: List[Dict[str, Any]]):
        """Display retrieved sources."""
        if not sources:
            return
        
        console.print("\n[bold cyan]Retrieved Sources[/bold cyan]")
        
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            chunk_idx = metadata.get("chunk_index", "?")
            distance = source.get("distance", 0)
            
            console.print(
                f"[dim]Source {i}: {filename} (chunk {chunk_idx}) "
                f"[similarity: {1 - distance:.2%}][/dim]"
            )
    
    def display_tool_calls(self, tool_calls: Optional[List[Dict[str, Any]]]):
        """Display MCP tool calls."""
        if not tool_calls:
            return
        
        console.print("\n[bold cyan]MCP Tool Calls[/bold cyan]")
        
        for call in tool_calls:
            console.print(f"[bold green]Tool:[/bold green] {call['tool']}")
            console.print(f"[bold yellow]Input:[/bold yellow] {json.dumps(call['input'], indent=2)}")
            console.print(f"[bold blue]Result:[/bold blue] {call['result']}")
            console.print()
    
    def display_usage(self, usage: Optional[Dict[str, int]]):
        """Display token usage."""
        if not usage:
            return
        
        console.print(
            f"\n[dim]Tokens - Input: {usage['input_tokens']:,} | "
            f"Output: {usage['output_tokens']:,} | "
            f"Total: {usage['input_tokens'] + usage['output_tokens']:,}[/dim]"
        )
    
    def query_standard(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ):
        """Execute a standard (non-streaming) query."""
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Querying RAG agent...", total=None)
            
            try:
                result = self.client.query(
                    question=question,
                    max_results=max_results,
                    use_mcp_tools=use_mcp_tools
                )
                progress.stop()
                
                # Display answer
                console.print("[bold cyan]Answer:[/bold cyan]")
                console.print(Panel(Markdown(result.answer), border_style="green"))
                
                # Display sources
                self.display_sources_panel(result.sources)
                
                # Display tool calls
                self.display_tool_calls(result.mcp_tool_calls)
                
                # Display usage
                self.display_usage(result.usage)
                console.print()
                
            except Exception as e:
                progress.stop()
                console.print(f"[red]Query failed: {e}[/red]\n")
    
    def query_streaming(
        self,
        question: str,
        max_results: int = 5,
        use_mcp_tools: bool = True
    ):
        """Execute a streaming query."""
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
        console.print("[bold cyan]Answer:[/bold cyan]")
        
        answer_parts = []
        sources = []
        usage = None
        
        try:
            for event in self.client.query_stream(
                question=question,
                max_results=max_results,
                use_mcp_tools=use_mcp_tools
            ):
                event_type = event.get("type")
                data = event.get("data")
                
                if event_type == "sources":
                    sources = data
                elif event_type == "content":
                    console.print(data, end="")
                    answer_parts.append(data)
                elif event_type == "usage":
                    usage = data
                elif event_type == "error":
                    console.print(f"\n[red]Error: {data}[/red]")
            
            console.print("\n")
            
            # Display sources
            self.display_sources_panel(sources)
            
            # Display usage
            self.display_usage(usage)
            console.print()
            
        except Exception as e:
            console.print(f"\n[red]Streaming query failed: {e}[/red]\n")
    
    def interactive_mode(self):
        """Run in interactive mode."""
        console.print("\n[bold green]RAG Agent Interactive Mode[/bold green]")
        console.print("Type 'quit' or 'exit' to exit")
        console.print("Type 'help' for available commands")
        console.print("=" * 60 + "\n")
        
        while True:
            try:
                question = console.input("[bold cyan]Question:[/bold cyan] ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit']:
                    console.print("\n[green]Goodbye![/green]\n")
                    break
                
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                if question.lower() == 'health':
                    self.display_health()
                    continue
                
                if question.lower() == 'sources':
                    self.display_sources()
                    continue
                
                if question.lower() == 'tools':
                    self.display_mcp_tools()
                    continue
                
                # Execute query
                self.query_standard(question)
                
            except KeyboardInterrupt:
                console.print("\n\n[green]Goodbye![/green]\n")
                break
            except EOFError:
                console.print("\n\n[green]Goodbye![/green]\n")
                break
    
    def show_help(self):
        """Show help message."""
        console.print("\n[bold cyan]Available Commands[/bold cyan]")
        console.print("=" * 60)
        console.print("  [green]<question>[/green]  - Ask a question")
        console.print("  [green]health[/green]      - Show health status")
        console.print("  [green]sources[/green]     - List available sources")
        console.print("  [green]tools[/green]       - List MCP tools")
        console.print("  [green]help[/green]        - Show this help message")
        console.print("  [green]quit/exit[/green]   - Exit interactive mode")
        console.print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RAG Agent REST Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rag_client.py

  # Single query
  python rag_client.py --query "What is the main topic?"

  # Streaming query
  python rag_client.py --query "Explain this concept" --stream

  # Health check
  python rag_client.py --health

  # List sources
  python rag_client.py --sources

  # List MCP tools
  python rag_client.py --tools
        """
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='RAG Agent API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--query', '-q',
        help='Question to ask (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming mode for queries'
    )
    parser.add_argument(
        '--max-results',
        type=int,
        default=5,
        help='Maximum number of documents to retrieve (default: 5)'
    )
    parser.add_argument(
        '--no-mcp',
        action='store_true',
        help='Disable MCP tool usage'
    )
    parser.add_argument(
        '--health',
        action='store_true',
        help='Show health check and exit'
    )
    parser.add_argument(
        '--sources',
        action='store_true',
        help='List available sources and exit'
    )
    parser.add_argument(
        '--tools',
        action='store_true',
        help='List MCP tools and exit'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60.0)'
    )
    
    args = parser.parse_args()
    
    # Create client
    client = RAGClient(base_url=args.url, timeout=args.timeout)
    cli = RAGClientCLI(client)
    
    # Handle different modes
    if args.health:
        cli.display_health()
    elif args.sources:
        cli.display_sources()
    elif args.tools:
        cli.display_mcp_tools()
    elif args.query:
        if args.stream:
            cli.query_streaming(
                question=args.query,
                max_results=args.max_results,
                use_mcp_tools=not args.no_mcp
            )
        else:
            cli.query_standard(
                question=args.query,
                max_results=args.max_results,
                use_mcp_tools=not args.no_mcp
            )
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == '__main__':
    main()