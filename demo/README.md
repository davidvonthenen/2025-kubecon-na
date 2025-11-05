# Anthropic RAG Agent with MCP Integration

This repository provides an end-to-end retrieval augmented generation (RAG) stack that combines

* **Voyage AI embeddings** for document vectorization
* **ChromaDB** for persistent vector storage
* **Anthropic Claude** for answer generation
* **Model Context Protocol (MCP)** for tool-augmented responses exposed through a lightweight time server

The codebase includes three runnable services:

1. A document ingestion pipeline that chunks Markdown/PDF files and stores Voyage embeddings in ChromaDB.
2. A FastAPI-powered REST service that answers questions with Claude and optionally calls MCP tools.
3. A Rich-based terminal client for issuing queries, exploring health metrics, and streaming responses.

## Prerequisites

* **Python** 3.10+
* **API keys**
  * `VOYAGE_API_KEY` for embedding generation
  * `ANTHROPIC_API_KEY` for Claude message completions
* (Optional) set `MCP_BASE_URL`, `VECTOR_STORE_DIR`, or `CLAUDE_MODEL` to override defaults used by the API server.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Content

Place the knowledge sources you want to index inside a directory such as `docs/`. The ingestion
pipeline currently accepts **Markdown (`.md`)** and **PDF (`.pdf`)** files. Each document is
converted to plain text, split into overlapping chunks, and tagged with metadata about the original
file.

## Build the Vector Store

Run the ingestion pipeline after exporting your Voyage API key:

```bash
export VOYAGE_API_KEY="sk_your_voyage_key"
python -m src.ingest.ingest --docs-dir ./docs --db-dir ./chroma_db
```

The script reports how many files and chunks were processed and persists the resulting embeddings to
the ChromaDB directory (default `./chroma_db`).

## Start the MCP Time Server

The REST agent can call MCP tools for supplementary data (currently a `get_current_time` tool).
Launch the Starlette-based server in a separate terminal:

```bash
python -m src.mcp.server
```

By default the server listens on `http://127.0.0.1:8080`, exposes a `/sse` endpoint for JSON-RPC
messages, and implements `/info` and `/health` discovery routes.

## Run the RAG API Service

With embeddings in place, start the FastAPI service. Export both API keys and optionally override the
vector store location or MCP endpoint:

```bash
export ANTHROPIC_API_KEY="sk_your_anthropic_key"
export VOYAGE_API_KEY="sk_your_voyage_key"
python -m src.agent.server --host 0.0.0.0 --port 8000 --vector-store-dir ./chroma_db
```

On startup the service loads the Chroma collection, connects to the MCP server to list tools, and then
serves the following endpoints:

| Method | Path            | Description                                                                 |
|--------|-----------------|-----------------------------------------------------------------------------|
| POST   | `/query`        | Returns a JSON answer, retrieved sources, tool call metadata, and usage.   |
| POST   | `/query/stream` | Streams NDJSON events (`sources`, `content`, `usage`, optional `error`).    |
| GET    | `/health`       | Reports document count, Claude model, and MCP connectivity status.         |
| GET    | `/mcp/tools`    | Lists tools discovered from the MCP server.                                |
| GET    | `/sources`      | Summarizes indexed files and chunk counts.                                 |

Interactive API docs are automatically available at `http://<host>:<port>/docs` when the server is running.

### Example Query (non-streaming)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Summarize the architecture of this project",
        "max_results": 4,
        "use_mcp_tools": true
      }'
```

### Example Streaming Query

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What time is it and how does the RAG flow work?",
        "stream": true,
        "use_mcp_tools": true
      }'
```

Each line in the response is a JSON object that can be parsed incrementally.

## Use the Terminal Client

The Rich-based CLI wraps the REST endpoints and provides interactive, single-shot, and streaming
modes:

```bash
python -m src.client.client --url http://localhost:8000 --query "How do I refresh the index?"
```

Key flags:

* `--stream` – stream the answer in real time.
* `--health` – display the `/health` response and exit.
* `--sources` – list indexed documents.
* `--tools` – show MCP tools currently available to the agent.
* Omitting `--query` launches an interactive prompt that maintains conversation history with the API.

## Makefile Shortcuts

Common commands are packaged as make targets once your environment is active:

```bash
make ingest       # python -m src.ingest.ingest
make mcp-server   # python -m src.mcp.server
make server       # python -m src.agent.server
make client       # python -m src.client.client --query "$(QUESTION)"
```

## Project Structure

```
src/
├── agent/            # RAG FastAPI service and orchestration logic
├── client/           # Terminal client for REST interactions
├── ingest/           # Document processor and embedding pipeline
└── mcp/              # Minimal MCP time server
```

## Troubleshooting

* Ensure both `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` are set before starting the API server; startup
  will fail fast if either is missing.
* The REST service expects the Chroma directory to exist and contain the `rag_documents` collection.
  Re-run the ingestion script after adding or updating documents.
* The MCP server must be reachable for Claude to request tool calls. Use `GET /mcp/tools` or
  `python -m src.client.client --tools` to verify connectivity.
