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
* A Kubernetes cluster in the cloud, on your laptop (using something like minikube), etc
* (Optional) set `MCP_BASE_URL`, `VECTOR_STORE_DIR`, or `CLAUDE_MODEL` to override defaults used by the API server.

## Configure 

Please visit the [k8sgpt documentation](https://github.com/k8sgpt-ai/k8sgpt) for configuration instructions.

## (Optional) Using Minikube

If you do't have a Kubernetes cluster already... Start a cluster and deploy a pod:

```bash
# start a cluster
minikube start

# export the kubeconfig
kubectl config use-context minikube

# create a simple pod
kubectl apply -f echoserver-pod.yaml

# verify
kubectl get pods
```

## Installation

Highly recommended to use a virtual environment something like `venv` or `conda`.

```bash
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
export VOYAGE_API_KEY="<YOUR API KEY>"
make ingest
```

The script reports how many files and chunks were processed and persists the resulting embeddings to
the ChromaDB directory (default `./chroma_db`).

## Start the MCP Time Server

The REST agent can call MCP tools for supplementary data (currently a `get_current_time` tool).
Launch the Starlette-based server in a separate terminal:

```bash
make mcp
```

By default the server listens on `http://127.0.0.1:8080`, exposes a `/sse` endpoint for JSON-RPC
messages, and implements `/info` and `/health` discovery routes.

## Run the RAG API Service

With embeddings in place, start the FastAPI service. Export both API keys and optionally override the
vector store location or MCP endpoint:

```bash
export ANTHROPIC_API_KEY="<YOUR API KEY>"
export VOYAGE_API_KEY="<YOUR API KEY>"
make server
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

## Use the Terminal Client

The Rich-based CLI wraps the REST endpoints and provides interactive, single-shot, and streaming
modes:

```bash
# the QUESTION is contained at the top of the Makefile
make client
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
