PYTHON ?= python
QUESTION ?= Provide guidance on what administrators should do when creating pods in a Kubernetes cluster.
# QUESTION ?= What is the current time?

.PHONY: ingest mcp mcp-server server agent client

ingest:
	$(PYTHON) -m src.ingest.ingest

mcp:
	$(PYTHON) -m src.mcp.server

mcp-server: mcp

rag-agent:
	$(PYTHON) -m src.agent.server

server: rag-agent

client:
	$(PYTHON) -m src.client.client --query "$(QUESTION)"
