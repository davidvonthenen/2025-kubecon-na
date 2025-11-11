PYTHON ?= python
# QUESTION ?= What is the current time?
QUESTION ?= Using k8sgpt, analyze the cluster and report any workloads where securityContext.allowPrivilegeEscalation is true or unset. Return a compact list of offending objects (namespace/kind/name), the detected condition, and a one-line remediation (set allowPrivilegeEscalation=false and run workloads as non-root). Conclude PASS if none found; otherwise FAIL.

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
