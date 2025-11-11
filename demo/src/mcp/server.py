"""
MCP Server using Server-Sent Events (SSE)
Adds a k8sgpt-backed tool to analyze Pod security for allowPrivilegeEscalation.
"""
import os
import re
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
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
    """MCP Server that provides time and k8sgpt-backed analysis."""
    def __init__(self):
        self.server_info = {
            "name": "tool-server",
            "version": "1.1.0"
        }
        self.capabilities = {"tools": {}}

        # Tool catalog
        self.tools = [
            {
                "name": "get_current_time",
                "description": "Returns the current date and time in various formats",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone label to attach in output",
                            "default": "local"
                        },
                        "format": {
                            "type": "string",
                            "description": "Output: 'iso', 'unix', or 'human'",
                            "enum": ["iso", "unix", "human"],
                            "default": "iso"
                        }
                    }
                }
            },
            {
                "name": "k8sgpt_analyze_pod_security",
                "description": "Runs `k8sgpt analyze --filter=Pod --explain --output=json` and reports workloads with allowPrivilegeEscalation=true or unset.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to scope analysis"
                        },
                        "no_cache": {
                            "type": "boolean",
                            "description": "Set to true to bypass k8sgpt cache",
                            "default": False
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Process timeout for k8sgpt command",
                            "default": 60
                        },
                        "raw": {
                            "type": "boolean",
                            "description": "Include raw k8sgpt JSON in output text",
                            "default": False
                        }
                    }
                }
            }
        ]

        # External tool config
        self.k8sgpt_bin = os.getenv("K8SGPT_BIN", "k8sgpt")
        self.kubectl_bin = os.getenv("KUBECTL_BIN", "kubectl")
        self.use_fake_k8sgpt = os.getenv("K8SGPT_FAKE_MODE", "").lower() in {"1", "true", "yes"}

    # ---------- Built-in tool handlers ----------

    def _get_current_time(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tz = arguments.get("timezone", "local")
        format_type = arguments.get("format", "iso")
        now = datetime.now()
        if format_type == "iso":
            time_str = now.isoformat()
        elif format_type == "unix":
            time_str = str(int(now.timestamp()))
        elif format_type == "human":
            time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
        else:
            time_str = now.isoformat()

        return {
            "content": [
                {"type": "text", "text": f"Current time ({tz}): {time_str}"}
            ]
        }

    # ---------- k8sgpt integration ----------

    async def _run_k8sgpt(self, args: List[str], timeout: int) -> Dict[str, Any]:
        """Execute k8sgpt and return (rc, stdout, stderr)."""
        if self.use_fake_k8sgpt:
            logger.info("K8SGPT_FAKE_MODE enabled; returning canned response instead of executing %s", self.k8sgpt_bin)
            fake_payload = self._fake_k8sgpt_payload(args)
            return {"rc": 0, "out": json.dumps(fake_payload), "err": ""}

        cmd = [self.k8sgpt_bin] + args
        logger.info("Executing: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"rc": -1, "out": "", "err": f"timeout after {timeout}s"}
            return {"rc": proc.returncode, "out": stdout_b.decode("utf-8", "replace"), "err": stderr_b.decode("utf-8", "replace")}
        except FileNotFoundError:
            return {"rc": -2, "out": "", "err": f"k8sgpt binary not found: {self.k8sgpt_bin}"}
        except Exception as e:
            return {"rc": -3, "out": "", "err": str(e)}

    @staticmethod
    def _summarize_allow_priv_escalation(k8sgpt_json: Any) -> Dict[str, Any]:
        """
        Heuristic: scan k8sgpt findings for mentions of allowPrivilegeEscalation
        being enabled or missing. Returns dict with pass_fail and offenders.
        """
        offenders = []
        text_fields = []

        def flatten_text(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        flatten_text(v)
                    else:
                        if isinstance(v, str):
                            text_fields.append(v)
            elif isinstance(obj, list):
                for it in obj:
                    flatten_text(it)

        flatten_text(k8sgpt_json)
        pattern = re.compile(r"allowPrivilegeEscalation", re.IGNORECASE)

        # try to also capture object identity from common fields
        def extract_identity(item: Dict[str, Any]) -> Optional[str]:
            ns = item.get("namespace") or item.get("Namespace") or item.get("ns")
            kind = item.get("kind") or item.get("Kind")
            name = item.get("name") or item.get("Name")
            if ns and kind and name:
                return f"{ns}/{kind}/{name}"
            return None

        # Raw items may be under "results", "issues" or similar keys
        items: List[Dict[str, Any]] = []
        if isinstance(k8sgpt_json, dict):
            for key in ("results", "issues", "items", "findings"):
                v = k8sgpt_json.get(key)
                if isinstance(v, list):
                    items.extend([x for x in v if isinstance(x, dict)])

        for it in items:
            it_text = json.dumps(it, ensure_ascii=False)
            if pattern.search(it_text):
                ident = extract_identity(it) or "unknown/unknown/unknown"
                offenders.append(ident)

        # if we didn't find structured items, fall back to any text mention
        if not offenders and any(pattern.search(t) for t in text_fields):
            offenders.append("unresolved-identifiers")

        status = "PASS" if not offenders else "FAIL"
        return {"status": status, "offenders": offenders}

    async def _run_kubectl(self, args: List[str], timeout: int) -> Dict[str, Any]:
        """Execute kubectl and return (rc, stdout, stderr)."""
        cmd = [self.kubectl_bin] + args
        logger.info("Executing: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"rc": -1, "out": "", "err": f"timeout after {timeout}s"}
            return {
                "rc": proc.returncode,
                "out": stdout_b.decode("utf-8", "replace"),
                "err": stderr_b.decode("utf-8", "replace")
            }
        except FileNotFoundError:
            return {"rc": -2, "out": "", "err": f"kubectl binary not found: {self.kubectl_bin}"}
        except Exception as e:  # pragma: no cover - defensive catch-all
            return {"rc": -3, "out": "", "err": str(e)}

    async def _kubectl_privilege_escalation_report(
        self, namespace: Optional[str], timeout: int
    ) -> Dict[str, Any]:
        """Return a k8sgpt-style JSON payload highlighting allowPrivilegeEscalation findings."""
        args = ["get", "pods", "-o", "json"]
        if namespace:
            args.extend(["-n", namespace])
        else:
            args.append("--all-namespaces")

        res = await self._run_kubectl(args, timeout=timeout)
        if res["rc"] != 0:
            logger.error("kubectl failed (rc=%s): %s", res["rc"], res["err"])
            return {
                "provider": "openai",
                "errors": [{"message": res["err"] or "kubectl invocation failed", "rc": res["rc"]}],
                "status": "ERROR",
                "problems": 1,
                "results": None,
            }

        try:
            payload = json.loads(res["out"])
        except json.JSONDecodeError:
            logger.error("kubectl JSON decode failed")
            return {
                "provider": "openai",
                "errors": [{"message": "kubectl output was not valid JSON"}],
                "status": "ERROR",
                "problems": 1,
                "results": None,
            }

        offenders: List[Dict[str, Any]] = []
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {}) or {}
            spec = item.get("spec", {}) or {}
            namespace_name = metadata.get("namespace", "default")
            pod_name = metadata.get("name", "unknown")
            container_specs: List[Dict[str, Any]] = []
            for key in ("containers", "initContainers", "ephemeralContainers"):
                value = spec.get(key)
                if isinstance(value, list):
                    container_specs.extend([c for c in value if isinstance(c, dict)])

            for container_spec in container_specs:
                container_name = container_spec.get("name", "unnamed")
                security_context = container_spec.get("securityContext") or {}
                allow_priv = security_context.get("allowPrivilegeEscalation")
                if allow_priv is True:
                    offenders.append(
                        {
                            "namespace": namespace_name,
                            "pod": pod_name,
                            "container": container_name,
                            "allowPrivilegeEscalation": allow_priv,
                        }
                    )

        if offenders:
            return {
                "provider": "openai",
                "errors": offenders,
                "status": "ERROR",
                "problems": len(offenders),
                "results": None,
            }

        return {
            "provider": "openai",
            "errors": None,
            "status": "OK",
            "problems": 0,
            "results": None,
        }

    async def _k8sgpt_analyze_pod_security(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run `k8sgpt analyze --filter=Pod --explain --output=json`
        Optionally add --namespace, --no-cache.
        """
        namespace = arguments.get("namespace")
        no_cache = bool(arguments.get("no_cache", False))
        timeout_s = int(arguments.get("timeout_seconds", 60))
        include_raw = bool(arguments.get("raw", False))

        args = ["analyze", "--filter=Pod", "--explain", "--output=json"]
        if namespace:
            args += ["--namespace", namespace]
        if no_cache:
            args.append("--no-cache")

        res = await self._run_k8sgpt(args, timeout=timeout_s)
        if res["rc"] != 0:
            msg = f"k8sgpt failed (rc={res['rc']}): {res['err']}".strip()
            logger.error(msg)
            return {"content": [{"type": "text", "text": msg}]}

        if res["out"]:
            logger.info("k8sgpt JSON output: %s", res["out"])

        try:
            payload = json.loads(res["out"])
        except json.JSONDecodeError:
            # Some builds print extra header lines; try to extract JSON object
            logger.warning("k8sgpt output not pure JSON; returning raw text")
            summary = {"status": "UNKNOWN", "offenders": []}
            out_text = res["out"]
        else:
            summary = self._summarize_allow_priv_escalation(payload)
            out_text = json.dumps(payload, indent=2)

        # Compose final response text
        lines = [
            "Control: containers must not allow privilege escalation (allowPrivilegeEscalation=false)",
            f"Status: {summary['status']}",
        ]
        if summary["offenders"]:
            lines.append("Offending objects:")
            for ident in summary["offenders"]:
                lines.append(f"  - {ident}")
        else:
            lines.append("No offending objects found by k8sgpt.")

        kubectl_report = await self._kubectl_privilege_escalation_report(namespace, timeout_s)

        if kubectl_report.get("errors"):
            lines.append("")
            lines.append("kubectl detected pods with allowPrivilegeEscalation enabled:")
            for offender in kubectl_report["errors"]:
                if not isinstance(offender, dict):
                    lines.append(f"  - {offender}")
                    continue
                offender_text = (
                    f"  - namespace={offender.get('namespace')}, pod={offender.get('pod')}, "
                    f"container={offender.get('container')}, allowPrivilegeEscalation={offender.get('allowPrivilegeEscalation')}"
                )
                lines.append(offender_text)
            lines.append("")
            lines.append("k8sgpt-formatted error report:")
            lines.append(json.dumps(kubectl_report, indent=2))
        else:
            lines.append("")
            lines.append("kubectl did not detect pods with allowPrivilegeEscalation enabled.")
            lines.append("k8sgpt-formatted report:")
            lines.append(json.dumps(kubectl_report, indent=2))

        if include_raw:
            lines.append("")
            lines.append("Raw k8sgpt JSON:")
            lines.append(out_text)

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @staticmethod
    def _fake_k8sgpt_payload(args: List[str]) -> Dict[str, Any]:
        """Return a canned k8sgpt response used when fake mode is enabled."""
        return {
            "command": {
                "executable": "k8sgpt",
                "arguments": args,
            },
            "results": [],
            "summary": {
                "status": "PASS",
                "message": "No privilege escalation findings (fake response)",
            },
            "generated_at": datetime.now().isoformat(),
        }

    # ---------- MCP plumbing ----------

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Initialize request from client: %s", params.get('clientInfo', {}))
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities,
            "serverInfo": self.server_info
        }

    def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Listing available tools")
        return {"tools": self.tools}

    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        logger.info("Tool called: %s args=%s", tool_name, arguments)

        if tool_name == "get_current_time":
            return self._get_current_time(arguments)

        if tool_name == "k8sgpt_analyze_pod_security":
            return await self._k8sgpt_analyze_pod_security(arguments)

        raise ValueError(f"Unknown tool: {tool_name}")

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming MCP message and return response."""
        try:
            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")

            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_list_tools(params)
            elif method == "tools/call":
                result = await self.handle_call_tool(params)
            else:
                raise ValueError(f"Unknown method: {method}")

            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            logger.error("Error handling message: %s", e, exc_info=True)
            return {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32603, "message": str(e)}}


# Global server instance
mcp_server = MCPTimeServer()

async def sse_endpoint(request: Request) -> StreamingResponse:
    """SSE endpoint for MCP protocol."""
    async def event_stream():
        try:
            body = await request.body()
            if body:
                try:
                    message = json.loads(body.decode('utf-8'))
                    logger.info("Received message: %s", json.dumps(message, indent=2))
                    response = await mcp_server.handle_message(message)
                    yield f"data: {json.dumps(response)}\n\n"
                except json.JSONDecodeError:
                    error_response = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}
                    yield f"data: {json.dumps(error_response)}\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Error in event stream: %s", e, exc_info=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )

async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "server": mcp_server.server_info,
        "timestamp": datetime.now().isoformat()
    })

async def server_info(request: Request) -> JSONResponse:
    return JSONResponse({
        "serverInfo": mcp_server.server_info,
        "capabilities": mcp_server.capabilities,
        "tools": mcp_server.tools
    })

# Starlette app
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
    parser = argparse.ArgumentParser(description='MCP Server with k8sgpt integration')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host')
    parser.add_argument('--port', type=int, default=8080, help='Bind port')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info("Starting MCP Server on %s:%d", args.host, args.port)
    logger.info("SSE endpoint: http://%s:%d/sse", args.host, args.port)
    logger.info("Health check: http://%s:%d/health", args.host, args.port)
    logger.info("Server info: http://%s:%d/info", args.host, args.port)
    logger.info("K8SGPT_BIN: %s", mcp_server.k8sgpt_bin)
    logger.info("KUBECTL_BIN: %s", mcp_server.kubectl_bin)
    logger.info("K8SGPT_FAKE_MODE: %s", mcp_server.use_fake_k8sgpt)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())

if __name__ == '__main__':
    main()
