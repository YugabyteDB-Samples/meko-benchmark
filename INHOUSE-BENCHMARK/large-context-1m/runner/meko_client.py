"""Standalone Meko MCP client, copied per this initiative's per-experiment convention."""
import json
import os
import urllib.request
import uuid

MEKO_URL = "https://mcp.mekodata.ai/mcp"


class MekoToolError(RuntimeError):
    """A tool-level refusal returned inside a successful JSON-RPC response.

    Distinct from a transport or JSON-RPC error. The body looks like
    {"error": "...", "detail": "..."} and has no `results` key, so a caller
    doing `r.get("results") or []` silently scores it as zero hits. Raising
    here makes a refusal impossible to confuse with an empty result set.
    """

    def __init__(self, tool, payload):
        self.tool = tool
        self.payload = payload
        super().__init__(f"Meko tool error from {tool}: {payload}")


def _raise_on_tool_error(tool, payload):
    if isinstance(payload, dict) and payload.get("error"):
        raise MekoToolError(tool, payload)
    return payload


class MekoMCPClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self._id = 0

    @classmethod
    def from_env(cls):
        key = os.environ.get("MEKO_API_KEY")
        if not key:
            raise RuntimeError("MEKO_API_KEY not set")
        return cls(key)

    def call_tool(self, name, arguments):
        self._id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        req = urllib.request.Request(
            MEKO_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
        # Server may respond as SSE ("event: message\ndata: {...}") or plain JSON.
        # Slower/larger requests emit leading ": ping - ..." SSE comment lines
        # before the real event, so scan for the last "data:" line rather than
        # gating on the response starting with "event:".
        payload = raw
        data_lines = [line[len("data:"):].strip() for line in raw.splitlines() if line.startswith("data:")]
        if data_lines:
            payload = data_lines[-1]
        parsed = json.loads(payload)
        if "error" in parsed:
            raise RuntimeError(f"Meko error calling {name}: {parsed['error']}")
        result = parsed.get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        if texts:
            try:
                parsed_text = json.loads(texts[0])
            except json.JSONDecodeError:
                return {"raw_text": texts[0]}
            return _raise_on_tool_error(name, parsed_text)
        return _raise_on_tool_error(name, result)


def extract_search_texts(result):
    """Pull memory text bodies out of a memory_search response, defensively."""
    texts = []
    items = result.get("results") or result.get("memories") or result.get("items") or []
    if isinstance(items, dict):
        items = items.get("results", [])
    for item in items:
        if isinstance(item, dict):
            t = item.get("memory") or item.get("text") or item.get("content")
            if t:
                texts.append(t)
    return texts


def run_id_for(conversation_id):
    return str(uuid.UUID(conversation_id))
