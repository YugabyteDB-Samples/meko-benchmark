#!/usr/bin/env python3
"""Unified coding-client caller with per-call token and cost accounting.

Every client here is a real coding CLI or a cloud model endpoint, called the
way an engineer would call it. Large prompts go over STDIN wherever the CLI
supports it, because the compaction stage feeds 200k-character chunks and
argv is not a safe place for those.

Usage accounting differs per provider and each result says which basis it is
on, so nothing downstream can silently add a metered token count to a
CLI-reported one:

  api       the model's own counts (Ollama prompt_eval_count / eval_count)
  cli_json  a CLI's self-reported usage, which includes that CLI's system
            prefix (~12-23k tokens per call) and any helper-model calls it
            made on its own initiative
  cli_text  scraped from CLI stdout (codex prints "tokens used N"); input and
            output are not separated
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request

CLAUDE = os.environ.get("XC1_CLAUDE", "claude")
AGY = os.environ.get("XC1_AGY", "agy")
CODEX = os.environ.get("XC1_CODEX", "codex")
JUNIE = os.environ.get("XC1_JUNIE", "junie")
OLLAMA_CHAT = os.environ.get("XC1_OLLAMA", "http://127.0.0.1:11434/api/chat")

# label -> (provider, model). The roster under test.
ROSTER = {
    "codex-terra":  ("codex",  "gpt-5.6-terra"),
    "codex-sol":    ("codex",  "gpt-5.6-sol"),
    "codex-luna":   ("codex",  "gpt-5.6-luna"),
    "codex-55":     ("codex",  "gpt-5.5"),
    "opus5":        ("claude", "claude-opus-5"),
    "sonnet5":      ("claude", "claude-sonnet-5"),
    "gemini-37":    ("agy",    "gemini-3.7-flash-low"),
    "glm":          ("ollama", "glm-5.2:cloud"),
    "minimax":      ("ollama", "minimax-m2.7:cloud"),
}


def _usage(inp=0, out=0, cache_read=0, cache_write=0, cost=0.0, basis="none"):
    return {"input_tokens": inp, "output_tokens": out,
            "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
            "billed_input_tokens": inp + cache_read + cache_write,
            "total_tokens": inp + cache_read + cache_write + out,
            "cost_usd": round(cost, 6), "basis": basis}


def isolated_env() -> dict:
    """Remove benchmark paths and Meko credentials from reader calls."""
    return {key: value for key, value in os.environ.items()
            if not key.startswith("XC1_") and not key.startswith("MEKO_")}


def _run(cmd, stdin_text=None, timeout=1800, cwd=None, env=None):
    return subprocess.run(
        cmd, input=stdin_text if stdin_text is not None else "",
        text=True, capture_output=True, timeout=timeout, cwd=cwd, env=env)


def call(provider: str, model: str, prompt: str, timeout: int = 1800,
         workdir: str | None = None, cwd: str | None = None,
         env: dict | None = None) -> dict:
    """Return {ok, text, usage, latency_s} — never raises for a model refusal."""
    t0 = time.monotonic()
    run_options = {"cwd": cwd, "env": env}

    def done(ok, text="", usage=None, err=None):
        return {"ok": ok, "text": text, "usage": usage or _usage(),
                "latency_s": round(time.monotonic() - t0, 2),
                **({"error": err} if err else {})}

    try:
        if provider == "claude":
            cmd = [CLAUDE, "-p", "--model", model, "--output-format", "json",
                   "--max-turns", "1", "--tools", ""]
            proc = _run(cmd, prompt, timeout, **run_options)
            if proc.returncode:
                return done(False, err=f"claude {proc.returncode}: {proc.stderr[-300:]}")
            payload = json.loads(proc.stdout)
            if payload.get("is_error"):
                return done(False, err=str(payload)[:300])
            u = payload.get("usage", {})
            return done(True, payload.get("result", ""),
                        _usage(u.get("input_tokens", 0), u.get("output_tokens", 0),
                               u.get("cache_read_input_tokens", 0),
                               u.get("cache_creation_input_tokens", 0),
                               payload.get("total_cost_usd", 0), "cli_json"))

        if provider == "codex":
            # "-" makes codex read the prompt from stdin, which argv cannot hold.
            cmd = [CODEX, "exec", "--skip-git-repo-check",
                   "--dangerously-bypass-approvals-and-sandbox", "-m", model, "-"]
            proc = _run(cmd, prompt, timeout, **run_options)
            if proc.returncode:
                return done(False, err=f"codex {proc.returncode}: {proc.stderr[-300:]}")
            # In stdin ("-") mode codex writes the clean reply to stdout and
            # puts the transcript plus "tokens used N" on stderr.
            m = re.search(r"tokens used\s+([\d,]+)", proc.stderr)
            total = int(m.group(1).replace(",", "")) if m else 0
            return done(True, proc.stdout.strip(),
                        _usage(total, 0, basis="cli_text"))

        if provider == "agy":
            cmd = [AGY, "--dangerously-skip-permissions", "--new-project",
                   "--output-format", "json", "--model", model, "--print", prompt]
            proc = _run(cmd, "", timeout, **run_options)
            if proc.returncode:
                return done(False, err=f"agy {proc.returncode}: {proc.stderr[-300:]}")
            payload = json.loads(proc.stdout)
            if payload.get("status") != "SUCCESS":
                return done(False, err=f"agy status {payload.get('status')}")
            u = payload.get("usage", {})
            return done(True, payload.get("response", ""),
                        _usage(u.get("input_tokens", 0),
                               u.get("output_tokens", 0) + u.get("thinking_tokens", 0),
                               u.get("cache_read_tokens", 0), 0, 0, "cli_json"))

        if provider == "junie":
            cmd = [JUNIE, "--output-format", "json", "--skip-update-check",
                   "--model", model, "--task", prompt]
            if workdir:
                cmd[1:1] = ["-p", workdir]
            proc = _run(cmd, "", timeout, **run_options)
            if proc.returncode:
                return done(False, err=f"junie {proc.returncode}: {proc.stderr[-300:]}")
            line = next((l for l in proc.stdout.splitlines()
                         if l.strip().startswith("{")), "")
            if not line:
                return done(False, err=f"junie no json: {proc.stdout[-300:]}")
            payload = json.loads(line)
            # Junie orchestrates helper models per task; bill all of them.
            inp = out_t = cr = cw = 0
            cost = 0.0
            for entry in payload.get("llmUsage", []):
                inp += entry.get("inputTokens", 0)
                out_t += entry.get("outputTokens", 0)
                cr += entry.get("cacheInputTokens", 0)
                cw += entry.get("cacheCreateTokens", 0)
                cost += entry.get("cost", 0) or 0
            return done(True, payload.get("result", ""),
                        _usage(inp, out_t, cr, cw, cost, "cli_json"))

        if provider == "ollama":
            body = json.dumps({"model": model, "stream": False, "think": False,
                               "messages": [{"role": "user", "content": prompt}],
                               "options": {"temperature": 0, "num_ctx": 131072}}).encode()
            req = urllib.request.Request(
                OLLAMA_CHAT, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            text = ((payload.get("message") or {}).get("content") or "").strip()
            if not text:
                return done(False, err=f"empty: {str(payload)[:200]}")
            return done(True, text,
                        _usage(payload.get("prompt_eval_count", 0),
                               payload.get("eval_count", 0), basis="api"))

        return done(False, err=f"unknown provider {provider}")

    except subprocess.TimeoutExpired:
        return done(False, err=f"timeout after {timeout}s")
    except Exception as e:  # noqa: BLE001
        return done(False, err=f"{type(e).__name__}: {str(e)[:300]}")


def call_label(label: str, prompt: str, **kw) -> dict:
    provider, model = ROSTER[label]
    return call(provider, model, prompt, **kw)
