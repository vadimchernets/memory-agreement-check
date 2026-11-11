"""Model backends.

Spec strings:
  cli:codex[:MODEL]        OpenAI models through a ChatGPT subscription (`codex exec`, read-only sandbox)
  cli:claude[:MODEL]       Anthropic models through a Claude subscription (`claude -p`, all tools disabled)
  cli:kimi[:MODEL]         Moonshot models through a Kimi subscription (`kimi -p`)
  api:openai:MODEL         OpenAI Chat Completions API, key in OPENAI_API_KEY (OPENAI_BASE_URL optional)
  api:anthropic:MODEL      Anthropic Messages API, key in ANTHROPIC_API_KEY
  dry:MODE                 offline stubs for tests: oracle | abstain | noisy | oracle-if-retrieved
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

FAMILY = {"codex": "OpenAI", "openai": "OpenAI", "claude": "Anthropic", "anthropic": "Anthropic",
          "kimi": "Moonshot", "dry": "offline"}


@dataclass
class Reply:
    text: str
    model: str
    seconds: float = 0.0
    tool_used: bool = False
    meta: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    pass


class Backend:
    spec = ""
    family = ""

    def ask(self, prompt: str, item=None) -> Reply:  # item: dict with gold/coverage, used by dry stubs only
        raise NotImplementedError


_SEMS: dict[str, threading.Semaphore] = {}


def _sem(key: str, n: int) -> threading.Semaphore:
    if key not in _SEMS:
        _SEMS[key] = threading.Semaphore(n)
    return _SEMS[key]


def _bin(name: str) -> str:
    p = os.environ.get(f"MEMCHECK_{name.upper()}_BIN") or shutil.which(name)
    if not p:
        raise BackendError(f"`{name}` CLI not found on PATH (or set MEMCHECK_{name.upper()}_BIN)")
    return p


class CliBackend(Backend):
    def __init__(self, tool: str, model: str | None, workdir: str, max_concurrent: int = 2,
                 timeout: int = 300, effort: str = "low"):
        self.tool, self.model, self.workdir = tool, model, workdir
        self.timeout, self.effort = timeout, effort
        self.family = FAMILY[tool]
        self.spec = f"cli:{tool}" + (f":{model}" if model else "")
        self.sem = _sem(f"cli:{tool}", max_concurrent)

    def _cmd(self, prompt: str):
        b = _bin(self.tool)
        if self.tool == "codex":
            cmd = [b, "exec", "--skip-git-repo-check", "-s", "read-only", "-C", self.workdir, "--json",
                   "-c", f"model_reasoning_effort={self.effort}"]
            if self.model:
                cmd += ["-m", self.model]
            return cmd + [prompt], None, None
        if self.tool == "claude":
            cmd = [b, "-p", "--model", self.model or "sonnet", "--setting-sources", "", "--tools", "",
                   "--strict-mcp-config", "--no-session-persistence", "--add-dir", self.workdir,
                   "--output-format", "json"]
            env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
            env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
            return cmd, prompt, env
        if self.tool == "kimi":
            cmd = [b, "-p", prompt, "--add-dir", self.workdir, "--output-format", "stream-json"]
            if self.model:
                cmd += ["-m", self.model]
            return cmd, None, None
        raise BackendError(f"unknown cli tool {self.tool}")

    def _parse(self, out: str) -> tuple[str, str, bool]:
        if self.tool == "codex":
            text, tool_used = "", False
            for line in out.splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                it = ev.get("item") or {}
                if it.get("type") == "agent_message":
                    text = it.get("text", "")
                elif it.get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                    tool_used = True
            return text.strip(), self.model or "codex-default", tool_used
        if self.tool == "claude":
            d = json.loads(out)
            if d.get("is_error"):
                raise BackendError(f"claude error: {str(d.get('result'))[:300]}")
            model = ",".join(d.get("modelUsage", {}).keys()) or (self.model or "sonnet")
            return str(d.get("result", "")).strip(), model, False
        # kimi --output-format stream-json: one JSON object per line; the reply is the last assistant text
        text, tool_used = "", False
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("role") == "tool" or ev.get("tool_calls"):
                tool_used = True
            if ev.get("role") == "assistant":
                c = ev.get("content")
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
                if isinstance(c, str) and c.strip():
                    text = c
        return text.strip(), self.model or "kimi-default", tool_used

    def ask(self, prompt: str, item=None) -> Reply:
        cmd, stdin, env = self._cmd(prompt)
        last = ""
        for attempt in range(4):
            with self.sem:
                t0 = time.time()
                try:
                    p = subprocess.run(cmd, input=stdin if stdin is not None else None,
                                       stdin=None if stdin is not None else subprocess.DEVNULL,
                                       capture_output=True, text=True, timeout=self.timeout, env=env,
                                       cwd=self.workdir)
                    dt = time.time() - t0
                    if p.returncode == 0:
                        text, model, tool_used = self._parse(p.stdout)
                        if text:
                            return Reply(text=text, model=model, seconds=round(dt, 2), tool_used=tool_used)
                        last = "empty reply"
                    else:
                        last = f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}"
                except subprocess.TimeoutExpired:
                    last = f"timeout after {self.timeout}s"
                except (ValueError, BackendError) as e:
                    last = str(e)
            time.sleep([10, 30, 90, 0][attempt])
        raise BackendError(f"{self.spec}: {last}")


class ApiBackend(Backend):
    def __init__(self, provider: str, model: str, max_concurrent: int = 4, timeout: int = 120):
        if not model:
            raise BackendError(f"api:{provider} needs a model name, e.g. api:{provider}:<model>")
        self.provider, self.model, self.timeout = provider, model, timeout
        self.family = FAMILY[provider]
        self.spec = f"api:{provider}:{model}"
        self.sem = _sem(f"api:{provider}", max_concurrent)

    def _request(self, prompt: str):
        if self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY") or ""
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            return f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, body, key
        key = os.environ.get("ANTHROPIC_API_KEY") or ""
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        body = {"model": self.model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
        return f"{base}/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, key

    def ask(self, prompt: str, item=None) -> Reply:
        url, headers, body, key = self._request(prompt)
        if not key:
            raise BackendError(f"{self.spec}: API key not set in the environment")
        last = ""
        for attempt in range(5):
            with self.sem:
                t0 = time.time()
                try:
                    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                                 headers={"content-type": "application/json", **headers})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        d = json.loads(r.read())
                    dt = time.time() - t0
                    if self.provider == "openai":
                        text = d["choices"][0]["message"]["content"] or ""
                    else:
                        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
                    return Reply(text=text.strip(), model=d.get("model", self.model), seconds=round(dt, 2))
                except urllib.error.HTTPError as e:
                    last = f"HTTP {e.code}: {e.read()[:300]!r}"
                    if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                        break
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                    last = str(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise BackendError(f"{self.spec}: {last}")


class DryBackend(Backend):
    """Deterministic offline stand-ins, for tests and for trying the pipeline without any model."""

    def __init__(self, mode: str, name: str = ""):
        self.mode, self.family = mode, "offline"
        self.spec = f"dry:{mode}" + (f":{name}" if name else "")

    def _coin(self, key: str) -> float:
        return int(hashlib.sha256(f"{self.spec}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def ask(self, prompt: str, item=None) -> Reply:
        item = item or {}
        gold, cov, qid = item.get("gold", ""), item.get("coverage", "unknown"), item.get("qid", prompt)
        if self.mode == "oracle":
            text = gold
        elif self.mode == "abstain":
            text = "Not mentioned"
        elif self.mode == "noisy":
            text = gold if self._coin(qid) < 0.7 else "something else entirely"
        elif self.mode == "oracle-if-retrieved":
            # right when memory holds the evidence; otherwise the same plausible wrong answer
            # (the first name in the excerpts), as two models reading one bad memory might give
            if cov == "full":
                text = gold
            else:
                m = re.search(r"\] ([^:]+):", prompt)
                text = m.group(1) if m else "Not mentioned"
        else:
            raise BackendError(f"unknown dry mode {self.mode}")
        return Reply(text=text, model=self.spec)


def make(spec: str, workdir: str, max_concurrent: int = 2, effort: str = "low") -> Backend:
    parts = spec.split(":", 2)
    kind = parts[0]
    if kind == "cli" and len(parts) >= 2:
        return CliBackend(parts[1], parts[2] if len(parts) > 2 and parts[2] else None, workdir,
                          max_concurrent=max_concurrent, effort=effort)
    if kind == "api" and len(parts) >= 2:
        return ApiBackend(parts[1], parts[2] if len(parts) > 2 else "", max_concurrent=max(4, max_concurrent))
    if kind == "dry" and len(parts) >= 2:
        return DryBackend(parts[1], parts[2] if len(parts) > 2 else "")
    raise BackendError(f"bad backend spec {spec!r}; see `python -m memcheck run --help`")


PRESETS = {
    # model A, model B, judge (a third company where possible)
    "cli": ("cli:codex", "cli:claude:sonnet", "cli:kimi"),
    # model names change often, so the api preset reads them from the environment
    "api": ("api:openai:" + os.environ.get("MEMCHECK_OPENAI_MODEL", ""),
            "api:anthropic:" + os.environ.get("MEMCHECK_ANTHROPIC_MODEL", ""), "rule"),
    "dry": ("dry:oracle-if-retrieved:a", "dry:oracle-if-retrieved:b", "rule"),
}
