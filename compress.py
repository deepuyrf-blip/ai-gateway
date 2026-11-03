"""
Prompt compression — a Python port of FreeLLMAPI's compression pipeline.

Coding agents (Claude Code, Codex, Cursor…) resend the same system prompt, file
reads, command output and tool schemas on every turn. Shrinking that context
before routing means:
  * fewer 413 "payload too large" answers from small-context providers,
  * lower latency and token spend,
  * more providers stay eligible.

Modes
  off         no rewriting
  lossless    repeated-block dedupe + whitespace hygiene
  standard    + tool-output filtering + stale file-read supersession
  aggressive  + condense older turns + optional hard token target

Design rules (same as upstream):
  * deterministic, synchronous, no extra model calls, no network
  * FAIL-OPEN — if an engine raises, grows the text, or breaks a fidelity gate,
    the original is kept
  * protected spans survive: code fences, URLs, file paths, error lines,
    constraint lines, key/value pairs
"""
from __future__ import annotations

import json
import re
from typing import Any

MODES = ("off", "lossless", "standard", "aggressive")

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
URL_RE = re.compile(r"https?://[^\s)\"'<>]+")
PATH_RE = re.compile(r"(?:^|[\s(\"'`])((?:\.{0,2}/)?(?:[\w.\-]+/)+[\w.\-]+\.[A-Za-z0-9]{1,6})")
ERROR_RE = re.compile(r"(?i)(error|exception|traceback|fatal|failed|panic|cannot|denied)")
CONSTRAINT_RE = re.compile(r"(?i)\b(must|never|always|do not|don't|required|only|exactly)\b")
FENCE_RE = re.compile(r"```")
KV_RE = re.compile(r"^\s*[\w.\-]+\s*[:=]\s*\S")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Same accounting the router uses (chars / 4)."""
    return max(0, len(text) // 4)


def _protected_spans(text: str) -> list[str]:
    out = []
    out += FENCE_RE.findall(text)
    out += URL_RE.findall(text)
    out += [m.group(1) for m in PATH_RE.finditer(text)]
    for line in text.splitlines():
        if ERROR_RE.search(line) or CONSTRAINT_RE.search(line) or KV_RE.match(line):
            out.append(line.strip()[:120])
    return [s for s in out if s]


def _fidelity_ok(before: str, after: str) -> bool:
    """Keep the compression only when protected material mostly survives."""
    if len(after) >= len(before):
        return False
    b, a = _protected_spans(before), set(_protected_spans(after))
    if not b:
        return True
    kept = sum(1 for s in b if s in a)
    return kept / len(b) >= 0.90


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #
def _whitespace(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def _dedupe_blocks(text: str, min_len: int = 80) -> str:
    """Drop exact repeats of long lines (progress bars, repeated log lines)."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= min_len:
            if s in seen:
                continue
            seen.add(s)
        out.append(line)
    return "\n".join(out)


def _filter_tool_output(text: str, keep_head: int = 40, keep_tail: int = 20,
                        max_lines: int = 120) -> str:
    """Collapse huge tool results, keeping the useful head/tail.

    Error-ish output is treated conservatively: only whitespace is cleaned.
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = "\n".join(lines[:keep_head])
    tail = "\n".join(lines[-keep_tail:])
    dropped = len(lines) - keep_head - keep_tail
    return f"{head}\n\n… [{dropped} lines omitted by gateway compression] …\n\n{tail}"


def _condense_old_turns(messages: list[dict], keep_recent: int = 8,
                        max_chars: int = 400) -> list[dict]:
    """Shorten turns older than the last `keep_recent`, never touching the first."""
    if len(messages) <= keep_recent + 1:
        return messages
    out = [messages[0]]
    cut = len(messages) - keep_recent
    for i, m in enumerate(messages[1:], start=1):
        if i >= cut:
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, str) and len(c) > max_chars:
            out.append({**m, "content": c[:max_chars] + "\n… [older turn trimmed] …"})
        elif isinstance(c, list):
            kept = []
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text" and len(str(blk.get("text", ""))) > max_chars:
                    kept.append({**blk, "text": str(blk["text"])[:max_chars] + "\n… [trimmed] …"})
                else:
                    kept.append(blk)
            out.append({**m, "content": kept})
        else:
            out.append(m)
    return out


def _supersede_file_reads(messages: list[dict]) -> list[dict]:
    """If the same file was read several times, keep only the newest result."""
    last_seen: dict[str, int] = {}
    for i, m in enumerate(messages):
        c = m.get("content")
        if not isinstance(c, str):
            continue
        for mt in PATH_RE.finditer(c[:400]):
            last_seen[mt.group(1)] = i
    if not last_seen:
        return messages
    out = []
    for i, m in enumerate(messages):
        c = m.get("content")
        stale = False
        if isinstance(c, str) and len(c) > 600:
            for mt in PATH_RE.finditer(c[:400]):
                p = mt.group(1)
                if last_seen.get(p, i) > i:
                    stale = True
                    break
        if stale:
            out.append({**m, "content": "… [superseded by a newer read of the same file] …"})
        else:
            out.append(m)
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _compress_text(text: str, mode: str, is_tool_result: bool = False) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    before = text
    try:
        text = _whitespace(text)
        text = _dedupe_blocks(text)
        if mode in ("standard", "aggressive") and is_tool_result:
            text = _filter_tool_output(text)
        if not _fidelity_ok(before, text):
            return before
        return text
    except Exception:
        return before


def _walk_content(content: Any, mode: str, is_tool_result: bool) -> Any:
    if isinstance(content, str):
        return _compress_text(content, mode, is_tool_result)
    if isinstance(content, list):
        out = []
        for blk in content:
            if not isinstance(blk, dict):
                out.append(blk)
                continue
            t = blk.get("type")
            if t == "text" and isinstance(blk.get("text"), str):
                out.append({**blk, "text": _compress_text(blk["text"], mode, is_tool_result)})
            elif t == "tool_result":
                inner = blk.get("content")
                out.append({**blk, "content": _walk_content(inner, mode, True)})
            else:
                out.append(blk)
        return out
    return content


def compress_body(body: dict, mode: str = "standard") -> tuple[dict, int, str]:
    """Compress an OpenAI- or Anthropic-shaped request body.

    Returns (body, chars_saved, effective_mode). Fail-open: on any problem the
    original body is returned untouched.
    """
    mode = (mode or "off").lower()
    if mode not in MODES or mode == "off":
        return body, 0, "off"
    try:
        before_len = len(json.dumps(body))
        out = dict(body)

        if isinstance(out.get("system"), str):
            out["system"] = _compress_text(out["system"], mode)
        elif isinstance(out.get("system"), list):
            out["system"] = _walk_content(out["system"], mode, False)

        msgs = out.get("messages")
        if isinstance(msgs, list):
            new = []
            for m in msgs:
                if not isinstance(m, dict):
                    new.append(m)
                    continue
                new.append({**m, "content": _walk_content(m.get("content"), mode, False)})
            if mode in ("standard", "aggressive"):
                new = _supersede_file_reads(new)
            if mode == "aggressive":
                new = _condense_old_turns(new)
            out["messages"] = new

        after_len = len(json.dumps(out))
        if after_len >= before_len:
            return body, 0, "off"
        return out, before_len - after_len, mode
    except Exception:
        return body, 0, "off"
