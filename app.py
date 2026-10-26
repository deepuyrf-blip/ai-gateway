"""
AI Gateway — a self-hosted, multi-provider LLM router with a web dashboard.

* Add many provider API keys (free tiers or paid) from one dashboard.
* One unified key for every client: OpenAI wire (/v1/chat/completions) and
  Anthropic wire (/v1/messages) — so Claude Code, Codex, OpenCode, Aider,
  Cline, Goose and anything else can all point at the same endpoint.
* Automatic failover: a rate-limited or dead key is cooled down and the next
  key/provider is tried.
* Usage tracking per key, per provider.

Run:  python3 app.py            (serves http://0.0.0.0:7860)
Env:  PORT, GATEWAY_DATA_DIR, GATEWAY_PASSWORD
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.environ.get("GATEWAY_DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "gateway.db"
WEB_DIR = Path(__file__).parent / "web"
PORT = int(os.environ.get("PORT", "7860"))
HOST = os.environ.get("HOST", "0.0.0.0")
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "120"))
TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "120"))

# --------------------------------------------------------------------------- #
# provider catalog  (type: openai = OpenAI-compatible, anthropic = Messages API)
# --------------------------------------------------------------------------- #
PROVIDERS: list[dict[str, Any]] = [
    dict(id="google", name="Google AI Studio", type="openai",
         base="https://generativelanguage.googleapis.com/v1beta/openai",
         signup="https://aistudio.google.com/apikey",
         free="Generous Gemini Flash free tier", card=False),
    dict(id="groq", name="Groq", type="openai", base="https://api.groq.com/openai/v1",
         signup="https://console.groq.com/keys", free="~15M tokens/mo, very fast", card=False),
    dict(id="openrouter", name="OpenRouter", type="openai", base="https://openrouter.ai/api/v1",
         signup="https://openrouter.ai/keys", free=":free model routes", card=False),
    dict(id="github", name="GitHub Models", type="openai", base="https://models.inference.ai.azure.com",
         signup="https://github.com/settings/tokens", free="~18M tokens/mo (GitHub account)", card=False),
    dict(id="cloudflare", name="Cloudflare Workers AI", type="openai",
         base="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
         signup="https://dash.cloudflare.com/profile/api-tokens",
         free="~30M tokens/mo", card=False, note="key format: account_id:token"),
    dict(id="huggingface", name="Hugging Face Router", type="openai",
         base="https://router.huggingface.co/v1",
         signup="https://huggingface.co/settings/tokens", free="Small monthly credit", card=False),
    dict(id="mistral", name="Mistral", type="openai", base="https://api.mistral.ai/v1",
         signup="https://console.mistral.ai/api-keys", free="Free tier", card=False),
    dict(id="nvidia", name="NVIDIA NIM", type="openai", base="https://integrate.api.nvidia.com/v1",
         signup="https://build.nvidia.com", free="Free credits", card=False),
    dict(id="cerebras", name="Cerebras", type="openai", base="https://api.cerebras.ai/v1",
         signup="https://cloud.cerebras.ai", free="$5 credit — needs a card", card=True),
    dict(id="deepseek", name="DeepSeek", type="openai", base="https://api.deepseek.com/v1",
         signup="https://platform.deepseek.com/api_keys", free="Signup tokens (paid after)", card=False),
    dict(id="moonshot", name="Moonshot (Kimi)", type="openai", base="https://api.moonshot.ai/v1",
         signup="https://platform.moonshot.ai/console/api-keys", free="Trial credit", card=False),
    dict(id="zai", name="Z.ai (GLM)", type="openai", base="https://api.z.ai/api/paas/v4",
         signup="https://z.ai/manage-apikey/apikey-list", free="GLM free models", card=False),
    dict(id="minimax", name="MiniMax", type="openai", base="https://api.minimax.io/v1",
         signup="https://platform.minimax.io", free="Trial credit", card=False),
    dict(id="together", name="Together AI", type="openai", base="https://api.together.xyz/v1",
         signup="https://api.together.ai/settings/api-keys", free="Small signup credit", card=False),
    dict(id="llm7", name="LLM7.io", type="openai", base="https://api.llm7.io/v1",
         signup="https://llm7.io", free="100 req/hr, keyless option", card=False),
    dict(id="pollinations", name="Pollinations", type="openai",
         base="https://text.pollinations.ai/openai",
         signup="https://pollinations.ai", free="Keyless (rate-limited)", card=False),
    dict(id="anthropic", name="Anthropic", type="anthropic", base="https://api.anthropic.com",
         signup="https://console.anthropic.com/settings/keys", free="Paid", card=True),
    dict(id="radoxai", name="Radox AI", type="anthropic", base="https://radoxai.com",
         signup="https://radoxai.com", free="Cheap paid (Claude-compatible)", card=True),
    dict(id="freemodel", name="FreeModel.in", type="anthropic", base="https://freemodel.in",
         signup="https://freemodel.in", free="Cheap paid (Claude-compatible)", card=True),
    dict(id="custom_openai", name="Custom — OpenAI-compatible", type="openai", base="",
         signup="", free="Your own endpoint (vLLM, LM Studio, relay…)", card=False),
    dict(id="custom_anthropic", name="Custom — Anthropic-compatible", type="anthropic", base="",
         signup="", free="Your own endpoint (Claude-compatible relay)", card=False),
]
PROVIDER_MAP = {p["id"]: p for p in PROVIDERS}

# agent setup cards shown in the dashboard
AGENTS = [
    dict(id="claude", name="Claude Code", wire="anthropic",
         cmd='export ANTHROPIC_BASE_URL="{base}"\nexport ANTHROPIC_AUTH_TOKEN="{key}"\nexport ANTHROPIC_API_KEY=""\nclaude',
         note="Base URL must be the server ROOT (no /v1)."),
    dict(id="codex", name="Codex CLI", wire="openai",
         cmd='export OPENAI_BASE_URL="{base}/v1"\nexport OPENAI_API_KEY="{key}"\ncodex',
         note="OpenAI wire — needs /v1."),
    dict(id="opencode", name="OpenCode", wire="openai",
         cmd='# ~/.config/opencode/opencode.json\n{{\n  "provider": {{\n    "gateway": {{\n      "npm": "@ai-sdk/openai-compatible",\n      "options": {{ "baseURL": "{base}/v1", "apiKey": "{key}" }}\n    }}\n  }}\n}}',
         note="OpenAI wire — needs /v1."),
    dict(id="aider", name="Aider", wire="openai",
         cmd='export OPENAI_API_BASE="{base}/v1"\nexport OPENAI_API_KEY="{key}"\naider --model openai/auto',
         note="OpenAI wire — needs /v1."),
    dict(id="cline", name="Cline / Continue", wire="openai",
         cmd='Base URL: {base}/v1\nAPI Key:  {key}',
         note="Paste into the extension settings."),
    dict(id="goose", name="Goose", wire="openai",
         cmd='export OPENAI_HOST="{base}"\nexport OPENAI_API_KEY="{key}"\nexport OPENAI_BASE_PATH="v1/chat/completions"\ngoose session',
         note="OpenAI wire — needs /v1."),
    dict(id="curl", name="cURL / SDK", wire="openai",
         cmd='curl {base}/v1/chat/completions \\\n  -H "Authorization: Bearer {key}" \\\n  -H "Content-Type: application/json" \\\n  -d \'{{"model":"auto","messages":[{{"role":"user","content":"Hi"}}]}}\'',
         note="Any OpenAI SDK works: base_url = {base}/v1"),
]

# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS keys (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            label TEXT DEFAULT '',
            secret TEXT NOT NULL,
            base_override TEXT DEFAULT '',
            enabled INTEGER DEFAULT 1,
            cooldown_until REAL DEFAULT 0,
            requests INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            last_error TEXT DEFAULT '',
            created REAL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, provider TEXT, key_id TEXT, wire TEXT,
            model TEXT, status INTEGER, ok INTEGER, ms INTEGER,
            prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
            error TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        """)


def setting(k: str, default: str = "") -> str:
    with db() as c:
        row = c.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
    return row["v"] if row else default


def set_setting(k: str, v: str) -> None:
    with db() as c:
        c.execute("INSERT INTO settings(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def hash_pw(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()


def bootstrap_token() -> str:
    """One-time token printed to the logs, needed to create the first account."""
    tok = setting("bootstrap_token")
    if not tok:
        tok = secrets.token_hex(4).upper()
        set_setting("bootstrap_token", tok)
        print(f"\n  ============================================\n"
              f"   FIRST-RUN SETUP CODE:  {tok}\n"
              f"  ============================================\n", flush=True)
    return tok


def unified_key() -> str:
    k = setting("unified_key")
    if not k:
        k = "sk-gw-" + secrets.token_hex(20)
        set_setting("unified_key", k)
    return k


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
app = FastAPI(title="AI Gateway", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    unified_key()
    bootstrap_token()


def require_session(request: Request) -> None:
    """Dashboard API auth: session cookie set at login."""
    if not setting("password_hash"):
        raise HTTPException(401, "setup required")
    tok = request.cookies.get("gw_session") or request.headers.get("x-gw-session", "")
    if not tok or not hmac_compare(tok, setting("session_token")):
        raise HTTPException(401, "unauthorized")


def hmac_compare(a: str, b: str) -> bool:
    return hmac_compare_raw(a.encode(), b.encode())


def hmac_compare_raw(a: bytes, b: bytes) -> bool:
    if not a or not b:
        return False
    return hashlib.sha256(a).digest() == hashlib.sha256(b).digest()


def require_unified(request: Request) -> str:
    """Client auth for /v1/* — Bearer <unified key> (or any key you minted)."""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("x-api-key", "")
    if not token:
        raise HTTPException(401, "missing api key")
    valid = {unified_key()} | {r["secret"] for r in with_db("SELECT secret FROM keys")}
    if token not in valid:
        raise HTTPException(401, "invalid api key")
    return token


def with_db(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    with db() as c:
        return c.execute(sql, args).fetchall()


# --------------------------------------------------------------------------- #
# dashboard API
# --------------------------------------------------------------------------- #
class SetupIn(BaseModel):
    code: str
    password: str


class LoginIn(BaseModel):
    password: str


class KeyIn(BaseModel):
    provider: str
    secret: str
    label: str = ""
    base_override: str = ""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    f = WEB_DIR / "index.html"
    return HTMLResponse(f.read_text(encoding="utf-8") if f.exists() else "<h1>AI Gateway</h1>")


@app.get("/api/status")
def api_status() -> dict:
    return {
        "setup_required": not bool(setting("password_hash")),
        "unified_key": unified_key() if setting("password_hash") else "",
    }


@app.post("/api/setup")
def api_setup(body: SetupIn, response: Response) -> dict:
    if setting("password_hash"):
        raise HTTPException(400, "already set up")
    if body.code.strip().upper() != bootstrap_token():
        raise HTTPException(403, "invalid setup code")
    if len(body.password) < 6:
        raise HTTPException(400, "password must be at least 6 characters")
    salt = secrets.token_hex(8)
    set_setting("password_salt", salt)
    set_setting("password_hash", hash_pw(body.password, salt))
    session = secrets.token_hex(24)
    set_setting("session_token", session)
    set_setting("bootstrap_token", "")  # burn it
    response.set_cookie("gw_session", session, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return {"ok": True, "unified_key": unified_key()}


@app.post("/api/login")
def api_login(body: LoginIn, response: Response) -> dict:
    if not setting("password_hash"):
        raise HTTPException(400, "setup required")
    if hash_pw(body.password, setting("password_salt")) != setting("password_hash"):
        raise HTTPException(401, "wrong password")
    session = secrets.token_hex(24)
    set_setting("session_token", session)
    response.set_cookie("gw_session", session, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return {"ok": True}


@app.post("/api/logout")
def api_logout(response: Response) -> dict:
    set_setting("session_token", "")
    response.delete_cookie("gw_session")
    return {"ok": True}


@app.get("/api/state")
def api_state(_: None = Depends(require_session)) -> dict:
    keys = [dict(r) for r in with_db("SELECT * FROM keys ORDER BY created")]
    for k in keys:
        k["secret_masked"] = (k["secret"][:6] + "…" + k["secret"][-4:]) if len(k["secret"]) > 12 else "•••"
        k.pop("secret", None)
        k["cooling"] = k["cooldown_until"] > time.time()

    since = time.time() - 86400
    stats = with_db("""
        SELECT COUNT(*) n, COALESCE(SUM(ok),0) ok,
               COALESCE(SUM(prompt_tokens),0) pt, COALESCE(SUM(completion_tokens),0) ct
        FROM events WHERE ts > ?""", (since,))[0]
    by_provider = [dict(r) for r in with_db("""
        SELECT provider, COUNT(*) n, COALESCE(SUM(ok),0) ok,
               COALESCE(SUM(prompt_tokens+completion_tokens),0) tokens
        FROM events WHERE ts > ? GROUP BY provider ORDER BY n DESC""", (since,))]
    recent = [dict(r) for r in with_db("""
        SELECT ts, provider, model, status, ok, ms, error FROM events
        ORDER BY id DESC LIMIT 40""")]

    return {
        "providers": PROVIDERS,
        "agents": AGENTS,
        "keys": keys,
        "unified_key": unified_key(),
        "stats": {"requests": stats["n"], "ok": stats["ok"],
                  "prompt_tokens": stats["pt"], "completion_tokens": stats["ct"],
                  "by_provider": by_provider},
        "recent": recent,
        "base_url": os.environ.get("PUBLIC_BASE_URL", ""),
    }


@app.post("/api/keys")
def api_add_key(body: KeyIn, _: None = Depends(require_session)) -> dict:
    if body.provider not in PROVIDER_MAP:
        raise HTTPException(400, "unknown provider")
    if not body.secret.strip():
        raise HTTPException(400, "secret required")
    kid = uuid.uuid4().hex[:12]
    with db() as c:
        c.execute("INSERT INTO keys(id,provider,label,secret,base_override,created) VALUES(?,?,?,?,?,?)",
                  (kid, body.provider, body.label, body.secret.strip(), body.base_override.strip(), time.time()))
    return {"ok": True, "id": kid}


@app.patch("/api/keys/{kid}")
def api_patch_key(kid: str, body: dict, _: None = Depends(require_session)) -> dict:
    allowed = {"enabled", "label", "cooldown_until", "base_override"}
    sets, args = [], []
    for k, v in body.items():
        if k in allowed:
            sets.append(f"{k}=?")
            args.append(int(bool(v)) if k == "enabled" else v)
    if not sets:
        return {"ok": False}
    args.append(kid)
    with db() as c:
        c.execute(f"UPDATE keys SET {', '.join(sets)} WHERE id=?", tuple(args))
    return {"ok": True}


@app.delete("/api/keys/{kid}")
def api_del_key(kid: str, _: None = Depends(require_session)) -> dict:
    with db() as c:
        c.execute("DELETE FROM keys WHERE id=?", (kid,))
    return {"ok": True}


@app.post("/api/rotate-unified")
def api_rotate(_: None = Depends(require_session)) -> dict:
    k = "sk-gw-" + secrets.token_hex(20)
    set_setting("unified_key", k)
    return {"ok": True, "unified_key": k}


@app.post("/api/test/{kid}")
async def api_test(kid: str, _: None = Depends(require_session)) -> dict:
    rows = with_db("SELECT * FROM keys WHERE id=?", (kid,))
    if not rows:
        raise HTTPException(404, "key not found")
    key = dict(rows[0])
    prov = PROVIDER_MAP[key["provider"]]
    base = key["base_override"] or prov["base"]
    try:
        if prov["type"] == "anthropic":
            url = base.rstrip("/") + "/v1/messages"
            headers = {"x-api-key": key["secret"], "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            payload = {"model": "claude-3-5-haiku-latest", "max_tokens": 8,
                       "messages": [{"role": "user", "content": "ping"}]}
        else:
            url = base.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {key['secret']}", "content-type": "application/json"}
            payload = {"model": "auto", "max_tokens": 8,
                       "messages": [{"role": "user", "content": "ping"}]}
        async with httpx.AsyncClient(timeout=30) as cx:
            r = await cx.post(url, headers=headers, json=payload)
        ok = r.status_code < 400
        with db() as c:
            c.execute("UPDATE keys SET last_error=?, cooldown_until=? WHERE id=?",
                      ("" if ok else f"{r.status_code} {r.text[:180]}", 0 if ok else 0, kid))
        return {"ok": ok, "status": r.status_code, "body": r.text[:400]}
    except Exception as e:
        with db() as c:
            c.execute("UPDATE keys SET last_error=? WHERE id=?", (str(e)[:200], kid))
        return {"ok": False, "status": 0, "body": str(e)[:300]}


# --------------------------------------------------------------------------- #
# routing core
# --------------------------------------------------------------------------- #
def pick_keys(wire: str) -> list[dict]:
    """Enabled, not-cooling keys, preferring providers that speak this wire."""
    rows = [dict(r) for r in with_db("SELECT * FROM keys WHERE enabled=1 ORDER BY created")]
    now = time.time()
    alive = [r for r in rows if r["cooldown_until"] <= now]
    same = [r for r in alive if PROVIDER_MAP[r["provider"]]["type"] == wire]
    other = [r for r in alive if r not in same]
    return same + other


def log_event(**kw: Any) -> None:
    with db() as c:
        c.execute("""INSERT INTO events(ts,provider,key_id,wire,model,status,ok,ms,
                     prompt_tokens,completion_tokens,error)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                  (time.time(), kw.get("provider", ""), kw.get("key_id", ""), kw.get("wire", ""),
                   kw.get("model", ""), kw.get("status", 0), int(bool(kw.get("ok"))),
                   kw.get("ms", 0), kw.get("prompt_tokens", 0), kw.get("completion_tokens", 0),
                   (kw.get("error") or "")[:400]))


def cool_down(kid: str, reason: str) -> None:
    with db() as c:
        c.execute("UPDATE keys SET cooldown_until=?, failures=failures+1, last_error=? WHERE id=?",
                  (time.time() + COOLDOWN_SECONDS, reason[:300], kid))


def bump(kid: str) -> None:
    with db() as c:
        c.execute("UPDATE keys SET requests=requests+1 WHERE id=?", (kid,))


def resolve_url(key: dict) -> tuple[str, dict]:
    prov = PROVIDER_MAP[key["provider"]]
    base = (key["base_override"] or prov["base"]).rstrip("/")
    secret = key["secret"]
    if prov["id"] == "cloudflare" and ":" in secret:
        acct, token = secret.split(":", 1)
        base = base.replace("{account_id}", acct)
        secret = token
    if prov["type"] == "anthropic":
        return base + "/v1/messages", {"x-api-key": secret, "anthropic-version": "2023-06-01"}
    return base + "/chat/completions", {"Authorization": f"Bearer {secret}"}


async def call_upstream(wire: str, payload: dict) -> tuple[int, dict]:
    """Try each key in order until one answers. Returns (status, body)."""
    keys = pick_keys(wire)
    if not keys:
        return 503, {"error": {"message": "no keys configured — add a provider key in the dashboard"}}

    model = payload.get("model") or "auto"
    last_err = ""
    for key in keys:
        prov = PROVIDER_MAP[key["provider"]]
        url, auth = resolve_url(key)
        headers = {**auth, "content-type": "application/json"}
        body = dict(payload)
        if prov["type"] == "openai" and body.get("model") in (None, "", "auto"):
            body.pop("model", None)

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
                r = await cx.post(url, headers=headers, json=body)
            ms = int((time.time() - t0) * 1000)
        except Exception as e:
            cool_down(key["id"], str(e))
            log_event(provider=prov["id"], key_id=key["id"], wire=wire, model=model,
                      status=0, ok=False, ms=int((time.time() - t0) * 1000), error=str(e))
            last_err = str(e)
            continue

        if r.status_code in (401, 402, 403, 429, 500, 502, 503, 529):
            cool_down(key["id"], f"{r.status_code}: {r.text[:150]}")
            log_event(provider=prov["id"], key_id=key["id"], wire=wire, model=model,
                      status=r.status_code, ok=False, ms=ms, error=r.text[:200])
            last_err = f"{r.status_code} {r.text[:150]}"
            continue

        try:
            data = r.json()
        except Exception:
            data = {"error": {"message": r.text[:300]}}

        bump(key["id"])
        usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        log_event(provider=prov["id"], key_id=key["id"], wire=wire, model=model,
                  status=r.status_code, ok=r.status_code < 400, ms=ms,
                  prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                  completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                  error="" if r.status_code < 400 else r.text[:200])
        if r.status_code < 400:
            data["_gateway_provider"] = prov["id"]
            return r.status_code, data
        last_err = f"{r.status_code} {r.text[:150]}"

    return 502, {"error": {"message": f"all providers failed. last: {last_err}"}}


# --------------------------------------------------------------------------- #
# wire translation
# --------------------------------------------------------------------------- #
def anthropic_to_openai(req: dict) -> dict:
    msgs: list[dict] = []
    sys_prompt = req.get("system")
    if sys_prompt:
        if isinstance(sys_prompt, list):
            sys_prompt = "\n".join(b.get("text", "") for b in sys_prompt if isinstance(b, dict))
        msgs.append({"role": "system", "content": sys_prompt})

    for m in req.get("messages", []):
        role, content = m.get("role", "user"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        text_parts, tool_calls, tool_results = [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                text_parts.append(b.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id"), "type": "function",
                                   "function": {"name": b.get("name"),
                                                "arguments": json.dumps(b.get("input", {}))}})
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                tool_results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                     "content": str(c)})
        if tool_results:
            msgs.extend(tool_results)
        if text_parts or tool_calls:
            m2: dict[str, Any] = {"role": role, "content": "\n".join(text_parts) or None}
            if tool_calls:
                m2["tool_calls"] = tool_calls
            msgs.append(m2)

    out: dict[str, Any] = {"messages": msgs}
    if req.get("max_tokens"):
        out["max_tokens"] = req["max_tokens"]
    for k_src, k_dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if req.get(k_src) is not None:
            out[k_dst] = req[k_src]
    if req.get("stop_sequences"):
        out["stop"] = req["stop_sequences"]
    if req.get("tools"):
        out["tools"] = [{"type": "function",
                         "function": {"name": t.get("name"), "description": t.get("description", ""),
                                      "parameters": t.get("input_schema", {"type": "object"})}}
                        for t in req["tools"]]
    return out


def openai_to_anthropic(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: list[dict] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {"raw": fn.get("arguments")}
        blocks.append({"type": "tool_use", "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:12],
                       "name": fn.get("name"), "input": args})
    if not blocks:
        blocks = [{"type": "text", "text": ""}]

    finish = choice.get("finish_reason")
    stop = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}.get(finish, "end_turn")
    usage = data.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex[:20],
        "type": "message", "role": "assistant", "model": model,
        "content": blocks, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                  "output_tokens": int(usage.get("completion_tokens") or 0)},
    }


def anthropic_sse(payload: dict) -> list[str]:
    """Turn a complete Anthropic message into the SSE event stream clients expect."""
    def ev(name: str, obj: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(obj)}\n\n"

    out = [ev("message_start", {"type": "message_start", "message": {**payload, "content": [],
                                                                   "stop_reason": None}})]
    for i, blk in enumerate(payload.get("content", [])):
        out.append(ev("content_block_start", {"type": "content_block_start", "index": i,
                                              "content_block": {"type": blk["type"], **({"text": ""} if blk["type"] == "text" else {"id": blk.get("id"), "name": blk.get("name"), "input": {}})}}))
        if blk["type"] == "text":
            out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                  "delta": {"type": "text_delta", "text": blk.get("text", "")}}))
        elif blk["type"] == "tool_use":
            out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                  "delta": {"type": "input_json_delta",
                                                            "partial_json": json.dumps(blk.get("input", {}))}}))
        out.append(ev("content_block_stop", {"type": "content_block_stop", "index": i}))
    out.append(ev("message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": payload.get("stop_reason"),
                                              "stop_sequence": None},
                                    "usage": payload.get("usage", {})}))
    out.append(ev("message_stop", {"type": "message_stop"}))
    return out


# --------------------------------------------------------------------------- #
# client-facing endpoints
# --------------------------------------------------------------------------- #
@app.post("/v1/chat/completions")
async def v1_chat(request: Request, _: str = Depends(require_unified)):
    body = await request.json()
    stream = bool(body.get("stream"))
    status, data = await call_upstream("openai", body)
    if status >= 400:
        return JSONResponse(data, status_code=status)
    if not stream:
        return JSONResponse(data)

    async def gen():
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        base = {"id": data.get("id", "chatcmpl-" + uuid.uuid4().hex[:10]),
                "object": "chat.completion.chunk", "created": int(time.time()),
                "model": data.get("model", body.get("model", ""))}
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        if msg.get("content"):
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': msg['content']}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': ch.get('finish_reason') or 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/messages")
async def v1_messages(request: Request, _: str = Depends(require_unified)):
    req = await request.json()
    model = req.get("model", "claude-3-5-sonnet-latest")
    keys = pick_keys("anthropic")
    native = [k for k in keys if PROVIDER_MAP[k["provider"]]["type"] == "anthropic"]

    if native:
        status, data = await call_upstream("anthropic", req)
    else:
        # translate to OpenAI, call an OpenAI-wire provider, translate back
        status, data = await call_upstream("openai", anthropic_to_openai(req))
        if status < 400:
            data = openai_to_anthropic(data, model)

    if status >= 400:
        return JSONResponse(data, status_code=status)

    if req.get("stream"):
        return StreamingResponse(iter(anthropic_sse(data)), media_type="text/event-stream")
    return JSONResponse(data)


@app.get("/v1/models")
def v1_models(_: str = Depends(require_unified)) -> dict:
    data = [{"id": "auto", "object": "model", "owned_by": "gateway"}]
    for k in with_db("SELECT DISTINCT provider FROM keys WHERE enabled=1"):
        data.append({"id": f"{k['provider']}/auto", "object": "model", "owned_by": k["provider"]})
    return {"object": "list", "data": data}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "keys": len(with_db("SELECT id FROM keys")),
            "setup": bool(setting("password_hash"))}


if __name__ == "__main__":
    import uvicorn

    init_db()
    unified_key()
    bootstrap_token()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
