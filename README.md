# AI Gateway — one key, every provider, every agent

A self-hosted **multi-provider LLM router** with a clean dark dashboard.
Add API keys for as many providers as you like (free tiers or paid), and get
**one unified key** that works with Claude Code, Codex, OpenCode, Aider, Cline,
Goose — anything that speaks the OpenAI or Anthropic wire.

```
            ┌──────────────────────────────────────────────┐
  clients → │  AI Gateway :7860                            │
            │   /v1/chat/completions   (OpenAI wire)       │
            │   /v1/messages           (Anthropic wire)    │──→ Google · Groq · OpenRouter
            │   router: load-balance + failover + cooldown │──→ Cloudflare · HF · GitHub Models
            │   dashboard: providers · keys · agents       │──→ Anthropic · Radox · FreeModel
            └──────────────────────────────────────────────┘
```

## Why

- **One endpoint, many providers.** Stop wiring a new base URL for every tool.
- **Automatic failover.** A key that returns `401/402/403/429/5xx` is cooled
  down and the next key/provider is tried — transparently.
- **Both wires.** Claude Code needs the Anthropic Messages API; most other
  agents need OpenAI chat-completions. This gateway speaks **both**, and
  translates between them (including tool calls + streaming SSE).
- **Usage you can see.** Requests, tokens and failures per provider and per key.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/deepuyrf-blip/ai-gateway/main/install.sh | bash
```

Or by hand:

```bash
git clone https://github.com/deepuyrf-blip/ai-gateway
cd ai-gateway
docker compose up -d --build
```

Or without Docker (Python 3.10+):

```bash
pip install -r requirements.txt
python3 app.py            # http://localhost:7860
```

## First run

1. Open `http://localhost:7860`.
2. The **setup code** is printed in the logs:
   ```bash
   docker logs ai-gateway | grep -A1 'SETUP CODE'
   ```
   (without Docker: it's in the terminal where `app.py` started)
3. Enter the code + a password (6+ chars).
4. **Providers → Add key** for each provider you use.
5. Copy the **unified key** from **Settings**.

## Connect your agents

| Agent | Base URL | Wire |
|---|---|---|
| **Claude Code** | `http://localhost:7860` — **root, no `/v1`** | Anthropic |
| **Codex CLI** | `http://localhost:7860/v1` | OpenAI |
| **OpenCode** | `http://localhost:7860/v1` | OpenAI |
| **Aider** | `http://localhost:7860/v1` | OpenAI |
| **Cline / Continue** | `http://localhost:7860/v1` | OpenAI |
| **Goose** | `http://localhost:7860` + base path `v1/chat/completions` | OpenAI |
| **cURL / SDK** | `http://localhost:7860/v1` | OpenAI |

The **Agents** page in the dashboard generates copy-paste commands with your
base URL and unified key already filled in.

### Claude Code

```bash
export ANTHROPIC_BASE_URL="http://localhost:7860"
export ANTHROPIC_AUTH_TOKEN="sk-gw-..."
export ANTHROPIC_API_KEY=""
claude
```

### Any OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:7860/v1", api_key="sk-gw-...")
client.chat.completions.create(model="auto", messages=[{"role":"user","content":"Hi"}])
```

## Providers

21 built-in entries, including:

`Google AI Studio` · `Groq` · `OpenRouter` · `GitHub Models` ·
`Cloudflare Workers AI` · `Hugging Face` · `Mistral` · `NVIDIA NIM` ·
`Cerebras` · `DeepSeek` · `Moonshot (Kimi)` · `Z.ai (GLM)` · `MiniMax` ·
`Together AI` · `LLM7` · `Pollinations` · `Anthropic` · `Radox AI` ·
`FreeModel.in` · **Custom (OpenAI-compatible)** · **Custom (Anthropic-compatible)**

- **Free tiers:** most entries need **no credit card** — see the badge on each
  provider card in the dashboard.
- **Bring your own keys.** This project never issues tokens; it routes the
  accounts *you* connect. Each provider's own limits and terms still apply.
- **Multiple keys per provider** are supported — they're pooled and rotated, and
  each key's usage is tracked separately.
- **Custom endpoints** let you point at vLLM, LM Studio, Ollama, or any relay.
  Cloudflare's credential format is `account_id:token`.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | unified key | OpenAI wire (streaming supported) |
| `POST` | `/v1/messages` | unified key | Anthropic wire (streaming supported) |
| `GET` | `/v1/models` | unified key | `auto` + `<provider>/auto` |
| `GET` | `/health` | none | liveness |
| `GET` | `/` | session | dashboard |
| `GET/POST` | `/api/*` | session cookie | dashboard API |

Use `"model": "auto"` to let the router pick the best available key, or
`"model": "<provider>/auto"` to pin a provider.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `PORT` | `7860` | listen port |
| `GATEWAY_DATA_DIR` | `./data` | SQLite location |
| `COOLDOWN_SECONDS` | `120` | how long a failing key is benched |
| `UPSTREAM_TIMEOUT` | `120` | upstream request timeout (s) |
| `PUBLIC_BASE_URL` | *(auto)* | shown in the Agents page when behind a proxy |

## Notes & limits

- Streaming is emitted by the gateway: upstream calls are non-streaming, then
  re-streamed as OpenAI/Anthropic SSE. Functionally identical for clients, and
  it keeps the translation path simple.
- Anthropic→OpenAI translation covers text, tools and tool results. Very
  exotic content blocks (images in tool results, thinking blocks) are passed
  through only on native Anthropic providers.
- The dashboard session is a cookie; the unified key is the credential that
  matters for clients. Rotate it any time from Settings.

## Deploy anywhere

- **VPS** — `docker compose up -d --build`
- **Hugging Face Spaces** — Docker SDK, `app_port: 7860` (data is ephemeral)
- **Railway / Render / Fly.io** — Dockerfile is standard

## License

MIT
