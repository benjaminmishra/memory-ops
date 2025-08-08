# MemoryOps – Context Compression & Memory Layer for LLM Agents

MemoryOps is a minimal, yet fully featured gateway for Large Language Models (LLMs).
It acts as a drop‑in replacement for the OpenAI API, providing a unified endpoint,
token‑aware rate limiting, API key authentication, cost tracking and—most
importantly—transparent context compression and persistent memory.  The
gateway sits between your application and any upstream LLM (OpenAI, Azure,
Anthropic, etc.), trimming away irrelevant tokens to save you money, while
retaining the information that matters for accurate responses.

This repository implements a monorepo with the core service, its configuration,
tests, Docker support and a GitHub Action for continuous integration.  You can
run it locally, deploy it to Fly.io or any other cloud provider and extend it
with your own compression or memory logic.

## Features

* **Unified `/v1/chat/completions` endpoint** – mimics the OpenAI Chat API.
* **API key authentication** – simple `X‑API‑Key` header validated against
  configured keys.
* **Token‑aware rate limiting** – configurable tokens‑per‑minute budget per key.
* **Context compression** – a pluggable "QR‑HEAD" module that uses
  LoRA‑patched transformer heads to score each token and keep only the
  top‑K most relevant.  If LoRA weights are not provided, a simple
  summarisation fallback is used.
* **Persistent memory store** – conversation history is stored in a local
  SQLite database and can be retrieved between requests.
* **Upstream LLM passthrough** – forwards requests to the configured LLM
  provider, optionally forwarding the caller’s `Authorization` header.
* **Cost and token logging** – usage statistics and compression ratios
  reported via response headers.

## Quick start

### Requirements

* Python 3.10+
* Access to an upstream LLM provider (e.g. OpenAI) with an API key

Install dependencies and run the service locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Make a request using curl:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Authorization: Bearer YOUR_OPENAI_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo", 
    "messages": [{"role": "user", "content": "Tell me a joke about cats."}]
  }'
```

The response will include headers such as `x-tokens-before`, `x-tokens-after`
and `x-tokens-saved` indicating how much context was trimmed.

### Configuration

Configuration is provided via environment variables.  See
`app/config.py` for defaults.  Important variables include:

| Variable             | Description                                          |
|----------------------|------------------------------------------------------|
| `API_KEYS`           | Comma‑separated list of allowed API keys             |
| `UPSTREAM_BASE`      | Base URL of the upstream LLM provider                |
| `UPSTREAM_MODEL`     | Default model name to call upstream                  |
| `UPSTREAM_API_KEY`   | Fallback API key for upstream calls                  |
| `GOOGLE_CLIENT_ID`   | OAuth client ID for Google Sign-In validation         |
| `TOP_K`              | Number of tokens to keep after compression           |
| `RATE_LIMIT_TPM`     | Tokens‑per‑minute budget per API key                 |
| `LORA_ID`            | HuggingFace ID or local path to LoRA weights         |
| `MODEL_NAME`         | Name of the base model used for QR‑HEAD scoring      |

## Project structure

```
memory-ops/
├── README.md               – this file
├── requirements.txt        – Python dependencies
├── Dockerfile              – container build definition
├── .github/workflows/
│   ├── ci.yml              – lint & test workflow
│   └── deploy.yml          – sample deployment workflow (can be adapted)
├── app/                    – service source code
│   ├── main.py             – FastAPI entry point
│   ├── config.py           – environment configuration
│   └── services/
│       ├── qr_retriever.py – QR‑HEAD context compressor
│       ├── compression.py  – top‑level API for compressing text
│       ├── memory.py       – SQLite memory store
│       ├── auth.py         – API key auth & rate limit logic
│       ├── rate_limit.py   – simple token‑based rate limiter
│       └── upstream.py     – upstream LLM caller
└── tests/
    ├── test_compression.py
    └── test_memory.py
```

## Development & testing

Run the test suite:

```bash
pytest -q
```

The tests cover the compression logic and memory store.  To
write more tests, add new files under the `tests/` directory.

## Docker

A `Dockerfile` is provided to build a container image.  It installs
the dependencies, copies the code and sets the default command to run
the FastAPI server.  Build and run the image as follows:

```bash
docker build -t memory-ops:latest .
docker run -p 8000:8000 -e API_KEYS="my-key" -e UPSTREAM_API_KEY="sk-..." memory-ops:latest
```

## Deploy to Fly.io

1. Install and authenticate Flyctl:
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```
2. Generate an app config (does not deploy yet):
```bash
fly launch --name memory-ops --image-name memory-ops --no-deploy
```
Edit `fly.toml` to set environment variables and mounts.
3. Create a volume for persistent storage:
```bash
fly volumes create db_storage --region <region> --size 1
```
4. Set required secrets:
```bash
fly secrets set API_KEYS=... UPSTREAM_BASE=https://api.openai.com   UPSTREAM_API_KEY=... GOOGLE_CLIENT_ID=...
```
5. Deploy:
```bash
fly deploy --ha=false
```
After deployment, test the API:
```bash
curl -H "X-API-Key: your-key" -X POST https://<app-name>.fly.dev/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}'
```

## GitHub Actions

Two workflows are included under `.github/workflows/`:

* **ci.yml** – runs linting (ruff) and tests on each push.
* **deploy.yml** – demonstrates how to build a Docker image and deploy it
  using Fly.io’s `flyctl`.  Adjust it to your environment or remove if not
  needed.

## Extending MemoryOps

MemoryOps was designed with extensibility in mind.  The QR‑HEAD module in
`app/services/qr_retriever.py` can be replaced with your own compression
strategy.  The upstream caller in `app/services/upstream.py` can be
extended to support additional providers (e.g. Azure, Anthropic) by
adding model‑specific routing and API keys.

Feel free to open issues or pull requests if you find bugs or have
suggestions.  Happy hacking!