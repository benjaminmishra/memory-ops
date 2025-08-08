1. Get familiar with the codebase
Extract the ZIP (memory-ops.zip) and inspect its structure. You should see:

Dockerfile, README.md, requirements.txt

app/ directory with modules:

main.py – FastAPI entrypoint exposing /v1/chat/completions.

auth.py – supports API key and Google ID token auth.

rate_limit.py – sliding‑window request/token limiter.

memory.py – simple SQLite message store via SQLAlchemy.

compression.py – wrapper around services/qr_retriever.py.

upstream.py – forwards requests to any OpenAI‑compatible endpoint.

config.py – Pydantic settings for all env variables.

services/qr_retriever.py – implements query‑focused context compression.

tests/ currently only has test_auth.py with thorough auth tests.

Review auth.py to understand how Google Sign-In works. The module lazily imports google-auth libraries, verifies the ID token when GOOGLE_CLIENT_ID is set, and falls back to API‑key validation. Errors raise HTTPException.

Read main.py carefully. It orchestrates identity check, rate limiting, memory retrieval, compression, upstream call, and memory persistence. This understanding is critical for writing integration tests.

2. Set up your development environment
Create a virtual environment:

sh
Copy
python -m venv venv
source venv/bin/activate
Install dependencies:

sh
Copy
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-benchmark httpx ruff
# google-auth is already in requirements but install if missing:
pip install google-auth google-auth-oauthlib google-auth-httplib2
Run the existing test suite to ensure it passes:

sh
Copy
pytest -q
3. Add missing unit tests
Use the tests/test_auth.py pattern as a template. Each test file should live in tests/ and start with test_. Use pytest fixtures and monkeypatch to adjust environment variables and patch heavy dependencies.

3.1 test_rate_limit.py
Goal: Verify the sliding‑window algorithm enforces both request and token limits.

What to test:

When limits are high, RateLimiter.check() does not raise.

When request count exceeds requests_per_minute, it raises HTTPException with status 429.

When token count exceeds tokens_per_minute, it raises HTTPException.

Hints:

Instantiate RateLimiter(requests_per_minute=2, tokens_per_minute=100).

Call check(identity, tokens, now) multiple times with simulated timestamps using time.time() or manual values.

Use pytest.raises(HTTPException) to assert exceptions.

3.2 test_memory.py
Goal: Ensure messages are stored and retrieved correctly.

What to test:

add_message() writes to the database.

get_messages() returns messages in order.

get_context() concatenates previous messages into a string (check token boundaries are correct).

clear_session() wipes data.

Hints:

Before importing app.memory, set os.environ["DATABASE_URL"] = "sqlite:///./test.db" or sqlite:///:memory: and call importlib.reload(app.memory) to use a fresh DB.

Use get_settings().cache_clear() to reset Pydantic settings between tests.

3.3 test_compression.py
Goal: Ensure compression.compress() delegates to qr_retriever.reduce().

What to test:

Patch services.qr_retriever.reduce to a stub that returns (“condensed text”, 200, 50) and confirm compress() returns those values.

Verify that top_k environment variable flows through settings.

Hints:

Use monkeypatch to set environment variables and patch functions.

3.4 test_upstream.py
Goal: Confirm upstream.call_llm() correctly forwards requests.

What to test:

Patch httpx.AsyncClient.post to return a dummy JSON/dict and ensure call_llm() returns that.

Verify that it sets Authorization header from UPSTREAM_API_KEY if present.

For streaming, patch httpx.AsyncClient.post to return an async generator and ensure call_llm() yields messages.

4. Write functional/integration tests
These tests exercise the full FastAPI stack using httpx.AsyncClient. They should patch heavy pieces to avoid loading models or hitting the network.

4.1 test_main.py
Set up: Use FastAPI app instance imported from app.main to spin up a test client.

Mock:

app.compression.compress → return a fixed condensed context and token counts.

app.upstream.call_llm → return a dummy assistant message (either JSON or streaming).

Test scenarios:

Successful call with valid API key:

Set API_KEYS = "testkey" in env; call POST /v1/chat/completions with header X-API-Key: testkey and JSON body {"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}]}.

Assert response status 200.

Assert memory storage: call get_context() to check the user message and assistant reply saved.

Assert headers x-tokens-before, x-tokens-after, x-tokens-saved.

Rate limit hit:

Configure low requests_per_minute=1.

Make two quick calls and expect second to return status 429.

Invalid auth:

Omit API key and expect 401 when API_KEYS is set.

Google auth (optional):

Set GOOGLE_CLIENT_ID and patch verify_oauth2_token to return a payload with sub field.

Send Authorization: Bearer validtoken and ensure call succeeds and identity begins with google:.

4.2 Integration with SQLite
For an end‑to‑end integration test, avoid patching memory to ensure DB writes succeed. Use sqlite:///:memory: or a temp file DB. Mock only app.compression.compress and app.upstream.call_llm to keep the test fast.

5. Benchmark tests
Create tests/test_benchmarks.py:

Use pytest-benchmark to measure compression.compress() while patching qr_retriever.reduce to a no-op. Example:

python
Copy
def test_compress_benchmark(benchmark, monkeypatch):
    monkeypatch.setattr(app.services.qr_retriever, "reduce", lambda q, c, top_k: ("", 1000, 10))
    benchmark(lambda: compress(query="q", context="a"*4000))
This ensures that the compression wrapper’s overhead remains low. You might also benchmark the rate limiter’s check().

Pytest will automatically include these benchmarks if pytest-benchmark is installed.

6. Continuous Integration (CI)
Add .github/workflows/ci.yml to run linting and tests. A typical CI could:

yaml
Copy
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-benchmark ruff
      - run: ruff check .
      - run: pytest --benchmark-min-rounds=1
You can adjust the benchmark rounds for quicker runs.

7. Update the README
Document the new Google authentication flow:

Explain how to set GOOGLE_CLIENT_ID.

Show example curl requests using Authorization: Bearer <id_token>.

Outline how to configure OAuth 2.0 client in Google Cloud and obtain ID tokens (link to Google docs).

Document how to run tests (pytest command) and how to run the benchmark tests.

Mention that heavy model loads are patched in tests.

8. Validation and launch
Local test run:

sh
Copy
export API_KEYS=testkey
export UPSTREAM_BASE=https://api.openai.com
export UPSTREAM_API_KEY=sk-...
uvicorn app.main:app --reload
Send test POST requests to http://localhost:8000/v1/chat/completions. Verify headers and storage.

For deployment, use the provided Dockerfile:

sh
Copy
docker build -t memory-ops .
docker run -p 8000:8000 -e API_KEYS=testkey -e UPSTREAM_BASE=... -e UPSTREAM_API_KEY=... memory-ops


9.Add GitHub Actions
Create .github/workflows/ci.yml (for tests) and .github/workflows/deploy.yml (for Fly deploy):

yaml
Copy
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
  push:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-benchmark ruff
      - run: ruff check .
      - run: pytest --benchmark-min-rounds=1
yaml
Copy
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pip install flyctl
      - run: |
          echo "$FLY_API_TOKEN" | flyctl auth login --access-token -
          flyctl deploy --remote-only --ha=false
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
You will need to store your Fly API token as a GitHub secret (FLY_API_TOKEN). The flyctl deploy command uses the existing Dockerfile.

3. Configure Fly.io deployment
Install and authenticate:

sh
Copy
curl -L https://fly.io/install.sh | sh  # Install flyctl:contentReference[oaicite:1]{index=1}
fly auth login                            # Log in and select your organization
Generate config:
Inside the project directory:

sh
Copy
fly launch --name memory-ops --image-name memory-ops --no-deploy
This creates fly.toml. Answer prompts (region, scaling) as desired; choose a region near Stockholm (e.g., arn). The --no-deploy flag ensures it only generates config.

Customise fly.toml: Set concurrency and environment variables:

toml
Copy
[deploy]
  release_command = "python -m app.migrate"  # if needed; else remove

[env]
  PORT = "8000"
  MODEL_NAME = "mistralai/Mistral-7B-v0.2"
  LORA_ID = "qrhead/mistral-7b-retriever"
  TOP_K = "64"
  DEVICE = "cuda"

[[mounts]]
  source = "db_storage"
  destination = "/data"
Create a volume to persist the SQLite DB:

sh
Copy
fly volumes create db_storage --region arn --size 1
Set secrets for API keys and Google client ID:

sh
Copy
fly secrets set API_KEYS=your-key-list \
  UPSTREAM_BASE=https://api.openai.com \
  UPSTREAM_API_KEY=sk-... \
  GOOGLE_CLIENT_ID=your-google-client-id
Deploy:

sh
Copy
fly deploy --ha=false
This builds the Docker image (using the existing Dockerfile) and starts the app. Fly will allocate a GPU machine since the base image includes CUDA. You can scale down concurrency via fly scale count if needed.

Verify:

Check logs with fly logs.

Test the endpoint with a cURL or HTTP request:

sh
Copy
curl -H "X-API-Key: your-key" -X POST https://<app-name>.fly.dev/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}'
📘 README updates
Document environment variables (API_KEYS, UPSTREAM_BASE, UPSTREAM_API_KEY, GOOGLE_CLIENT_ID) and Fly-specific variables.

Add a section “Deploy to Fly.io” describing the flyctl steps, secret management, and scaling.

Provide a snippet showing how to test the deployed API.

📌 Final notes
The existing Google authentication support must be tested thoroughly (mock verify_oauth2_token in unit and integration tests).

For streaming responses, ensure upstream.call_llm() yields tokens properly; adjust tests accordingly.

Keep the code modular to facilitate adding features like semantic caching, cost budgeting, or RAG pipelines later.