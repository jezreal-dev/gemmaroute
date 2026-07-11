# GemmaRoute

**4-Layer AMD-Native AI Routing Engine.** Cut your LLM API bill by up to 80% without losing quality.

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-E8001C?logo=amd)](https://www.amd.com/en/developer/rocm.html)
[![Gemma](https://img.shields.io/badge/Google-Gemma%202-8B5CF6?logo=google)](https://deepmind.google/technologies/gemma/)
[![Fireworks AI](https://img.shields.io/badge/Fireworks-AI-3B82F6)](https://fireworks.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-blueviolet)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

| Resource | URL |
|---|---|
| Frontend | https://gemmaroute.vercel.app |
| Frontend Repo | https://github.com/jezreal-dev/gemmaroute-frontend |
| Backend Repo | https://github.com/jezreal-dev/gemmaroute |

---

## Message to Judges

- Watch the 3-minute demo video first.
- Test the live Vercel link directly. If the backend tunnel is offline, the UI automatically falls back to a mock that correctly simulates all 4 routing tiers.
- To run locally: add your own Fireworks API key to `backend/.env` before running Docker.
- **Gemma Bonus:** Gemma 2 powers all 3 intelligence layers — classifier, executor, and quality judge.

---

## Quick Start (Docker)

```bash
# 1. Clone and configure
git clone https://github.com/jezreal-dev/gemmaroute.git
cd gemmaroute

# 2. Set environment variables
cp .env.example backend/.env
# Edit backend/.env — add FIREWORKS_API_KEY and API_KEY

# 3. Start the full stack (Ollama + backend + dashboard)
docker compose up --build -d

# 4. Test the API
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gemmaroute-demo-2026" \
  -d '{"prompt": "What are your business hours?"}'
```

---

## Running Locally Without Docker

Use this to run the backend directly with Python and expose it via Ngrok.

### Prerequisites

- Python 3.11+
- A [Fireworks AI](https://fireworks.ai) account with a serverless API key
- [Ngrok](https://ngrok.com/download) installed and authenticated

### Step 1 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Create the environment file

Create `backend/.env`:

```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
API_KEY=gemmaroute-demo-2026
OLLAMA_HOST=http://localhost:11434
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
```

### Step 3 — Start the backend

Open Terminal 1:

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
INFO:  SQLite database initialised.
INFO:  Uvicorn running on http://0.0.0.0:8000
```

### Step 4 — Start the Ngrok tunnel

Open Terminal 2:

```bash
# Windows
.\ngrok.exe http 8000

# macOS / Linux
ngrok http 8000
```

Copy the HTTPS forwarding URL from the Ngrok output.

### Step 5 — Verify the tunnel

```bash
curl https://YOUR-NGROK-URL/health
```

Expected response:
```json
{"status": "ok", "ollama": "unreachable", "fireworks_circuit": {"state": "CLOSED"}, "db": "ok"}
```

### Step 6 — Connect the Vercel frontend

1. Vercel dashboard → gemmaroute project → Settings → Environment Variables
2. Set `NEXT_PUBLIC_API_URL` to your Ngrok HTTPS URL (no trailing slash)
3. Set `NEXT_PUBLIC_API_KEY` to `gemmaroute-demo-2026`
4. Deployments → Redeploy

The frontend header will show a green "live backend" dot once connected.

Note: Free Ngrok URLs change on every restart. Update the Vercel env var each time.

### Keep the machine awake (Windows)

```powershell
powercfg /change standby-timeout-ac 0
```

---

## Architecture

```
User Prompt
     |
     v
+----------------------------------------------------------+
|  LAYER 0 - Pre-Classifier Signal Check  (~0ms, $0.00)   |
|  Instant vocab scan for high-signal words.              |
|  "legal", "SLA", "fraud"  -> force complex              |
|  "refund", "billing"      -> set medium floor           |
+----------------------------------------------------------+
     | no strong signal
     v
+----------------------------------------------------------+
|  LAYER 1a - Heuristic Filter  (<1ms, $0.00)             |
|  Regex rules for greetings, hours, thanks, goodbye.     |
|  Zero LLM calls. Returns a canned response instantly.   |
+----------------------------------------------------------+
     | no regex match
     v
+----------------------------------------------------------+
|  LAYER 1b - Gemma 2B Classifier  (~300ms, $0.00, AMD)   |
|  Classifies into: simple / medium / complex             |
|  Confidence gate: score < 0.62 -> bump tier up          |
|  Signal floor: result never drops below Layer 0 tier    |
+----------------------------------------------------------+
     |
     +- simple  -> Gemma 2B via Ollama (AMD local)    $0.00/req
     +- medium  -> DeepSeek V4 Flash via Fireworks    ~$0.00006/req
     +- complex -> DeepSeek V4 Pro via Fireworks      ~$0.00040/req
                          |
                          v
+----------------------------------------------------------+
|  LAYER 3 - Gemma 2B Quality Judge  (~300ms, $0.00, AMD) |
|  Scores response 0.0-1.0 on a 4-criteria rubric.        |
|  score >= 0.75 -> return response                       |
|  score < 0.75  -> escalate to next tier (max 2 hops)    |
+----------------------------------------------------------+
     |
     v
  Logger -> SQLite -> Dashboard
```

### Resilience Patterns

| Pattern | What it does |
|---|---|
| Circuit Breaker | Trips after 3 Fireworks failures. Reroutes to local Ollama for 60s then auto-resets. |
| Exponential Backoff | Retries at 1s, 2s, 4s before declaring failure. |
| Hop Budget | Hard cap of 2 escalations per request. |
| Confidence Gate | Classifier confidence below 0.62 bumps tier up one level. |
| Signal Floor | Pre-classifier result sets a minimum tier the LLM cannot go below. |
| Ollama Fallback | If Fireworks fails, local Gemma handles the request. |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI 0.115 |
| Routing Engine | LangGraph StateGraph 0.2.45 |
| Local Models | Gemma 2B (gemma2:2b, gemma:2b) via Ollama |
| Cloud Models | DeepSeek V4 Flash + DeepSeek V4 Pro via Fireworks AI |
| Database | SQLite + SQLAlchemy async 2.0.36 |
| Observability | Streamlit dashboard |
| Frontend | Next.js 16, React 19, Tailwind v4 |
| Containerisation | Docker Compose |
| Tests | pytest + pytest-asyncio (52 tests) |

---

## API Reference

All endpoints except `/health` and `/docs` require the `X-API-Key` header.

### POST /route

**Request:**
```json
{
  "prompt": "I have a legal SLA violation regarding section 4B of my contract.",
  "session_id": "demo-001",
  "max_cost_tier": "complex"
}
```

**Response:**
```json
{
  "response": "I understand this is a serious matter...",
  "routing": {
    "initial_tier": "complex",
    "final_tier": "complex",
    "escalations": 0,
    "classifier_confidence": 0.97,
    "quality_score": 0.88,
    "model_used": "accounts/fireworks/models/deepseek-v4-pro",
    "latency_ms": 25700,
    "estimated_cost_usd": 0.00040,
    "cost_saved_vs_max_usd": 0.0
  },
  "session_id": "demo-001"
}
```

**Routing examples:**

| Prompt | Tier | Cost |
|---|---|---|
| "Hello!" | trivial | $0.00 |
| "What are your business hours?" | trivial | $0.00 |
| "Where is my order?" | simple | $0.00 |
| "I want to return an item" | medium | ~$0.00006 |
| "I have a legal SLA violation" | complex | ~$0.00040 |

### GET /stats

Returns aggregate routing statistics.

### GET /health

No auth required. Returns backend status and circuit breaker state.

---

## Environment Variables

**Required (`backend/.env`):**

```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
API_KEY=gemmaroute-demo-2026
OLLAMA_HOST=http://localhost:11434
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
```

**Optional overrides:**

```env
CLOUD_MEDIUM_MODEL=accounts/fireworks/models/deepseek-v4-flash
CLOUD_COMPLEX_MODEL=accounts/fireworks/models/deepseek-v4-pro
QUALITY_THRESHOLD=0.75
MAX_ESCALATION_DEPTH=2
```

---

## AMD ROCm Deployment

Switch from CPU to AMD ROCm in `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:rocm
  devices:
    - /dev/kfd
    - /dev/dri
```

---

## Running Tests

```bash
python -m pytest tests/ -v --asyncio-mode=auto
```

52 tests, all passing. No live services required.

---

## Project Structure

```
gemmaroute/
├── backend/
│   ├── clients/
│   │   ├── fireworks_client.py   # Circuit breaker + exponential backoff
│   │   └── ollama_client.py      # Hybrid classifier + quality judge
│   ├── graph/
│   │   ├── builder.py            # LangGraph StateGraph assembly
│   │   ├── edges.py              # Conditional routing functions
│   │   ├── nodes.py              # All 9 node implementations
│   │   └── state.py              # AgentState TypedDict
│   ├── routers/
│   │   ├── route_endpoint.py     # POST /route
│   │   └── stats_endpoint.py     # GET /stats, GET /health
│   ├── services/
│   │   └── cost_estimator.py     # Cost and savings calculation
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── dashboard/
├── tests/
│   ├── test_api.py
│   └── test_graph.py
├── docker-compose.yml
└── README.md
```

---

## FAQ

**Do I need to run this locally?**
No. The live Vercel frontend is at gemmaroute.vercel.app. The backend is served via Ngrok during the judging period.

**Why does the frontend show "demo mode"?**
The backend Ngrok tunnel is not active. The frontend falls back to a local mock that routes prompts correctly based on vocabulary scoring.

**What Fireworks models are used?**
Medium tier: DeepSeek V4 Flash. Complex tier: DeepSeek V4 Pro. Both are serverless on the Fireworks platform.

**How does the Gemma Bonus apply?**
Gemma 2 is used in 3 roles: classifier (gemma2:2b), simple-tier executor (gemma:2b), and quality judge (gemma:2b). All run locally on AMD hardware at zero cloud cost.

---

## Contact

jezrealmomoh1234@gmail.com

---

## License

MIT. See [LICENSE](LICENSE).
