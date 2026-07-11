# GemmaRoute

**3-Layer AMD-Native AI Routing Engine.** Cut your LLM API bill by up to 80% without losing quality.

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
| Frontend (Vercel) | https://gemmaroute.vercel.app |
| Frontend Repo | https://github.com/Nickysantus/gemmaroute |
| Backend Repo | https://github.com/jezreal-dev/gemmaroute |

---

## Message to Judges

- Watch our 3-minute demo video first for the full live experience.
- The local routing engine is powered by AMD hardware using ROCm architecture.
- Test the live Vercel link directly. If our local AMD server is offline during grading, the UI falls back to intelligent simulated data automatically.
- To run locally: provide your own Fireworks API key in `backend/.env` before running Docker.
- **Gemma Bonus:** Gemma 2 is used in all 3 layers — classifier, executor, and quality judge.

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/jezreal-dev/gemmaroute.git
cd gemmaroute

# 2. Set environment variables
cp .env.example backend/.env
# Edit backend/.env and fill in FIREWORKS_API_KEY and API_KEY

# 3. Launch the full stack (Ollama + backend + dashboard)
docker compose up --build -d

# 4. Open the Streamlit dashboard
start http://localhost:8501

# 5. Test the API
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gemmaroute-demo-2026" \
  -d '{"prompt": "What are your business hours?"}'
```

---

## Running Locally Without Docker (Development / Demo Mode)

Use this if you want to run the backend directly with Python and expose it
publicly via Ngrok so the Vercel frontend can connect to it.

### Prerequisites

- Python 3.11+ installed
- A [Fireworks AI](https://fireworks.ai) account with an API key
- [Ngrok](https://ngrok.com/download) installed and authenticated

### Step 1 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Create the environment file

Create `backend/.env` with the following:

```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
API_KEY=gemmaroute-demo-2026
OLLAMA_HOST=http://localhost:11434
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
```

> The `FIREWORKS_API_KEY` must be a serverless-enabled key from your Fireworks
> account. Get one at fireworks.ai → Settings → API Keys.

### Step 3 — Start the backend

Open **Terminal 1** and run:

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for this output before moving on:
```
✅ SQLite database initialised.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4 — Expose via Ngrok tunnel

Open **Terminal 2** (leave Terminal 1 running) and run:

```bash
# Windows
.\ngrok.exe http 8000

# macOS / Linux
ngrok http 8000
```

Ngrok will display a forwarding URL like:
```
Forwarding   https://xxxx-xx-xx-xxx-xxx.ngrok-free.app -> http://localhost:8000
```

Copy that HTTPS URL — this is your live backend URL.

### Step 5 — Verify the tunnel works

```bash
curl https://YOUR-NGROK-URL/health
# Expected: {"status":"ok","ollama":"unreachable","fireworks_circuit":{"state":"CLOSED",...},"db":"ok"}
```

### Step 6 — Connect the Vercel frontend

Tell your Vercel project the backend URL:

1. Go to **vercel.com** → your gemmaroute project → **Settings → Environment Variables**
2. Set `NEXT_PUBLIC_API_URL` to your Ngrok HTTPS URL (no trailing slash)
3. Set `NEXT_PUBLIC_API_KEY` to `gemmaroute-demo-2026`
4. Go to **Deployments** → click **Redeploy** on the latest deployment

Once redeployed, open `gemmaroute.vercel.app` — the header should show a
green **"live backend"** dot confirming the frontend is connected.

> **Important:** The free Ngrok URL changes every time you restart Ngrok.
> If you restart the tunnel, repeat Step 6 with the new URL.
> Keep both terminals open for the duration of your demo session.

### Keeping the Machine Awake (Windows)

Prevent the laptop from sleeping and killing the tunnel during a demo:

```powershell
# Disable sleep (run in PowerShell)
powercfg /change standby-timeout-ac 0

# Re-enable after demo
powercfg /change standby-timeout-ac 30
```

---

## Architecture

GemmaRoute implements a 4-stage routing waterfall. Every stage uses Gemma 2.

```
User Prompt
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 0 — Pre-Classifier Signal Check  (~0ms, $0.00)        │
│  Instant vocab matching for high-signal words.               │
│  "legal", "SLA", "fraud"  → force complex  (skip Ollama)     │
│  "refund", "billing"      → set medium floor                 │
└──────────────────────────────────────────────────────────────┘
     │ no strong signal
     ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1a — Heuristic Filter  (<1ms, $0.00)                  │
│  Regex rules for greetings, hours, thanks, goodbye.          │
│  Zero LLM calls. Instant canned response.                    │
└──────────────────────────────────────────────────────────────┘
     │ no regex match
     ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1b — Gemma 2B Classifier  (~300ms, $0.00, AMD local)  │
│  Classifies into simple / medium / complex with confidence.  │
│  Confidence gate: score < 0.62 → bump tier up one level.     │
│  Signal floor: LLM tier can never go below pre-classifier.   │
└──────────────────────────────────────────────────────────────┘
     │
     ├─ simple  → Gemma 2B local via Ollama           ($0.00/req)
     ├─ medium  → DeepSeek V4 Flash via Fireworks AI  (~$0.00006/req)
     └─ complex → DeepSeek V4 Pro via Fireworks AI    (~$0.00040/req)
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3 — Gemma 2B Quality Judge  (~300ms, $0.00, AMD)      │
│  Scores response 0.0–1.0 on a 4-criteria rubric.             │
│  score ≥ 0.75 → return response                              │
│  score < 0.75 → escalate to next tier (max 2 escalations)    │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
  Logger → SQLite → Dashboard
```

### Resilience Patterns

| Pattern | What it does |
|---|---|
| **Circuit Breaker** | Trips OPEN after 3 Fireworks failures. All traffic falls back to local Ollama for 60s. Auto-resets. |
| **Exponential Backoff** | Tenacity retries at 1s → 2s → 4s before declaring failure. |
| **Hop Budget** | Hard cap of 2 escalations per request. Prevents infinite loops. |
| **Confidence Gate** | If Gemma classifier confidence < 0.62, bumps tier up one level. |
| **Signal Floor** | Pre-classifier detects high-signal vocab — LLM result can never go below this floor. |
| **Ollama Fallback** | If both Fireworks and circuit breaker path fail, final fallback to local Gemma with a degradation message. |

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Backend API | FastAPI | 0.115 |
| Routing Engine | LangGraph StateGraph | 0.2.45 |
| Local Models | Gemma 2B / gemma2:2b via Ollama | — |
| Cloud Models | DeepSeek V4 Flash (medium) + DeepSeek V4 Pro (complex) via Fireworks AI | — |
| Database | SQLite + SQLAlchemy async | 2.0.36 |
| Observability | Streamlit dashboard | — |
| Frontend | Next.js 16, React 19, Tailwind v4 | — |
| Containerisation | Docker Compose | — |
| Testing | pytest + pytest-asyncio | 52 tests |

---

## API Reference

All endpoints except `/health` and `/docs` require the `X-API-Key` header.

### POST /route

Routes a prompt through the full 4-layer pipeline.

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

**Tier routing examples:**

| Prompt | Layer hit | Tier | Cost |
|---|---|---|---|
| `"Hello!"` | Heuristic filter | trivial | $0.00 |
| `"What are your business hours?"` | Heuristic filter | trivial | $0.00 |
| `"Where is my order?"` | Gemma classifier | simple | $0.00 |
| `"I want to return an item"` | Signal floor → medium | medium | ~$0.00006 |
| `"I have a legal SLA violation"` | Signal check → complex (no Ollama call) | complex | ~$0.00040 |

### GET /stats

Returns aggregate routing statistics. Requires `X-API-Key` header.

```json
{
  "total_requests": 142,
  "routing_distribution": {"trivial": 45, "simple": 38, "medium": 32, "complex": 27},
  "total_cost_usd": 0.000842,
  "total_saved_vs_always_complex_usd": 89.21,
  "avg_latency_ms": 845.2,
  "avg_quality_score": 0.938,
  "escalation_rate": 0.12,
  "recent_logs": [...]
}
```

### GET /health

No auth required. Returns service status and circuit breaker state.

```json
{
  "status": "ok",
  "ollama": "reachable",
  "fireworks_circuit": {
    "state": "CLOSED",
    "failure_count": 0,
    "failure_threshold": 3,
    "seconds_until_reset": 0.0
  },
  "db": "ok"
}
```

---

## Environment Variables

Create `backend/.env` with the following:

```env
FIREWORKS_API_KEY=your_full_fireworks_api_key_here
API_KEY=gemmaroute-demo-2026
OLLAMA_HOST=http://localhost:11434
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
```

Optional overrides:

```env
LOCAL_ROUTER_MODEL=gemma2:2b
LOCAL_EXECUTOR_MODEL=gemma:2b
LOCAL_JUDGE_MODEL=gemma:2b
CLOUD_MEDIUM_MODEL=accounts/fireworks/models/deepseek-v4-flash
CLOUD_COMPLEX_MODEL=accounts/fireworks/models/deepseek-v4-pro
QUALITY_THRESHOLD=0.75
MAX_ESCALATION_DEPTH=2
```

---

## AMD Developer Cloud Deployment (ROCm)

Switch from CPU to AMD ROCm by editing `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:rocm   # swap from ollama/ollama
  devices:
    - /dev/kfd
    - /dev/dri
```

---

## Running Tests

```bash
# From project root
python -m pytest tests/ -v --asyncio-mode=auto
```

**52 tests — all pass. No live services required.** All LLM and DB calls are mocked.

Test coverage:
- Layer 0: pre-classifier signal detection (complex + medium + no-signal + case-insensitive)
- Layer 0: confidence gate (bump up, ceiling, signal floor enforcement)
- Layer 1a: heuristic filter (all 5 patterns + non-trivial cases)
- Layer 1b: classifier node (max_cost_tier cap, Ollama down fallback)
- Layer 3: quality judge (score clamping, Ollama down fallback)
- Edges: all routing decisions (filter, classifier, judge at threshold/below/ceiling/max-depth)
- Escalation: simple→medium, medium→complex, complex stays at ceiling
- API: auth middleware (missing key, wrong key, OPTIONS preflight bypass)
- API: input validation (empty prompt, missing prompt, too long)
- API: trivial routing end-to-end (hours query, greeting)
- API: stats schema validation
- Services: cost estimator (trivial free+savings, complex cost+no-savings, medium partial)
- Resilience: CB_FALLBACK tag format and tier-based route detection

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
│   │   └── cost_estimator.py     # Cost + savings calculation
│   ├── config.py                 # Pydantic settings
│   ├── database.py               # Async SQLAlchemy engine
│   ├── main.py                   # FastAPI app factory + CORS + auth
│   ├── models.py                 # RoutingLog ORM model
│   └── requirements.txt
├── dashboard/                    # Streamlit observability UI
├── tests/
│   ├── conftest.py               # sys.path setup
│   ├── test_api.py               # 12 API integration tests
│   └── test_graph.py             # 40 graph + service unit tests
├── docker-compose.yml
└── README.md
```

---

## Changelog (Recent)

| Commit | Change |
|---|---|
| `bd8e4f7` | docs: add local dev + Ngrok tunnel setup guide |
| `4d71cce` | fix: set correct working Fireworks serverless model IDs (DeepSeek V4) |
| `50847c0` | test+docs: 52/52 tests passing, full README rewrite, max_cost_tier cap |
| `ff3909c` | feat: hybrid 3-layer classifier + confidence gate + calibrated judge rubric |
| `3e5e427` | fix: CORS preflight bypass, un-bypass classifier/judge nodes, cost savings fix |

---

## FAQ

**Q: Do I need to run this locally?**
No. The live Vercel frontend and Ngrok backend tunnel are running during the judging period. Test directly at https://gemmaroute.vercel.app.

**Q: Why does the frontend show "demo mode"?**
The backend runs via Ngrok on a local machine. If the tunnel is inactive, the frontend falls back to an intelligent mock that simulates all 4 tiers correctly using a weighted vocabulary scorer.

**Q: Why is the Fireworks API key needed?**
Simple/trivial queries go to local Gemma for free. Medium/complex queries route to Fireworks AI cloud models. The key is required to test those tiers.

**Q: What if the local AMD server drops offline during judging?**
The circuit breaker detects failures and routes everything to Ollama fallback. If Ollama is also down, the frontend mock kicks in — judges can still review the full UI and architecture.

**Q: How does the Gemma Bonus apply?**
Gemma 2 is used in all 3 layers: Layer 1b classifier (`gemma2:2b`), Layer 2 simple executor (`gemma:2b`), and Layer 3 quality judge (`gemma:2b`). All local, all free, all AMD ROCm.

---

## Contact

📧 jezreelmomoh1234@gmail.com — available during the judging period.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
