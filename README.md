# GemmaRoute

**3-Layer AMD-Native AI Routing Engine.** Cut your LLM API bill by 60 to 80 percent without losing quality.

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-E8001C?logo=amd)](https://www.amd.com/en/developer/rocm.html)
[![Gemma](https://img.shields.io/badge/Google-Gemma%202-8B5CF6?logo=google)](https://deepmind.google/technologies/gemma/)
[![Fireworks AI](https://img.shields.io/badge/Fireworks-AI-3B82F6)](https://fireworks.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏆 Message to the Judges
- **Please watch our 2-minute demo video first for the full live experience.**
- **Our local routing engine is powered by an AMD GPU using ROCm architecture.**
- **You can test the live Vercel link. Note: If our local AMD server happens to be offline during grading, the UI elegantly falls back to simulated data.**
- **To run the backend locally on your own AMD hardware: You MUST provide your own Fireworks API key in the `.env` file before running Docker.**

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/jezreal-dev/gemmaroute.git
cd gemmaroute
copy .env.example .env
# Edit .env and add your FIREWORKS_API_KEY and API_KEY (for securing the backend)

# 2. Launch the stack
docker compose up --build -d

# 3. Open the dashboard
start http://localhost:8501

# 4. Test the API (Requires X-API-Key header)
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"prompt": "What are your business hours?"}'
```

---

## Architecture Overview

GemmaRoute implements a 3-layer routing waterfall using Gemma 2 at every stage.

```
User Prompt
    |
    v
[Layer 1] Heuristic Filter  ----> trivial ----> Instant response ($0.00)
    | non-trivial
    v
[Layer 1] Gemma 2 2B Classifier  (local AMD, $0.00)
    |  JSON: { tier, confidence }
    +---> simple  ----> Gemma 2B  (local AMD, $0.00/req)
    +---> medium  ----> Gemma 2 9B / Cloud (Fireworks AI)
    +---> complex ----> Gemma 2 27B / Cloud (Fireworks AI)
                          |
                    [Layer 3] Gemma Quality Judge
                          |  score 0.0 to 1.0
                    score >= 0.75 ----> Return response
                    score < 0.75 ----> Escalate to next tier
```

### System Resilience & Security
GemmaRoute is designed for production reliability.
*   **API Key Security:** Fully secured API via `X-API-Key` middleware. Only authorized clients can route prompts.
*   **Circuit Breaker:** If the cloud provider (Fireworks AI) goes down or rate-limits, the circuit trips open for 60 seconds. All traffic falls back to local AMD hardware.
*   **Hop Budget:** The router is hard-capped to a maximum of 2 escalation hops to prevent infinite loops and latency spikes.
*   **Exponential Backoff:** Network calls use a 1s, 2s, and 4s backoff schedule for transient errors.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, LangGraph |
| Routing Engine | LangGraph StateGraph |
| Local Models | Gemma 2 2B and Gemma 2B via Ollama |
| Cloud Models | Fireworks AI |
| Database | SQLite and SQLAlchemy |
| Dashboard | Streamlit |
| Container | Docker Compose |

---

## AMD Developer Cloud Deployment

Switch from CPU to AMD ROCm by editing `docker-compose.yml`.

```yaml
    image: ollama/ollama:rocm
    devices:
      - /dev/kfd
      - /dev/dri
```

---

## API Reference

### POST /route
Routes a prompt through the engine. Requires `X-API-Key` header.

**Request:**
```json
{
  "prompt": "I need a refund analysis for order #4821",
  "session_id": "demo-001"
}
```

**Response:**
```json
{
  "response": "I'd be happy to help with your refund...",
  "routing": {
    "initial_tier": "complex",
    "final_tier": "complex",
    "escalations": 0,
    "classifier_confidence": 0.94,
    "quality_score": 0.88,
    "model_used": "accounts/fireworks/models/gemma2-27b-it",
    "latency_ms": 1840,
    "estimated_cost_usd": 0.000081,
    "cost_saved_vs_max_usd": 0.0
  },
  "session_id": "demo-001"
}
```

### GET /stats
Returns aggregate routing statistics and cost savings. Requires `X-API-Key` header.

### GET /health
Returns service health and circuit breaker status.

---

## Built for AMD Developer Hackathon: ACT II
**Track 3: Unicorn (Open Innovation)**

GemmaRoute demonstrates a real-world enterprise use case. Companies paying for expensive closed-source models can deploy GemmaRoute as a proxy and immediately reduce their LLM costs by routing simple queries to local AMD hardware.

---

## License
MIT License. See LICENSE file for details.
