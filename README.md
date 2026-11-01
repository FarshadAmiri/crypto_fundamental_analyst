# crypto_fundamental_analyst

An agentic fundamental analyst for cryptocurrency — weekly picks elected by a local LLM pipeline.

No API keys, no paid services. Data comes from public endpoints (CoinGecko, RSS feeds, community posts) and all the "thinking" goes through your own vLLM server. One run per week ranks a handful of coins worth watching, saves the picks, and tracks how last week's picks actually performed. Small web UI included.

## How it works

Three frameworks, three jobs — glued together by LangGraph:

- **LangGraph** owns the pipeline (state, control flow, parallel fan-out)
- **CrewAI** does the debate step only: a bull, a bear and a quant discuss the shortlist
- **Pydantic AI** is the gatekeeper: converts the crew's free-text consensus into a strict typed verdict, retrying on schema violations
- **fastmcp** powers a small MCP server (`research-mcp`) — the only place in the system that touches the internet

All three frameworks talk to the same vLLM endpoint (OpenAI-compatible). "Different models" in the debate = different personas, not different LLMs.

## The process, step by step

Once a week (or when you click **Run now**), the pipeline does this, in order:

1. **Gather data** — fetches market prices, news headlines and community posts from the internet. *Tool: MCP tools served by fastmcp (data fetched with httpx and feedparser).*
2. **Triage** — one cheap LLM call reads everything gathered and picks ~8 coins worth analyzing. *Tool: LangChain's ChatOpenAI (pointed at vLLM), structured output.*
3. **Analyze each coin** — one small agent per coin writes a mini-report (fundamentals, sentiment, signals, score). All coins run in parallel, and each agent only reads the already-fetched data. *Tool: LangGraph (parallel fan-out) + ChatOpenAI.*
4. **Debate** — three CrewAI personas (bull, bear, quant) discuss the reports: bull makes a case, bear attacks it, quant synthesizes a final ranking. *Tool: CrewAI.*
5. **Validate** — the crew's free-text answer is forced into a strict typed verdict (ranks, scores, risks). If it doesn't fit the schema, it retries; if it still fails, a fallback ranking is built from step 3. *Tool: Pydantic AI.*
6. **Save & track** — the final picks are stored in the database with their current price; on the next run, last week's picks get their return % filled in. *Tool: SQLite.*

The orchestrator that runs steps 1→6 in order, passes data between them and handles failures is **LangGraph**. Steps 1–6 all use the same local LLM through vLLM's OpenAI-compatible API.

## Workflow schema

```mermaid
flowchart TD
    A["1. ingest<br/>fastmcp tools → httpx + feedparser"] --> B["2. triage<br/>langchain-openai (ChatOpenAI + structured output)"]
    B --> C["3. analyze × N (parallel fan-out)<br/>langchain-openai, one agent per coin"]
    C --> D["4. debate<br/>CrewAI: bull → bear rebuttal → quant synthesis"]
    D --> E["5. validate<br/>Pydantic AI → WeeklyVerdict schema<br/>(auto-retry, falls back to report ranking)"]
    E --> F["6. rank & persist<br/>SQLite + price tracking"]

    subgraph MCP ["research-mcp server (fastmcp, HTTP :8001)"]
        T1["get_market_snapshot<br/>CoinGecko /coins/markets"]
        T2["get_news_headlines<br/>RSS: CoinDesk, Cointelegraph, Decrypt"]
        T3["get_community_sentiment<br/>Reddit JSON → HN Algolia → CoinGecko trending (fallback chain)"]
        T4["get_coin_context<br/>per-symbol bundle"]
    end

    A -.calls.-> MCP
```

Everything downstream of ingest reads pre-fetched evidence — the analyst agents never browse. Fetchers are hard-budgeted: ≤30 items per source, ~1500 chars per item, 15s timeout, one attempt, results cached for 6h. A dead source degrades to an empty bundle instead of killing the run.

A run costs roughly: 1 triage call + N parallel coin analyses (N≈8) + 3 debate calls + 1 validation call. A few minutes on a single consumer GPU.

## Layout

```
app/
  main.py            FastAPI + weekly cron scheduler (Mon 06:00 by default)
  mcp_server/        research-mcp: tool defs + budgeted fetchers
  graph/
    pipeline.py      LangGraph state machine
    nodes.py         ingest / triage / per-coin analysis
    debate.py        CrewAI debate crew
    validator.py     Pydantic AI verdict validator
  db.py              SQLite: runs, picks, return% tracking
web/index.html       minimal UI (picks, rationale, run button, history)
docker-compose.yml   mcp + app (vLLM optional, see below)
```

## Setup

Prereqs: Docker, and a vLLM server serving an instruct model that supports tool calling (tested with Qwen3-4B-AWQ on an RTX 3060 Ti 8GB).

1. vLLM outside Docker is the default setup (it's the heaviest piece, and you may already run it in WSL):

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --host 0.0.0.0 --port 8002 \
     --model /path/to/Qwen3-4B-AWQ \
     --served-model-name Qwen3-4B \
     --quantization awq --max-model-len 16384 \
     --gpu-memory-utilization 0.80 --enforce-eager \
     --enable-auto-tool-choice --tool-call-parser hermes
   ```

   The last two flags matter — structured output in the pipeline needs them.

2. Configure and start:

   ```bash
   cp .env.example .env   # VLLM_BASE_URL + MODEL_NAME must match your vLLM
   docker compose build
   docker compose up -d
   ```

   (If PyPI is unreliable on your network, the Dockerfile already resolves from the TUNA mirror.)

3. Open http://localhost:8000 → **Run now**. Weekly runs happen automatically (Monday 06:00, configurable in `.env`).

A fully dockerized vLLM also exists as an optional profile: `docker compose --profile docker-gpu up -d vllm`.

## Customizing a run

The left panel controls what each run does (saved in your browser, defaults in `.env`):

- **Watchlist** — comma-separated tickers (e.g. `BTC, ETH, SOL`). Empty = the triage agent picks the shortlist itself.
- **Shortlist size** — how many coins get deep analysis (auto mode).
- **Lookback window** — 3/7/14 days of news and community posts.
- **Debate depth** — `quick` (2 LLM turns), `standard` (3), `extended` (4).
- **Data sources** — tick market snapshot, individual RSS feeds, and community sources (Reddit → HackerNews → CoinGecko trending, tried in that order with automatic fallback).

## Observability (run details)

Every run records a full trace. The **Run details** button in the UI opens a modal showing:

- each pipeline step with its duration and ok/fail status,
- every data source attempted, with its URL, item count and success flag,
- the shortlist (and whether it came from your watchlist or the LLM),
- each per-coin analyst report,
- the **complete crew debate transcript**, turn by turn with agent roles,
- whether Pydantic AI validated the crew output or the fallback ranking was used.

Same data over the API: `GET /api/runs/{id}/trace`.

## Notes / gotchas

- Reddit and Google-fronted services (e.g. HN Algolia) are blocked from some networks; the sentiment tool falls back automatically (Reddit → HN → CoinGecko trending).
- Past picks get `return_pct` filled in on the next weekly run — a free, naive backtest.
- To reuse the MCP tools elsewhere (any MCP client), point it at `http://localhost:8001/mcp`.

## Stack

Python 3.11 · LangGraph · CrewAI · Pydantic AI · fastmcp · FastAPI · httpx/feedparser · APScheduler · SQLite · vLLM · Docker Compose
