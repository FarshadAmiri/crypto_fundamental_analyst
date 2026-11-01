"""LangGraph pipeline: ingest (MCP) → triage → parallel per-coin analysis → CrewAI debate → Pydantic validation → rank & persist.

The whole run is recorded as a structured trace (sources, shortlist, reports, debate
turns, validation outcome, timings) stored on the run row for the UI's details modal.
"""
import json
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app import config, db
from app.graph import nodes
from app.graph.debate import run_debate
from app.graph.validator import validate_verdict


class PipelineState(TypedDict):
    run_id: int
    week_start: str
    settings: dict
    market: dict
    news: dict
    sentiment: dict
    shortlist: list[dict]
    reports: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    debate_text: str
    verdict: dict
    error: str


def normalize_settings(settings: dict | None) -> dict:
    """Fill defaults and sanitize whatever the UI sent."""
    s = settings or {}
    valid_feeds = [name for name, _ in config.NEWS_FEEDS]
    watchlist = [str(x).strip().upper() for x in (s.get("watchlist") or []) if str(x).strip()]
    return {
        "watchlist": watchlist or list(config.DEFAULT_WATCHLIST),
        "shortlist_size": max(3, min(15, int(s.get("shortlist_size") or config.SHORTLIST_SIZE))),
        "lookback_days": max(1, min(30, int(s.get("lookback_days") or config.LOOKBACK_DAYS))),
        "debate_depth": s.get("debate_depth") if s.get("debate_depth") in ("quick", "standard", "extended")
        else config.DEFAULT_DEBATE_DEPTH,
        "use_market": bool(s.get("use_market", True)),
        "feeds": [f for f in (s.get("feeds") or valid_feeds) if f in valid_feeds],
        "use_reddit": bool(s.get("use_reddit", True)),
        "use_hackernews": bool(s.get("use_hackernews", True)),
        "use_trending": bool(s.get("use_trending", True)),
    }


def monday_of_current_week() -> str:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=today.weekday())).isoformat()


async def debate_node(state: dict) -> dict:
    t0 = datetime.now(timezone.utc)
    depth = state["settings"]["debate_depth"]
    try:
        debate_text, turns = await run_debate(state["shortlist"], state["reports"], depth)
        event = {
            "step": "debate", "depth": depth,
            "duration_s": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
            "turns": turns,
        }
        return {"debate_text": debate_text, "trace": [event]}
    except Exception as e:
        event = {
            "step": "debate", "depth": depth,
            "duration_s": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
            "error": str(e),
        }
        return {"debate_text": "", "error": f"crew failed: {e}", "trace": [event]}


async def validate_node(state: dict) -> dict:
    t0 = datetime.now(timezone.utc)
    verdict, status = await validate_verdict(
        state.get("debate_text", ""), state["week_start"], state["reports"]
    )
    event = {
        "step": "validate", "status": status,
        "duration_s": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
        "n_picks": len(verdict.picks),
        "summary": verdict.summary,
    }
    return {"verdict": verdict.model_dump(), "trace": [event]}


def _assemble_trace(events: list[dict]) -> dict:
    """Group the parallel/interleaved event stream into the ordered step list for the UI."""
    by_step: dict[str, list[dict]] = {}
    for e in events:
        by_step.setdefault(e.get("step", "?"), []).append(e)

    def first(step):
        return by_step.get(step, [None])[0]

    reports = [e["report"] for e in by_step.get("analyze", []) if "report" in e]
    analysis_errors = [f"{e.get('symbol')}: {e['error']}" for e in by_step.get("analyze", []) if "error" in e]

    steps = []
    if ing := first("ingest"):
        steps.append(ing)
    if tr := first("triage"):
        steps.append(tr)
    steps.append({"step": "analyze", "reports": reports, "errors": analysis_errors})
    if de := first("debate"):
        steps.append(de)
    if va := first("validate"):
        steps.append(va)
    return {"steps": steps}


async def rank_node(state: dict) -> dict:
    """Attach prices, persist picks and the run trace, update last week's picks' performance."""
    verdict = state["verdict"]
    prices = {
        c.get("symbol"): c.get("price_usd")
        for c in (state.get("market") or {}).get("coins", [])
        if c.get("symbol")
    }
    picks = []
    for p in verdict["picks"]:
        pick = dict(p)
        pick["price_at_pick"] = prices.get(p["symbol"].upper())
        picks.append(pick)
    db.save_picks(state["run_id"], state["week_start"], picks)
    db.update_previous_picks_performance(prices)

    trace = _assemble_trace(state.get("trace", []))
    trace["steps"].append({
        "step": "rank", "n_saved": len(picks),
        "settings": state["settings"],
    })
    db.finish_run(state["run_id"], "completed", summary=verdict.get("summary", ""),
                  trace=json.dumps(trace))
    return {}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", nodes.ingest)
    g.add_node("triage", nodes.triage)
    g.add_node("analyze", nodes.analyze)
    g.add_node("debate", debate_node)
    g.add_node("validate", validate_node)
    g.add_node("rank", rank_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "triage")
    # fan-out: one analyze task per shortlisted coin, run in parallel
    g.add_conditional_edges("triage", nodes.fan_out_analyses, ["analyze"])
    g.add_edge("analyze", "debate")
    g.add_edge("debate", "validate")
    g.add_edge("validate", "rank")
    g.add_edge("rank", END)
    return g.compile()


async def run_pipeline(settings: dict | None = None) -> int:
    """Entry point for both the weekly scheduler and the manual trigger. Returns run id."""
    norm = normalize_settings(settings)
    week_start = monday_of_current_week()
    run_id = db.start_run(week_start, settings=json.dumps(norm))
    graph = build_graph()
    try:
        await graph.ainvoke(
            {"run_id": run_id, "week_start": week_start, "settings": norm,
             "reports": [], "trace": []}
        )
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        raise
    return run_id
