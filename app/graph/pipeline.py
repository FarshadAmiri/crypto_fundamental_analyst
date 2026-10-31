"""LangGraph pipeline: ingest (MCP) → triage → parallel per-coin analysis → CrewAI debate → Pydantic validation → rank & persist."""
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app import db
from app.graph import nodes
from app.graph.debate import run_debate
from app.graph.validator import validate_verdict


class PipelineState(TypedDict):
    run_id: int
    week_start: str
    market: dict
    news: dict
    sentiment: dict
    shortlist: list[dict]
    reports: Annotated[list[dict], operator.add]
    debate_text: str
    verdict: dict
    error: str


def monday_of_current_week() -> str:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=today.weekday())).isoformat()


async def debate_node(state: dict) -> dict:
    try:
        debate_text = await run_debate(state["shortlist"], state["reports"])
    except Exception as e:
        debate_text = ""  # validator falls back to per-coin reports
        return {"debate_text": "", "error": f"crew failed: {e}"}
    return {"debate_text": debate_text}


async def validate_node(state: dict) -> dict:
    verdict = await validate_verdict(
        state.get("debate_text", ""), state["week_start"], state["reports"]
    )
    return {"verdict": verdict.model_dump()}


async def rank_node(state: dict) -> dict:
    """Attach prices, persist picks, and update last week's picks' performance."""
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
    db.finish_run(state["run_id"], "completed", summary=verdict.get("summary", ""))
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


async def run_pipeline() -> int:
    """Entry point for both the weekly scheduler and the manual trigger. Returns run id."""
    week_start = monday_of_current_week()
    run_id = db.start_run(week_start)
    graph = build_graph()
    try:
        await graph.ainvoke({"week_start": week_start, "run_id": run_id, "reports": []})
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        raise
    return run_id
