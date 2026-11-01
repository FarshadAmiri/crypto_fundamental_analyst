"""Graph nodes: deterministic ingest via MCP tools, LLM triage, parallel per-coin analysis.

Every node emits trace events (gathered sources, shortlist, reports, timings) that the
UI replays in the run-details modal.
"""
import time
from typing import Any

from fastmcp import Client
from langgraph.types import Send

from app import config
from app.graph.llm import chat_llm
from app.schemas import CoinReport, TriageResult


async def _call_mcp_tool(client: Client, name: str, arguments: dict | None = None) -> Any:
    result = await client.call_tool(name, arguments or {})
    # fastmcp deserializes structured output into .data; fall back to raw content
    data = getattr(result, "data", None)
    if data is not None:
        return data
    return [c.model_dump() for c in result.content]


async def _safe_mcp_tool(client: Client, name: str, arguments: dict | None = None) -> Any:
    """A dead data source degrades to an empty bundle; it must not kill the whole run."""
    try:
        return await _call_mcp_tool(client, name, arguments)
    except Exception as e:
        return {"error": f"tool {name} failed: {e}", "items": [], "posts": [], "coins": [], "sources": []}


async def ingest(state: dict) -> dict:
    """Gather ALL data up front through the MCP server, honoring the run's source selection."""
    t0 = time.monotonic()
    s = state["settings"]
    async with Client(config.MCP_URL) as client:
        if s["use_market"]:
            market = await _safe_mcp_tool(client, "get_market_snapshot")
        else:
            market = {"coins": [], "sources": [{"name": "CoinGecko markets", "ok": None, "count": 0}]}
        news = await _safe_mcp_tool(
            client, "get_news_headlines", {"feeds": s["feeds"] or None, "days": s["lookback_days"]}
        )
        sentiment = await _safe_mcp_tool(
            client,
            "get_community_sentiment",
            {"reddit": s["use_reddit"], "hackernews": s["use_hackernews"],
             "trending": s["use_trending"], "days": s["lookback_days"]},
        )

    sources = []
    for bundle in (market, news, sentiment):
        sources.extend(bundle.get("sources") or [])
    event = {
        "step": "ingest",
        "duration_s": round(time.monotonic() - t0, 1),
        "sources": sources,
        "counts": {
            "news_items": len(news.get("items") or []),
            "community_posts": len(sentiment.get("posts") or []),
            "market_coins": len(market.get("coins") or []),
        },
    }
    return {"market": market or {}, "news": news or {}, "sentiment": sentiment or {}, "trace": [event]}


def _compact_news(news: dict, limit: int = 40) -> str:
    items = (news or {}).get("items", [])
    lines = [
        f"- [{i.get('source')}] {i.get('title')} :: {i.get('summary', '')[:200]}"
        for i in items[:limit]
    ]
    return "\n".join(lines) or "(no news fetched)"


def _compact_sentiment(sentiment: dict, limit: int = 40) -> str:
    posts = (sentiment or {}).get("posts", [])
    posts = sorted(posts, key=lambda p: p.get("score") or 0, reverse=True)[:limit]
    lines = [
        f"- r/{p.get('subreddit')} (score {p.get('score')}, {p.get('num_comments')} comments): {p.get('title')}"
        for p in posts
    ]
    return "\n".join(lines) or "(no community posts fetched)"


def _compact_market(market: dict, limit: int = 25) -> str:
    coins = (market or {}).get("coins", [])
    lines = [
        f"- {c.get('symbol')} ({c.get('name')}): ${c.get('price_usd')}, mcap {c.get('market_cap')}, 7d {c.get('change_7d_pct')}%"
        for c in coins[:limit]
    ]
    return "\n".join(lines) or "(no market data)"


def _coin_evidence(symbol: str, coin: str, state: dict) -> str:
    """Pre-assembled per-coin evidence bundle so analysis agents need no tool calls."""
    market = state.get("market") or {}
    row = next(
        (c for c in market.get("coins", []) if c.get("symbol") == symbol), None
    )
    news = state.get("news") or {}
    coin_news = [
        n
        for n in news.get("items", [])
        if symbol in (n.get("title", "") + " " + n.get("summary", "")).upper()
    ][:10]
    sentiment = state.get("sentiment") or {}
    coin_posts = [
        p
        for p in sentiment.get("posts", [])
        if symbol in p.get("title", "").upper()
        or symbol in p.get("selftext_excerpt", "").upper()
    ][:10]

    parts = [f"COIN: {coin} ({symbol})"]
    if row:
        parts.append(
            f"MARKET: price ${row.get('price_usd')}, mcap {row.get('market_cap')}, 7d change {row.get('change_7d_pct')}%"
        )
    else:
        parts.append("MARKET: not in top-40 by market cap")
    parts.append(
        "RELATED NEWS:\n"
        + (
            "\n".join(f"- [{n['source']}] {n['title']} :: {n.get('summary', '')[:200]}" for n in coin_news)
            or "(no specific news found)"
        )
    )
    parts.append(
        "COMMUNITY MENTIONS:\n"
        + (
            "\n".join(
                f"- r/{p['subreddit']} (score {p.get('score')}): {p['title']}" for p in coin_posts
            )
            or "(no specific community mentions found)"
        )
    )
    return "\n\n".join(parts)


def _shortlist_entry(symbol: str, reason: str, state: dict) -> dict:
    name = symbol
    for c in (state.get("market") or {}).get("coins", []):
        if c.get("symbol") == symbol:
            name = c.get("name") or symbol
            break
    return {"coin": name, "symbol": symbol, "reason": reason, "evidence": _coin_evidence(symbol, name, state)}


async def triage(state: dict) -> dict:
    """Build the week's shortlist: from the user's watchlist if given, else one cheap LLM pass. No tools."""
    t0 = time.monotonic()
    watchlist = state["settings"]["watchlist"]
    if watchlist:
        shortlist = [_shortlist_entry(sym, "user watchlist", state) for sym in watchlist]
        event = {"step": "triage", "mode": "watchlist", "duration_s": round(time.monotonic() - t0, 1),
                 "shortlist": [{k: s[k] for k in ("coin", "symbol", "reason")} for s in shortlist]}
        return {"shortlist": shortlist, "trace": [event]}

    evidence = (
        "MARKET SNAPSHOT:\n"
        + _compact_market(state["market"])
        + "\n\nNEWS HEADLINES (last 7 days):\n"
        + _compact_news(state["news"])
        + "\n\nCOMMUNITY POSTS (top by upvotes):\n"
        + _compact_sentiment(state["sentiment"])
    )
    llm = chat_llm().with_structured_output(TriageResult)
    result = await llm.ainvoke(
        [
            (
                "system",
                "You are a crypto research triage analyst. From the evidence below, pick the "
                f"{state['settings']['shortlist_size']} coins most worth deep analysis this week. Favor coins with "
                "concrete catalysts (news, funding, upgrades, strong momentum) over stale mega-caps. "
                "Never invent data that is not in the evidence.",
            ),
            ("user", evidence),
        ]
    )
    shortlist = [
        _shortlist_entry(c.symbol.upper(), c.reason, state)
        for c in result.coins[: state["settings"]["shortlist_size"]]
    ]
    event = {"step": "triage", "mode": "auto", "duration_s": round(time.monotonic() - t0, 1),
             "shortlist": [{k: s[k] for k in ("coin", "symbol", "reason")} for s in shortlist]}
    return {"shortlist": shortlist, "trace": [event]}


async def analyze(payload: dict) -> dict:
    """One lightweight agent per coin. Reads the pre-fetched evidence; no live tool calls."""
    t0 = time.monotonic()
    symbol = payload.get("symbol", "?")
    try:
        llm = chat_llm().with_structured_output(CoinReport)
        report = await llm.ainvoke(
            [
                (
                    "system",
                    "You are a crypto fundamental analyst. Using ONLY the evidence bundle provided, "
                    "write a short weekly assessment: fundamentals, community sentiment, notable signals "
                    "(investments, partnerships, whale/exchange-flow news), a 0-10 attractiveness score "
                    "for the coming week, and your confidence (0-1) given how thin the evidence is. "
                    "If evidence is sparse, say so and lower confidence. Be concise.",
                ),
                ("user", payload["evidence"]),
            ]
        )
        r = report.model_dump()
        return {
            "reports": [r],
            "trace": [{
                "step": "analyze", "symbol": symbol,
                "duration_s": round(time.monotonic() - t0, 1),
                "report": r,
            }],
        }
    except Exception as e:
        # skip this coin rather than failing the whole week
        return {
            "reports": [],
            "trace": [{"step": "analyze", "symbol": symbol, "error": f"{e}",
                       "duration_s": round(time.monotonic() - t0, 1)}],
        }


def fan_out_analyses(state: dict) -> list[Send]:
    """Fan the shortlist out to parallel per-coin analysis nodes."""
    return [
        Send("analyze", {"coin": s["coin"], "symbol": s["symbol"], "evidence": s["evidence"]})
        for s in state["shortlist"]
    ]
