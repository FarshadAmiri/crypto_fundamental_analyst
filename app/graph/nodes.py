"""Graph nodes: deterministic ingest via MCP tools, LLM triage, parallel per-coin analysis."""
import json
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


async def _safe_mcp_tool(client: Client, name: str) -> Any:
    """A dead data source degrades to an empty bundle; it must not kill the whole run."""
    try:
        return await _call_mcp_tool(client, name)
    except Exception as e:
        return {"error": f"tool {name} failed: {e}", "items": [], "posts": [], "coins": []}


async def ingest(state: dict) -> dict:
    """Gather ALL data up front through the MCP server. Downstream agents never browse."""
    async with Client(config.MCP_URL) as client:
        market = await _safe_mcp_tool(client, "get_market_snapshot")
        news = await _safe_mcp_tool(client, "get_news_headlines")
        sentiment = await _safe_mcp_tool(client, "get_community_sentiment")
    return {"market": market or {}, "news": news or {}, "sentiment": sentiment or {}}


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


async def triage(state: dict) -> dict:
    """One cheap LLM pass over the fetched evidence: build the week's shortlist. No tools."""
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
                f"{config.SHORTLIST_SIZE} coins most worth deep analysis this week. Favor coins with "
                "concrete catalysts (news, funding, upgrades, strong momentum) over stale mega-caps. "
                "Never invent data that is not in the evidence.",
            ),
            ("user", evidence),
        ]
    )
    shortlist = []
    for c in result.coins[: config.SHORTLIST_SIZE]:
        shortlist.append(
            {
                "coin": c.coin,
                "symbol": c.symbol.upper(),
                "reason": c.reason,
                "evidence": _coin_evidence(c.symbol.upper(), c.coin, state),
            }
        )
    return {"shortlist": shortlist}


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


async def analyze(payload: dict) -> dict:
    """One lightweight agent per coin. Reads the pre-fetched evidence; no live tool calls."""
    llm = chat_llm().with_structured_output(CoinReport)
    try:
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
        return {"reports": [report.model_dump()]}
    except Exception as e:
        # skip this coin rather than failing the whole week
        return {"reports": [], "_analysis_errors": [f"{payload.get('symbol')}: {e}"]}


def fan_out_analyses(state: dict) -> list[Send]:
    """Fan the shortlist out to parallel per-coin analysis nodes."""
    return [
        Send("analyze", {"coin": s["coin"], "symbol": s["symbol"], "evidence": s["evidence"]})
        for s in state["shortlist"]
    ]
