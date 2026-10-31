"""The 'research-mcp' MCP server: every data source the agents can touch is a tool here.

Run standalone:  python -m app.mcp_server.server  (serves streamable HTTP on :8001 at /mcp)
"""
from fastmcp import FastMCP

from app.mcp_server import sources

mcp = FastMCP("research-mcp")


@mcp.tool
async def get_market_snapshot() -> dict:
    """Top ~40 coins by market cap: price, market cap, 7-day price change."""
    return await sources.fetch_market_snapshot()


@mcp.tool
async def get_news_headlines() -> dict:
    """Crypto news headlines from public RSS feeds within the last 7 days (title + short summary)."""
    return await sources.fetch_news_headlines()


@mcp.tool
async def get_community_sentiment() -> dict:
    """Hot posts from public crypto subreddits: title, upvotes, comment count. Public JSON, no auth."""
    return await sources.fetch_community_sentiment()


@mcp.tool
async def get_coin_context(symbol: str) -> dict:
    """Everything cached about one coin (by ticker, e.g. 'ETH'): market row, related news, community mentions."""
    return await sources.fetch_coin_context(symbol)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
