"""Budgeted fetchers behind the MCP tools: TTL cache, hard item/char caps, single attempt, no crawling."""
import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx

from app import config

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str):
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < config.CACHE_TTL_HOURS * 3600:
        return hit[1]
    return None


def _cache_set(key: str, value: object):
    _cache[key] = (time.monotonic(), value)
    return value


def _clip(text: str, limit: int | None = None) -> str:
    text = (text or "").strip()
    return text[: (limit or config.MAX_CHARS_PER_ITEM)]


async def _get(url: str, headers: dict | None = None, params: dict | None = None) -> httpx.Response | None:
    """Single-attempt GET; never retries, never raises. Returns None on any failure."""
    try:
        async with httpx.AsyncClient(
            timeout=config.FETCH_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp
    except Exception:
        return None


def _within_lookback(dt: datetime | None) -> bool:
    if dt is None:
        return True  # keep items without a parseable date; triage LLM will judge
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)
    return dt >= cutoff


def _safe_json(resp: httpx.Response):
    """Parse a response body; corrupted/truncated payloads (seen on this network) become None."""
    try:
        return resp.json()
    except Exception:
        return None


async def fetch_market_snapshot() -> dict:
    """Top coins by market cap with weekly price change (CoinGecko public API)."""
    key = "market"
    if (hit := _cache_get(key)) is not None:
        return hit
    resp = await _get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "40",
            "page": "1",
            "price_change_percentage": "7d",
        },
    )
    if resp is None or (data := _safe_json(resp)) is None:
        return {"coins": [], "error": "market data unavailable"}  # not cached: next run retries
    coins = []
    for c in data:
        coins.append(
            {
                "name": c.get("name"),
                "symbol": (c.get("symbol") or "").upper(),
                "price_usd": c.get("current_price"),
                "market_cap": c.get("market_cap"),
                "change_7d_pct": c.get("price_change_percentage_7d_in_currency"),
            }
        )
    return _cache_set(key, {"coins": coins})


async def fetch_news_headlines() -> dict:
    """Headlines from crypto RSS feeds within the lookback window (summary clipped, no article crawling)."""
    key = "news"
    if (hit := _cache_get(key)) is not None:
        return hit
    items = []
    for source, url in config.NEWS_FEEDS:
        resp = await _get(url)
        if resp is None:
            continue
        try:
            feed = feedparser.parse(resp.text)
        except Exception:
            continue
        for entry in feed.entries[: config.MAX_ITEMS_PER_SOURCE]:
            published = None
            if entry.get("published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if not _within_lookback(published):
                continue
            items.append(
                {
                    "source": source,
                    "title": _clip(entry.get("title", ""), 300),
                    "summary": _clip(entry.get("summary", "")),
                    "published": published.isoformat() if published else None,
                    "link": entry.get("link", ""),
                }
            )
    if items:
        return _cache_set(key, {"items": items})
    return {"items": []}  # all feeds failed: don't cache the empty result


HN_QUERIES = ["cryptocurrency", "bitcoin", "ethereum", "stablecoin", "crypto exchange"]


async def _sentiment_from_hackernews() -> list[dict]:
    """Fallback when Reddit is unreachable/blocked: HN Algolia public search API (no auth)."""
    posts = []
    for query in HN_QUERIES:
        resp = await _get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"query": query, "tags": "story", "hitsPerPage": "10",
                    "numericFilters": f"created_at_i>{int(time.time() - config.LOOKBACK_DAYS * 86400)}"},
        )
        if resp is None or (data := _safe_json(resp)) is None:
            continue
        for hit in data.get("hits", []):
            posts.append(
                {
                    "subreddit": f"HackerNews ({query})",
                    "title": _clip(hit.get("title", ""), 300),
                    "score": hit.get("points"),
                    "num_comments": hit.get("num_comments"),
                    "selftext_excerpt": _clip(hit.get("story_text", "") or "", 400),
                }
            )
    return posts


async def _sentiment_from_trending() -> list[dict]:
    """Final fallback: CoinGecko trending search (community attention proxy, same host as market data)."""
    resp = await _get("https://api.coingecko.com/api/v3/search/trending")
    if resp is None or (data := _safe_json(resp)) is None:
        return []
    posts = []
    for i, entry in enumerate(data.get("coins", [])[:15]):
        item = entry.get("item", {})
        if not item.get("name"):
            continue
        extra = item.get("data") or {}
        chg = extra.get("price_change_percentage_24h")
        if isinstance(chg, dict):
            chg = chg.get("usd")
        posts.append(
            {
                "subreddit": "CoinGecko trending",
                "title": _clip(
                    f"Trending coin #{i + 1}: {item.get('name')} ({item.get('symbol')}) "
                    f"mcap rank {item.get('market_cap_rank')}, 24h change {chg if chg is not None else 'n/a'}",
                    300,
                ),
                "score": 15 - i,  # descending attention rank
                "num_comments": None,
                "selftext_excerpt": "",
            }
        )
    return posts


async def fetch_community_sentiment() -> dict:
    """Hot posts (title + score + comment count) from public sources, no auth.

    First tries public subreddit JSON endpoints; if those are blocked (Reddit now
    403s many unauthenticated clients), falls back to the HN Algolia public API.
    """
    key = "sentiment"
    if (hit := _cache_get(key)) is not None:
        return hit
    headers = {"User-Agent": "crypto-research-mcp/0.1 (local research tool)"}
    posts = []
    for sub in config.SENTIMENT_SUBREDDITS:
        resp = await _get(f"https://www.reddit.com/r/{sub}/hot.json?limit=15", headers=headers)
        if resp is None:
            continue
        try:
            children = resp.json()["data"]["children"]
        except Exception:
            continue
        for child in children[: config.MAX_ITEMS_PER_SOURCE]:
            d = child.get("data", {})
            created = datetime.fromtimestamp(d.get("created_utc", 0), tzinfo=timezone.utc)
            if not _within_lookback(created):
                continue
            posts.append(
                {
                    "subreddit": sub,
                    "title": _clip(d.get("title", ""), 300),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "selftext_excerpt": _clip(d.get("selftext", ""), 400),
                }
            )
    if not posts:
        posts = await _sentiment_from_hackernews()
    if not posts:
        posts = await _sentiment_from_trending()
    if posts:
        return _cache_set(key, {"posts": posts})
    return {"posts": []}  # all sources failed: don't cache the empty result


async def fetch_coin_context(symbol: str) -> dict:
    """Market row + news headlines + reddit mentions for one symbol (uses the cached bundles above)."""
    symbol = symbol.upper()
    market = await fetch_market_snapshot()
    news = await fetch_news_headlines()
    sentiment = await fetch_community_sentiment()

    coin_row = next(
        (c for c in market.get("coins", []) if c.get("symbol") == symbol), None
    )
    coin_news = [
        n
        for n in news.get("items", [])
        if symbol in (n.get("title", "") + " " + n.get("summary", "")).upper()
    ][:10]
    coin_posts = [
        p
        for p in sentiment.get("posts", [])
        if symbol in p.get("title", "").upper()
        or symbol in p.get("selftext_excerpt", "").upper()
    ][:10]
    return {
        "symbol": symbol,
        "market": coin_row,
        "news": coin_news,
        "community_posts": coin_posts,
    }
