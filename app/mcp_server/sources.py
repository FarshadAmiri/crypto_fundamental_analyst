"""Budgeted fetchers behind the MCP tools: TTL cache, hard item/char caps, single attempt, no crawling.

Each fetcher reports provenance (source name, URL, item count, ok/fail) so the
pipeline trace can show exactly what was gathered from where.
"""
import asyncio
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
    """GET with one immediate retry after a short pause (flaky-network tolerance); never raises."""
    last = None
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(
                timeout=config.FETCH_TIMEOUT_SECONDS, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                return resp
        except Exception as e:
            last = e
    return None


def _safe_json(resp: httpx.Response):
    """Parse a response body; corrupted/truncated payloads (seen on some networks) become None."""
    try:
        return resp.json()
    except Exception:
        return None


def _within_lookback(dt: datetime | None, days: int) -> bool:
    if dt is None:
        return True  # keep items without a parseable date; triage LLM will judge
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return dt >= cutoff


async def fetch_market_snapshot() -> dict:
    """Top coins by market cap with weekly price change (CoinGecko public API)."""
    key = "market"
    if (hit := _cache_get(key)) is not None:
        return hit
    url = "https://api.coingecko.com/api/v3/coins/markets"
    resp = await _get(
        url,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "40",
            "page": "1",
            "price_change_percentage": "7d",
        },
    )
    sources = [{"name": "CoinGecko markets", "url": url, "count": 0, "ok": False}]
    if resp is None or (data := _safe_json(resp)) is None:
        return {"coins": [], "sources": sources}  # not cached: next run retries
    coins = [
        {
            "name": c.get("name"),
            "symbol": (c.get("symbol") or "").upper(),
            "price_usd": c.get("current_price"),
            "market_cap": c.get("market_cap"),
            "change_7d_pct": c.get("price_change_percentage_7d_in_currency"),
        }
        for c in data
    ]
    sources[0].update(count=len(coins), ok=True)
    return _cache_set(key, {"coins": coins, "sources": sources})


async def fetch_news_headlines(feeds: list[str] | None = None, days: int | None = None) -> dict:
    """Headlines from selected crypto RSS feeds within the lookback window (summary clipped, no article crawling)."""
    days = days or config.LOOKBACK_DAYS
    feeds = feeds or [name for name, _ in config.NEWS_FEEDS]
    key = f"news:{','.join(sorted(feeds))}:{days}d"
    if (hit := _cache_get(key)) is not None:
        return hit
    items, sources = [], []
    for source, url in config.NEWS_FEEDS:
        if source not in feeds:
            continue
        meta = {"name": f"{source} (RSS)", "url": url, "count": 0, "ok": False}
        sources.append(meta)
        resp = await _get(url)
        if resp is None:
            continue
        try:
            feed = feedparser.parse(resp.text)
        except Exception:
            continue
        n_before = len(items)
        for entry in feed.entries[: config.MAX_ITEMS_PER_SOURCE]:
            published = None
            if entry.get("published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if not _within_lookback(published, days):
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
        meta.update(count=len(items) - n_before, ok=True)
    result = {"items": items, "sources": sources}
    if items:
        return _cache_set(key, result)
    return result  # all feeds failed: don't cache the empty result


HN_QUERIES = ["cryptocurrency", "bitcoin", "ethereum", "stablecoin", "crypto exchange"]


async def _sentiment_from_hackernews(days: int) -> tuple[list[dict], list[dict]]:
    """Fallback when Reddit is unreachable/blocked: HN Algolia public search API (no auth).

    Stops early after 3 consecutive failed queries (source is clearly unreachable).
    """
    posts, sources = [], []
    consecutive_fails = 0
    for query in HN_QUERIES:
        if consecutive_fails >= 2:
            break
        url = "https://hn.algolia.com/api/v1/search_by_date"
        meta = {"name": f"HackerNews search: {query}", "url": url, "count": 0, "ok": False}
        sources.append(meta)
        resp = await _get(
            url,
            params={"query": query, "tags": "story", "hitsPerPage": "10",
                    "numericFilters": f"created_at_i>{int(time.time() - days * 86400)}"},
        )
        if resp is None or (data := _safe_json(resp)) is None:
            consecutive_fails += 1
            continue
        consecutive_fails = 0
        n_before = len(posts)
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
        meta.update(count=len(posts) - n_before, ok=True)
    return posts, sources


async def _sentiment_from_trending() -> tuple[list[dict], list[dict]]:
    """Final fallback: CoinGecko trending search (community attention proxy, same host as market data)."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    sources = [{"name": "CoinGecko trending", "url": url, "count": 0, "ok": False}]
    resp = await _get(url)
    if resp is None or (data := _safe_json(resp)) is None:
        return [], sources
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
    sources[0].update(count=len(posts), ok=bool(posts))
    return posts, sources


async def _sentiment_from_reddit(subs: list[str], days: int) -> tuple[list[dict], list[dict]]:
    posts, sources = [], []
    consecutive_fails = 0
    headers = {"User-Agent": "crypto-research-mcp/0.1 (local research tool)"}
    for sub in subs:
        if consecutive_fails >= 2:
            break  # subreddit source is clearly blocked; don't waste time on the rest
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=15"
        meta = {"name": f"Reddit r/{sub}", "url": url, "count": 0, "ok": False}
        sources.append(meta)
        resp = await _get(url, headers=headers)
        if resp is None:
            consecutive_fails += 1
            continue
        try:
            children = resp.json()["data"]["children"]
        except Exception:
            consecutive_fails += 1
            continue
        consecutive_fails = 0
        n_before = len(posts)
        for child in children[: config.MAX_ITEMS_PER_SOURCE]:
            d = child.get("data", {})
            created = datetime.fromtimestamp(d.get("created_utc", 0), tzinfo=timezone.utc)
            if not _within_lookback(created, days):
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
        meta.update(count=len(posts) - n_before, ok=True)
    return posts, sources


async def fetch_community_sentiment(
    reddit: bool = True, hackernews: bool = True, trending: bool = True, days: int | None = None
) -> dict:
    """Hot posts (title + score + comment count) from the enabled public sources, no auth.

    Tries them in order: Reddit → HackerNews → CoinGecko trending; stops at the
    first source that yields anything (unless none do).
    """
    days = days or config.LOOKBACK_DAYS
    flags = f"r{int(reddit)}h{int(hackernews)}t{int(trending)}:{days}d"
    key = f"sentiment:{flags}"
    if (hit := _cache_get(key)) is not None:
        return hit

    posts: list[dict] = []
    sources: list[dict] = []
    if reddit:
        posts, sources = await _sentiment_from_reddit(config.SENTIMENT_SUBREDDITS, days)
    if not posts and hackernews:
        posts, hn_sources = await _sentiment_from_hackernews(days)
        sources += hn_sources
    if not posts and trending:
        posts, tr_sources = await _sentiment_from_trending()
        sources += tr_sources

    result = {"posts": posts, "sources": sources}
    if posts:
        return _cache_set(key, result)
    return result  # all sources failed: don't cache the empty result


async def fetch_coin_context(symbol: str) -> dict:
    """Market row + news headlines + community mentions for one symbol (uses the cached bundles above)."""
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
