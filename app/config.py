"""Central configuration, driven entirely by environment variables (set in .env / docker-compose)."""
import os

# --- LLM (vLLM, OpenAI-compatible) ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# --- MCP research server ---
MCP_URL = os.getenv("MCP_URL", "http://mcp:8001/mcp")

# --- Research budgets: keep data gathering light and predictable ---
MAX_ITEMS_PER_SOURCE = int(os.getenv("MAX_ITEMS_PER_SOURCE", "30"))
MAX_CHARS_PER_ITEM = int(os.getenv("MAX_CHARS_PER_ITEM", "1500"))
FETCH_TIMEOUT_SECONDS = int(os.getenv("FETCH_TIMEOUT_SECONDS", "15"))
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "6"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "8"))

# --- Storage / app ---
DB_PATH = os.getenv("DB_PATH", "/data/app.db")

# Weekly run: Monday 06:00 container time
SCHEDULER_DAY_OF_WEEK = os.getenv("SCHEDULER_DAY_OF_WEEK", "mon")
SCHEDULER_HOUR = int(os.getenv("SCHEDULER_HOUR", "6"))

# --- Data sources (all free, public, no API keys) ---
NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
]

SENTIMENT_SUBREDDITS = [
    "CryptoCurrency",
    "CryptoMarkets",
    "bitcoin",
    "ethfinance",
]
