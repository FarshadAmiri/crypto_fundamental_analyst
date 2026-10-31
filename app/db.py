"""SQLite persistence. The API process owns the DB; runs are recorded weekly with picks and performance tracking."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app import config


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                error TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                week_start TEXT NOT NULL,
                coin TEXT NOT NULL,
                symbol TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                rationale TEXT DEFAULT '',
                risks TEXT DEFAULT '',
                price_at_pick REAL,
                price_latest REAL,
                return_pct REAL
            );
            """
        )


def start_run(week_start: str) -> int:
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO runs (week_start, status, created_at) VALUES (?, 'running', ?)",
            (week_start, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, summary: str = "", error: str = ""):
    with get_conn() as c:
        c.execute(
            "UPDATE runs SET status=?, summary=?, error=? WHERE id=?",
            (status, summary, error, run_id),
        )


def save_picks(run_id: int, week_start: str, picks: list[dict]):
    with get_conn() as c:
        for p in picks:
            c.execute(
                """INSERT INTO picks
                   (run_id, week_start, coin, symbol, rank, score, confidence,
                    rationale, risks, price_at_pick)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, week_start, p["coin"], p["symbol"], p["rank"],
                    p["score"], p["confidence"], p["rationale"], p["risks"],
                    p.get("price_at_pick"),
                ),
            )


def update_previous_picks_performance(prices: dict[str, float]):
    """Fill price_latest / return_pct for last week's picks using this week's market snapshot."""
    with get_conn() as c:
        rows = c.execute(
            """SELECT p.id, p.symbol, p.price_at_pick FROM picks p
               JOIN runs r ON r.id = p.run_id
               WHERE p.price_latest IS NULL AND p.price_at_pick IS NOT NULL"""
        ).fetchall()
        for row in rows:
            price_now = prices.get(row["symbol"].upper())
            if price_now and row["price_at_pick"]:
                c.execute(
                    "UPDATE picks SET price_latest=?, return_pct=? WHERE id=?",
                    (
                        price_now,
                        round((price_now / row["price_at_pick"] - 1) * 100, 2),
                        row["id"],
                    ),
                )


def list_runs(limit: int = 20) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, week_start, status, summary, created_at, error FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_picks(run_id: int) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM picks WHERE run_id=? ORDER BY rank", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def latest_completed_run() -> dict | None:
    with get_conn() as c:
        row = c.execute(
            "SELECT id FROM runs WHERE status='completed' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def is_run_in_progress() -> bool:
    with get_conn() as c:
        row = c.execute("SELECT id FROM runs WHERE status='running' LIMIT 1").fetchone()
        return row is not None
