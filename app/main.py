"""FastAPI app: serves the web UI, the JSON API, the manual run trigger, and the weekly scheduler."""
import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import config, db
from app.graph.pipeline import run_pipeline

_scheduler: AsyncIOScheduler | None = None
_run_lock = asyncio.Lock()


async def scheduled_run():
    if db.is_run_in_progress():
        return
    async with _run_lock:
        await run_pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    global _scheduler
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        scheduled_run,
        CronTrigger(
            day_of_week=config.SCHEDULER_DAY_OF_WEEK, hour=config.SCHEDULER_HOUR, minute=0
        ),
        id="weekly-research",
    )
    _scheduler.start()
    yield
    _scheduler.shutdown(wait=False)


app = FastAPI(title="Crypto Fundamental Analyst", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/api/picks/{run_id}")
def picks(run_id: int):
    return db.get_picks(run_id)


@app.get("/api/runs")
def runs():
    return db.list_runs()


@app.get("/api/latest")
def latest():
    run = db.latest_completed_run()
    if not run:
        return {"run": None, "picks": []}
    return {"run": run, "picks": db.get_picks(run["id"])}


@app.get("/api/status")
def status():
    return {"run_in_progress": db.is_run_in_progress()}


@app.post("/api/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if db.is_run_in_progress():
        raise HTTPException(status_code=409, detail="A run is already in progress")
    background_tasks.add_task(_guarded_run)
    return {"started": True}


async def _guarded_run():
    if db.is_run_in_progress():
        return
    async with _run_lock:
        try:
            await run_pipeline()
        except Exception:
            pass  # failure already recorded on the run row
