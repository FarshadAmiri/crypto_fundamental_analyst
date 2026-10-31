"""Pydantic AI validator: turns the crew's free-text consensus into a typed WeeklyVerdict.

Pydantic AI retries automatically when the model's output fails schema validation,
so the graph gets a guaranteed-typed verdict (or a clean failure it can fall back from).
"""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app import config
from app.schemas import WeeklyVerdict


def _model() -> OpenAIChatModel:
    return OpenAIChatModel(
        config.MODEL_NAME,
        provider=OpenAIProvider(base_url=config.VLLM_BASE_URL, api_key=config.LLM_API_KEY),
    )


validator_agent = Agent(
    _model(),
    output_type=WeeklyVerdict,
    system_prompt=(
        "You are a strict output formatter. You receive a debate transcript that ends with a "
        "proposed weekly crypto ranking. Convert it into the required structured format: "
        "week_start (ISO Monday date you are told), ranked picks (rank, coin, symbol, score 0-10, "
        "confidence 0-1, rationale, risks) and a one-paragraph summary. Do not invent coins that "
        "are not in the transcript."
    ),
    retries=2,
)


async def validate_verdict(debate_text: str, week_start: str, fallback_reports: list[dict]) -> WeeklyVerdict:
    """Try to validate the crew output; on failure, synthesize a verdict from per-coin reports."""
    try:
        result = await validator_agent.run(
            f"week_start: {week_start}\n\nDEBATE TRANSCRIPT:\n{debate_text}"
        )
        verdict = result.output
        if verdict.picks:
            return verdict
    except Exception:
        pass

    # Fallback: rank directly from per-coin reports so a bad crew output can't lose the week
    ranked = sorted(
        fallback_reports, key=lambda r: r.get("score", 0) * max(r.get("confidence", 0), 0.1),
        reverse=True,
    )[:5]
    from app.schemas import Pick

    return WeeklyVerdict(
        week_start=week_start,
        picks=[
            Pick(
                rank=i + 1,
                coin=r.get("coin", "?"),
                symbol=r.get("symbol", "?"),
                score=r.get("score", 0),
                confidence=r.get("confidence", 0),
                rationale=(r.get("fundamentals", "") or "")[:500],
                risks=(r.get("signals", "") or "no specific risks found")[:300],
            )
            for i, r in enumerate(ranked)
        ],
        summary="Fallback ranking derived from per-coin analyst reports (crew output failed validation).",
    )
