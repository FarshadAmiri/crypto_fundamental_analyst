"""CrewAI debate node: three personas (bull, bear, quant) discuss the shortlist and propose a ranking.

All agents share the same local vLLM endpoint — the diversity comes from roles, not models.
Kept deliberately small: 3 tasks total, tools disabled, output is raw text that the
Pydantic AI validator turns into a typed WeeklyVerdict.
"""
import asyncio

from crewai import Agent, Crew, LLM, Process, Task

from app import config


def _build_debate_context(shortlist: list[dict], reports: list[dict]) -> str:
    reports_text = ""
    for r in reports:
        reports_text += (
            f"\n### {r.get('coin')} ({r.get('symbol')}) — score {r.get('score')}/10, "
            f"confidence {r.get('confidence')}\n"
            f"Fundamentals: {r.get('fundamentals')}\n"
            f"Sentiment: {r.get('sentiment')}\n"
            f"Signals: {r.get('signals')}\n"
        )
    shortlist_text = "\n".join(
        f"- {s['coin']} ({s['symbol']}): shortlisted because {s['reason']}" for s in shortlist
    )
    return f"SHORTLIST:\n{shortlist_text}\n\nPER-COIN ANALYST REPORTS:{reports_text or '(none survived analysis)'}"


def _crew_llm() -> LLM:
    return LLM(
        model=f"openai/{config.MODEL_NAME}",
        base_url=config.VLLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=0.4,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def run_debate_sync(shortlist: list[dict], reports: list[dict]) -> str:
    context = _build_debate_context(shortlist, reports)
    llm = _crew_llm()

    bull = Agent(
        role="Bullish Analyst",
        goal="Argue the strongest case for the 3-5 coins with the best upside potential next week",
        backstory="An experienced crypto fund analyst who hunts for asymmetric upside and never "
        "argues from vibes — only from the evidence in front of him.",
        llm=llm,
        allow_delegation=False,
    )
    bear = Agent(
        role="Bearish Analyst",
        goal="Stress-test the bull case: highlight risks, weak evidence and downside scenarios",
        backstory="A risk manager burned by three bull-market collapses. Trusts nothing without "
        "evidence and always assigns explicit risks.",
        llm=llm,
        allow_delegation=False,
    )
    quant = Agent(
        role="Quant Moderator",
        goal="Weigh both positions and produce ONE final ranking of the 3-5 most promising coins",
        backstory="A systematic portfolio manager who synthesizes bull/bear debate into an "
        "actionable weekly ranking with clear reasoning.",
        llm=llm,
        allow_delegation=False,
    )

    task_bull = Task(
        description=(
            "Below is this week's shortlist and per-coin analyst reports.\n\n"
            f"{context}\n\n"
            "Pick the 3-5 coins with the best expected upside for the coming week and argue why, "
            "based strictly on the evidence above."
        ),
        expected_output="A bulleted bull case for 3-5 coins, each with a 0-10 score.",
        agent=bull,
    )
    task_bear = Task(
        description=(
            "Review the bull case above. For each proposed coin, state the main risk or the weakest "
            "part of the evidence, and flag any coin that should be dropped."
        ),
        expected_output="A bulleted bear rebuttal: risks per coin, plus coins to drop.",
        agent=bear,
        context=[task_bull],
    )
    task_final = Task(
        description=(
            "Synthesize the bull case and the bear rebuttal into ONE final ranking of the 3-5 best "
            "coins for next week. For each: score 0-10, confidence 0-1, one-paragraph rationale, "
            "and the main risk. End with a one-paragraph overall market summary."
        ),
        expected_output=(
            "A numbered ranking (rank. SYMBOL — score X/10 — confidence Y — rationale — main risk), "
            "followed by 'SUMMARY:' and a one-paragraph market summary."
        ),
        agent=quant,
        context=[task_bull, task_bear],
    )

    crew = Crew(
        agents=[bull, bear, quant],
        tasks=[task_bull, task_bear, task_final],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


async def run_debate(shortlist: list[dict], reports: list[dict]) -> str:
    """CrewAI's kickoff is blocking; run it in a thread so the graph stays async."""
    return await asyncio.to_thread(run_debate_sync, shortlist, reports)
