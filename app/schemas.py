"""Typed contracts between pipeline stages. These are the shapes the Pydantic validator enforces."""
from pydantic import BaseModel, Field


class CoinPick(BaseModel):
    """A single coin in the triage shortlist."""
    coin: str = Field(description="Full coin name, e.g. 'Ethereum'")
    symbol: str = Field(description="Ticker symbol, e.g. 'ETH'")
    reason: str = Field(description="Why this coin made the shortlist this week, one sentence")


class TriageResult(BaseModel):
    coins: list[CoinPick] = Field(description="Shortlist of coins to analyze this week")


class CoinReport(BaseModel):
    """Per-coin analyst output (one per shortlisted coin)."""
    coin: str
    symbol: str
    fundamentals: str = Field(description="2-4 sentences on fundamentals found in the evidence")
    sentiment: str = Field(description="2-4 sentences on community sentiment found in the evidence")
    signals: str = Field(description="Notable news/investment/whale signals found in the evidence, or 'none'")
    score: float = Field(ge=0, le=10, description="Overall attractiveness 0-10 for the coming week")
    confidence: float = Field(ge=0, le=1, description="How confident, given evidence quality")


class Pick(BaseModel):
    """Final ranked pick for the week."""
    rank: int = Field(ge=1)
    coin: str
    symbol: str
    score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    rationale: str
    risks: str


class WeeklyVerdict(BaseModel):
    """The validated final output of the whole pipeline."""
    week_start: str = Field(description="ISO date of the Monday this verdict applies to")
    picks: list[Pick] = Field(description="Ranked picks, best first")
    summary: str = Field(description="One-paragraph market summary for the week")
