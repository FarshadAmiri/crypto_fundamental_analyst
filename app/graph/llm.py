"""Single place where the vLLM endpoint is wired into each framework (LangChain / CrewAI / Pydantic AI)."""
from langchain_openai import ChatOpenAI

from app import config


def chat_llm(temperature: float | None = None) -> ChatOpenAI:
    """LangChain-compatible client pointing at the local vLLM OpenAI endpoint."""
    return ChatOpenAI(
        model=config.MODEL_NAME,
        base_url=config.VLLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=config.LLM_MAX_TOKENS,
    )
