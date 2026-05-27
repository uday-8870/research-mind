"""
LLM Client Abstraction — supports OpenAI and Anthropic seamlessly.
Switch providers by changing LLM_PROVIDER in .env.
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from backend.core.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """Return the configured LLM client."""
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=4096,
        )


@lru_cache
def get_eval_llm() -> BaseChatModel:
    """Cheaper model for evaluation tasks."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.eval_judge_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
