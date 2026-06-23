from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from backend.core.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.1) -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "groq":
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )
    elif settings.llm_provider == "anthropic":
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
    settings = get_settings()
    if settings.llm_provider == "groq":
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )
    return ChatOpenAI(
        model=settings.eval_judge_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )