from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY") 
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")

    # Search
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    max_search_results: int = Field(default=8, alias="MAX_SEARCH_RESULTS")

    # Vector Store
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    max_chunks_per_doc: int = Field(default=20, alias="MAX_CHUNKS_PER_DOC")

    # Research
    research_depth: str = Field(default="deep", alias="RESEARCH_DEPTH")

    # API
    cors_origins: str = Field(
        default="http://localhost:5173", alias="CORS_ORIGINS"
    )

    # Eval
    eval_judge_model: str = Field(default="gpt-4o-mini", alias="EVAL_JUDGE_MODEL")
    run_eval: bool = Field(default=True, alias="RUN_EVAL")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="research-mind", alias="LANGCHAIN_PROJECT")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def depth_config(self) -> dict:
        configs = {
            "quick": {"max_sources": 3, "sub_queries": 2, "max_tokens": 800},
            "standard": {"max_sources": 5, "sub_queries": 4, "max_tokens": 1500},
            "deep": {"max_sources": 8, "sub_queries": 6, "max_tokens": 3000},
        }
        return configs.get(self.research_depth, configs["standard"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
