from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DepthEnum(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="Research question")
    depth: DepthEnum = Field(default=DepthEnum.deep, description="Research depth")
    domain: Optional[str] = Field(default=None, description="Optional domain hint")


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int


class EvalScores(BaseModel):
    faithfulness: float
    coverage: float
    hallucination_rate: float
    quality_score: float
    reasoning: str
    flagged_claims: list[str]


class SourceMeta(BaseModel):
    url: str
    title: str
    score: float


class ResearchResult(BaseModel):
    task_id: str
    query: str
    sub_queries: list[str]
    report: str
    sources: list[SourceMeta]
    eval_scores: Optional[EvalScores]
    status: str
    error: Optional[str]


class HealthResponse(BaseModel):
    status: str
    version: str
