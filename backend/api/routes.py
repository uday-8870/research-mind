"""
API Routes — REST endpoints for ResearchMind.
Supports both blocking and streaming (SSE) modes.
"""
import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from backend.api.schemas import (
    ResearchRequest,
    ResearchResult,
    EvalScores,
    SourceMeta,
    HealthResponse,
    TaskStatus,
)
from backend.agents.orchestrator import run_research
import structlog

logger = structlog.get_logger()
router = APIRouter()

# In-memory task store (use Redis in production)
_tasks: dict[str, dict] = {}


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="1.0.0")


@router.post("/research", response_model=TaskStatus)
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a research task asynchronously. Returns task_id immediately."""
    import uuid
    task_id = str(uuid.uuid4())

    _tasks[task_id] = {"status": "queued", "progress": 0, "result": None}

    async def _run():
        try:
            _tasks[task_id]["status"] = "running"
            result = await run_research(query=req.query, depth=req.depth.value)
            _tasks[task_id]["result"] = result
            _tasks[task_id]["status"] = result.get("status", "done")
            _tasks[task_id]["progress"] = 100
        except Exception as e:
            logger.error("Task failed", task_id=task_id, error=str(e))
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(e)

    background_tasks.add_task(_run)

    return TaskStatus(task_id=task_id, status="queued", progress=0)


@router.get("/research/{task_id}", response_model=ResearchResult)
async def get_result(task_id: str):
    """Poll for completed result."""
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")

    task = _tasks[task_id]

    if task["status"] in ("queued", "running"):
        raise HTTPException(202, "Task still running")

    if task["status"] == "failed":
        raise HTTPException(500, task.get("error", "Unknown error"))

    result = task["result"]

    eval_scores = None
    if result.get("eval_result"):
        eval_scores = EvalScores(**result["eval_result"])

    sources = [
        SourceMeta(
            url=s.get("url", ""),
            title=s.get("title", ""),
            score=s.get("score", 0.0),
        )
        for s in result.get("enriched_sources", [])[:10]
    ]

    return ResearchResult(
        task_id=task_id,
        query=result["query"],
        sub_queries=result.get("sub_queries", []),
        report=result.get("report", ""),
        sources=sources,
        eval_scores=eval_scores,
        status=result.get("status", "done"),
        error=result.get("error"),
    )


@router.get("/research/{task_id}/status", response_model=TaskStatus)
async def get_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    task = _tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
    )


@router.post("/research/sync", response_model=ResearchResult)
async def research_sync(req: ResearchRequest):
    """
    Blocking research endpoint — waits for full result.
    Good for testing. Use async endpoint for production.
    """
    try:
        result = await run_research(query=req.query, depth=req.depth.value)
    except Exception as e:
        raise HTTPException(500, str(e))

    eval_scores = None
    if result.get("eval_result"):
        eval_scores = EvalScores(**result["eval_result"])

    sources = [
        SourceMeta(
            url=s.get("url", ""),
            title=s.get("title", ""),
            score=s.get("score", 0.0),
        )
        for s in result.get("enriched_sources", [])[:10]
    ]

    return ResearchResult(
        task_id=result["task_id"],
        query=result["query"],
        sub_queries=result.get("sub_queries", []),
        report=result.get("report", ""),
        sources=sources,
        eval_scores=eval_scores,
        status=result.get("status", "done"),
        error=result.get("error"),
    )
