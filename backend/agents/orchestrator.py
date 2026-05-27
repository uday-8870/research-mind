"""
Orchestrator — LangGraph state machine that coordinates all agents.
This is the brain of the system.

State flow:
SEARCH → READ → SYNTHESIZE → EVALUATE → DONE
"""
import uuid
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from backend.agents.search_agent import run_search_agent
from backend.agents.reader_agent import run_reader_agent
from backend.agents.synthesis_agent import run_synthesis_agent
from backend.agents.critic_agent import run_critic_agent
from backend.utils.metrics import EvalResult
from backend.core.config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()


# ── State Definition ────────────────────────────────────────────
class ResearchState(TypedDict):
    task_id: str
    query: str
    depth: str
    depth_config: dict

    # Populated by agents
    sub_queries: list[str]
    sources: list[dict]
    enriched_sources: list[dict]
    report: str
    eval_result: Optional[dict]

    # Progress tracking
    status: str
    progress: int  # 0-100
    error: Optional[str]


# ── Agent Nodes ──────────────────────────────────────────────────
async def search_node(state: ResearchState) -> dict:
    logger.info("Search node executing", task_id=state["task_id"])
    try:
        sub_queries, sources = await run_search_agent(
            query=state["query"],
            depth_config=state["depth_config"],
        )
        return {
            "sub_queries": sub_queries,
            "sources": sources,
            "status": "reading",
            "progress": 35,
        }
    except Exception as e:
        return {"error": str(e), "status": "failed", "progress": 0}


async def reader_node(state: ResearchState) -> dict:
    logger.info("Reader node executing", task_id=state["task_id"])
    try:
        enriched = await run_reader_agent(
            sources=state["sources"],
            task_id=state["task_id"],
        )
        return {
            "enriched_sources": enriched,
            "status": "synthesizing",
            "progress": 60,
        }
    except Exception as e:
        return {"error": str(e), "status": "failed", "progress": 0}


async def synthesis_node(state: ResearchState) -> dict:
    logger.info("Synthesis node executing", task_id=state["task_id"])
    try:
        report = await run_synthesis_agent(
            query=state["query"],
            sub_queries=state["sub_queries"],
            sources=state["enriched_sources"],
            task_id=state["task_id"],
            depth_config=state["depth_config"],
        )
        return {
            "report": report,
            "status": "evaluating",
            "progress": 85,
        }
    except Exception as e:
        return {"error": str(e), "status": "failed", "progress": 0}


async def critic_node(state: ResearchState) -> dict:
    logger.info("Critic node executing", task_id=state["task_id"])
    try:
        eval_result = await run_critic_agent(
            query=state["query"],
            sub_queries=state["sub_queries"],
            report=state["report"],
            sources=state["enriched_sources"],
        )
        eval_dict = None
        if eval_result:
            eval_dict = {
                "faithfulness": eval_result.faithfulness,
                "coverage": eval_result.coverage,
                "hallucination_rate": eval_result.hallucination_rate,
                "quality_score": eval_result.quality_score,
                "reasoning": eval_result.reasoning,
                "flagged_claims": eval_result.flagged_claims,
            }
        return {
            "eval_result": eval_dict,
            "status": "done",
            "progress": 100,
        }
    except Exception as e:
        # Eval failure shouldn't kill the whole task
        logger.warning("Critic failed gracefully", error=str(e))
        return {"eval_result": None, "status": "done", "progress": 100}


# ── Conditional Routing ──────────────────────────────────────────
def route_after_search(state: ResearchState) -> str:
    if state.get("error") or not state.get("sources"):
        return END
    return "reader"


def route_after_reader(state: ResearchState) -> str:
    if state.get("error") or not state.get("enriched_sources"):
        return END
    return "synthesis"


def route_after_synthesis(state: ResearchState) -> str:
    if state.get("error"):
        return END
    return "critic"


# ── Build Graph ──────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("search")
    graph.add_conditional_edges("search", route_after_search, {"reader": "reader", END: END})
    graph.add_conditional_edges("reader", route_after_reader, {"synthesis": "synthesis", END: END})
    graph.add_conditional_edges("synthesis", route_after_synthesis, {"critic": "critic", END: END})
    graph.add_edge("critic", END)

    return graph.compile()


RESEARCH_GRAPH = build_research_graph()


async def run_research(query: str, depth: str = "deep") -> dict:
    """
    Entry point for the full research pipeline.
    Returns final state dict.
    """
    task_id = str(uuid.uuid4())
    cfg = get_settings().depth_config

    initial_state: ResearchState = {
        "task_id": task_id,
        "query": query,
        "depth": depth,
        "depth_config": cfg,
        "sub_queries": [],
        "sources": [],
        "enriched_sources": [],
        "report": "",
        "eval_result": None,
        "status": "searching",
        "progress": 10,
        "error": None,
    }

    final_state = await RESEARCH_GRAPH.ainvoke(initial_state)
    return final_state
