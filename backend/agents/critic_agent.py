"""
Critic Agent — Self-evaluation pipeline.
This is the differentiator: most RAG apps have no eval.
ResearchMind knows HOW GOOD its own outputs are.
"""
from backend.utils.metrics import evaluate_report, EvalResult
from backend.core.config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()


async def run_critic_agent(
    query: str,
    sub_queries: list[str],
    report: str,
    sources: list[dict],
) -> EvalResult | None:
    """
    Run the full eval pipeline on the generated report.
    Returns None if eval is disabled in settings.
    """
    if not settings.run_eval:
        logger.info("Eval disabled, skipping critic agent")
        return None

    logger.info("Running critic agent", query=query)

    try:
        result = await evaluate_report(
            query=query,
            sub_queries=sub_queries,
            report=report,
            sources=sources,
        )
        logger.info(
            "Eval complete",
            faithfulness=result.faithfulness,
            coverage=result.coverage,
            hallucination_rate=result.hallucination_rate,
            quality=result.quality_score,
        )
        return result
    except Exception as e:
        logger.error("Critic agent failed", error=str(e))
        return None
