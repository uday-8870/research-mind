"""
Search Agent — Web search via Tavily with query decomposition.
Generates sub-queries, runs parallel searches, deduplicates results.
"""
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import AsyncTavilyClient
from backend.core.config import get_settings
from backend.core.llm import get_llm
import structlog
import json, re

logger = structlog.get_logger()
settings = get_settings()


DECOMPOSE_PROMPT = """You are a research assistant. Break down this research question into {n} specific, 
searchable sub-queries that together will fully answer the main question.

Main Question: {query}

Requirements:
- Each sub-query should be independently searchable
- Cover different aspects of the topic
- Be specific and concrete (not vague)
- Avoid redundancy

Respond ONLY with a JSON array of strings:
["sub-query 1", "sub-query 2", ...]"""


async def decompose_query(query: str, n_sub: int = 4) -> list[str]:
    """Use LLM to break query into targeted sub-queries."""
    llm = get_llm(temperature=0.3)
    resp = await llm.ainvoke([
        SystemMessage(content="You generate precise search queries. Respond only in JSON."),
        HumanMessage(content=DECOMPOSE_PROMPT.format(query=query, n=n_sub)),
    ])
    text = re.sub(r"```json|```", "", resp.content).strip()
    try:
        sub_queries = json.loads(text)
        return [q for q in sub_queries if isinstance(q, str)][:n_sub]
    except Exception:
        logger.warning("Failed to parse sub-queries, using original query")
        return [query]


async def search_single(client: AsyncTavilyClient, query: str, max_results: int) -> list[dict]:
    """Search Tavily for a single query."""
    try:
        result = await client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
        )
        sources = []
        for r in result.get("results", []):
            sources.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "raw_content": r.get("raw_content", ""),
                "score": r.get("score", 0.0),
                "source_query": query,
            })
        return sources
    except Exception as e:
        logger.error("Search failed", query=query, error=str(e))
        return []


async def run_search_agent(
    query: str,
    depth_config: dict,
) -> tuple[list[str], list[dict]]:
    """
    Full search pipeline:
    1. Decompose query into sub-queries
    2. Run parallel searches
    3. Deduplicate by URL
    4. Return (sub_queries, sources)
    """
    n_sub = depth_config["sub_queries"]
    max_sources = depth_config["max_sources"]

    logger.info("Decomposing query", query=query, n_sub=n_sub)
    sub_queries = await decompose_query(query, n_sub)
    logger.info("Sub-queries generated", sub_queries=sub_queries)

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    per_query = max(2, max_sources // len(sub_queries))

    tasks = [search_single(client, q, per_query) for q in sub_queries]
    results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    all_sources: list[dict] = []
    for batch in results:
        for source in batch:
            url = source["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(source)

    # Sort by relevance score
    all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_sources = all_sources[:max_sources]

    logger.info("Search complete", n_sources=len(all_sources))
    return sub_queries, all_sources
