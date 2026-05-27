"""
Synthesis Agent — Generates a structured, cited research report
using retrieved context and source documents.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from backend.core.llm import get_llm
from backend.core.memory import similarity_search
import structlog

logger = structlog.get_logger()


SYNTHESIS_SYSTEM = """You are an expert research synthesizer. Your job is to write clear, 
comprehensive research reports that are:
- Factually grounded in the provided sources
- Well-structured with clear sections
- Properly cited (use [Source N] format)
- Honest about uncertainty or knowledge gaps

Always cite your sources. Never make claims not supported by the provided context."""


SYNTHESIS_PROMPT = """Research Question: {query}

Sub-questions to address:
{sub_queries}

Retrieved Source Documents:
{context}

Write a comprehensive research report that:
1. Opens with a concise executive summary (2-3 sentences)
2. Has clearly labeled sections addressing each sub-question
3. Cites sources as [Source N: URL]
4. Ends with a "Key Takeaways" section (bullet points)
5. Flags any areas where sources were limited or conflicting

Format in clean Markdown. Be thorough but not padded."""


def build_context_from_sources(sources: list[dict], max_chars: int = 8000) -> str:
    """Build a formatted context string from source documents."""
    context_parts = []
    total_chars = 0

    for i, source in enumerate(sources):
        content = source.get("content", "")[:1500]
        url = source.get("url", "")
        title = source.get("title", "")
        part = f"[Source {i+1}: {url}]\nTitle: {title}\n{content}\n"

        if total_chars + len(part) > max_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n---\n".join(context_parts)


async def augment_context_with_rag(
    query: str,
    sources: list[dict],
    task_id: str,
) -> str:
    """
    Augment source context with similarity-searched chunks from vector store.
    This handles cases where the best answer is buried deep in a source.
    """
    try:
        from backend.core.memory import similarity_search
        chunks = similarity_search(
            query=query,
            n_results=8,
            collection_name=f"task_{task_id}",
        )
        rag_parts = [f"[Relevant Chunk]\n{c['text']}" for c in chunks[:5]]
        return "\n\n".join(rag_parts)
    except Exception:
        return ""


async def run_synthesis_agent(
    query: str,
    sub_queries: list[str],
    sources: list[dict],
    task_id: str,
    depth_config: dict,
) -> str:
    """
    Generate the final research report.
    Combines direct source context with RAG-retrieved chunks.
    """
    llm = get_llm(temperature=0.2)

    # Build context
    source_context = build_context_from_sources(
        sources, max_chars=depth_config["max_tokens"] * 4
    )
    rag_context = await augment_context_with_rag(query, sources, task_id)

    full_context = source_context
    if rag_context:
        full_context += f"\n\n--- Additional Retrieved Context ---\n{rag_context}"

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        sub_queries="\n".join(f"- {q}" for q in sub_queries),
        context=full_context,
    )

    logger.info("Synthesizing report", query=query, sources=len(sources))

    resp = await llm.ainvoke([
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    report = resp.content
    logger.info("Report generated", chars=len(report))
    return report
