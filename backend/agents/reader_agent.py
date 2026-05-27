"""
Reader Agent — Parses raw web content, chunks intelligently,
embeds into vector store for retrieval.
"""
import re
from bs4 import BeautifulSoup
from backend.utils.chunker import chunk_text
from backend.core.memory import upsert_chunks
from backend.core.config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()


def clean_html(raw: str) -> str:
    """Strip HTML tags and clean up whitespace."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    # Remove script/style
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Clean whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_content(source: dict) -> str:
    """
    Prefer raw_content (full page) over content (snippet).
    Falls back to snippet if raw is empty.
    """
    raw = source.get("raw_content", "")
    if raw and len(raw) > 500:
        return clean_html(raw)
    snippet = source.get("content", "")
    return snippet.strip()


async def run_reader_agent(
    sources: list[dict],
    task_id: str,
) -> list[dict]:
    """
    Process all sources:
    1. Extract and clean text
    2. Chunk intelligently
    3. Store in vector DB
    4. Return enriched source objects
    """
    enriched_sources = []
    all_chunks = []
    all_metadatas = []

    for source in sources:
        content = extract_content(source)
        if not content or len(content) < 100:
            logger.warning("Skipping thin source", url=source.get("url"))
            continue

        chunks = chunk_text(content, max_tokens=400, overlap_tokens=60)
        chunk_texts = [c.text for c in chunks[: settings.max_chunks_per_doc]]

        for chunk in chunk_texts:
            all_chunks.append(chunk)
            all_metadatas.append({
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "task_id": task_id,
                "source_query": source.get("source_query", ""),
            })

        enriched_sources.append({
            **source,
            "content": content[:2000],  # Truncate for report context
            "chunk_count": len(chunk_texts),
            "char_count": len(content),
        })

        logger.debug(
            "Processed source",
            url=source.get("url"),
            chunks=len(chunk_texts),
        )

    # Batch upsert to vector store
    if all_chunks:
        collection_name = f"task_{task_id}"
        inserted = upsert_chunks(all_chunks, all_metadatas, collection_name)
        logger.info("Chunks stored", count=inserted, task_id=task_id)

    return enriched_sources
