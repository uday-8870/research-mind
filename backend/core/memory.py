import hashlib
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.core.config import get_settings

settings = get_settings()

_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


def get_or_create_collection(name: str = "research_docs") -> chromadb.Collection:
    return _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: list[str],
    metadatas: list[dict],
    collection_name: str = "research_docs",
) -> int:
    collection = get_or_create_collection(collection_name)
    embeddings = _embeddings.embed_documents(chunks)
    ids = [
        hashlib.md5(f"{m.get('url','')}{i}{c[:50]}".encode()).hexdigest()
        for i, (c, m) in enumerate(zip(chunks, metadatas))
    ]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def similarity_search(
    query: str,
    n_results: int = 6,
    collection_name: str = "research_docs",
    where: Optional[dict] = None,
) -> list[dict]:
    collection = get_or_create_collection(collection_name)
    query_embedding = _embeddings.embed_query(query)

    kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "distance": dist})
    return output