"""
core/retrieval/vector_store.py
─────────────────────────────────────────────────────────────────────────────
ChromaDB interface for project-scoped vector storage and retrieval.

Design:
  - One ChromaDB collection per project namespace (derived from project path).
  - Embeddings are stored with metadata: source_file, line_start, line_end,
    source_type, and the plain-text chunk itself.
  - Querying returns ranked results with raw cosine distances exposed for
    confidence scoring by ranker.py.
  - The persistent database is stored locally at BAKUP_CHROMA_DIR.

ChromaDB distance metric: "cosine"
  distance = 1 - cosine_similarity  (0 = identical, 1 = orthogonal)
  With L2-normalised embeddings this is exact and efficient.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.ingestion.chunker import Chunk
from core.retrieval.models import RetrievedChunk

# ── Singleton client ───────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        persist_dir = os.environ.get("BAKUP_CHROMA_DIR", "vectordb")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


# ── Collection naming ──────────────────────────────────────────────────────────

def _collection_name(namespace: str) -> str:
    """
    Derive a stable, safe ChromaDB collection name from a project namespace.
    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens.
    """
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:16]
    return f"bakup-{digest}"


def _get_or_create_collection(namespace: str) -> chromadb.Collection:
    client = _get_client()
    name = _collection_name(namespace)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Write ──────────────────────────────────────────────────────────────────────

def add_chunks(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    namespace: str,
    batch_size: int = 200,
) -> int:
    """
    Upsert chunks and their embeddings into the collection for namespace.
    Returns the count of chunks stored.

    Uses upsert so re-indexing a project does not create duplicate entries.
    IDs are derived deterministically from (namespace, source_file, line_start).
    """
    if not chunks:
        return 0

    collection = _get_or_create_collection(namespace)
    stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        vecs: List[List[float]] = []

        for chunk, vec in zip(batch_chunks, batch_embeddings):
            chunk_id = hashlib.sha256(
                f"{namespace}:{chunk.source_file}:{chunk.line_start}:{chunk.line_end}".encode()
            ).hexdigest()

            ids.append(chunk_id)
            docs.append(chunk.text)
            meta_dict = {
                "source_file": chunk.source_file,
                "line_start":  chunk.line_start,
                "line_end":    chunk.line_end,
                "source_type": chunk.source_type,
                "file_name":   chunk.file_name or "",
                "severity":    chunk.severity or "info",
            }
            if chunk.detected_timestamp:
                meta_dict["detected_timestamp"] = chunk.detected_timestamp
            if chunk.last_modified:
                meta_dict["last_modified"] = chunk.last_modified
            # Code-aware metadata
            if getattr(chunk, 'language', ''):
                meta_dict["language"] = chunk.language
            if getattr(chunk, 'function_name', ''):
                meta_dict["function_name"] = chunk.function_name
            if getattr(chunk, 'class_name', ''):
                meta_dict["class_name"] = chunk.class_name
            if getattr(chunk, 'chunk_kind', ''):
                meta_dict["chunk_kind"] = chunk.chunk_kind
            if getattr(chunk, 'docstring', ''):
                meta_dict["docstring"] = chunk.docstring[:500]
            if getattr(chunk, 'imports', ''):
                meta_dict["imports"] = chunk.imports[:500]
            metas.append(meta_dict)
            vecs.append(vec)

        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=vecs,
            metadatas=metas,
        )
        stored += len(batch_chunks)

    return stored


def delete_namespace(namespace: str) -> None:
    """Remove all chunks stored under a given namespace."""
    client = _get_client()
    name = _collection_name(namespace)
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ── Read ───────────────────────────────────────────────────────────────────────


def query_chunks(
    query_embedding: List[float],
    namespace: str,
    top_k: int = 8,
) -> List[RetrievedChunk]:
    """
    Return the top_k most similar chunks to query_embedding in namespace.
    Results are ordered ascending by distance (most similar first).
    Returns an empty list if the collection is empty or does not exist.
    """
    collection = _get_or_create_collection(namespace)
    count = collection.count()
    if count == 0:
        return []
    k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(RetrievedChunk(
            text=doc,
            source_file=meta.get("source_file", "unknown"),
            line_start=int(meta.get("line_start", 0)),
            line_end=int(meta.get("line_end", 0)),
            source_type=meta.get("source_type", "code"),
            distance=float(dist),
            file_name=meta.get("file_name", ""),
            severity=meta.get("severity", "info"),
            detected_timestamp=meta.get("detected_timestamp"),
            language=meta.get("language", ""),
            function_name=meta.get("function_name", ""),
            class_name=meta.get("class_name", ""),
            chunk_kind=meta.get("chunk_kind", ""),
            docstring=meta.get("docstring", ""),
            imports=meta.get("imports", ""),
        ))

    return chunks


def collection_count(namespace: str) -> int:
    """Return the number of chunks stored in a namespace."""
    return _get_or_create_collection(namespace).count()


# ── Diagnostic helpers ──────────────────────────────────────────────────────────

def collection_stats(namespace: str) -> dict:
    """Return diagnostic stats for a namespace: count, type breakdown, sample entries."""
    collection = _get_or_create_collection(namespace)
    count = collection.count()

    stats: dict = {
        "namespace":       namespace,
        "collection_name": _collection_name(namespace),
        "total_chunks":    count,
        "source_types":    {},
        "samples":         [],
    }

    if count == 0:
        return stats

    # Sample entries
    sample_size = min(10, count)
    peek = collection.peek(limit=sample_size)

    for doc, meta in zip(peek["documents"], peek["metadatas"]):
        stats["samples"].append({
            "source_file": meta.get("source_file", "unknown"),
            "source_type": meta.get("source_type", "unknown"),
            "line_start":  meta.get("line_start", 0),
            "line_end":    meta.get("line_end", 0),
            "text_preview": (doc or "")[:300],
        })

    # Source-type breakdown (requires scanning all metadata)
    all_data = collection.get(include=["metadatas"])
    for meta in all_data["metadatas"]:
        st = meta.get("source_type", "unknown")
        stats["source_types"][st] = stats["source_types"].get(st, 0) + 1

    return stats


def keyword_search(
    namespace: str,
    keywords: List[str],
    top_k: int = 10,
) -> List[RetrievedChunk]:
    """
    Search for chunks containing any of the given keywords using ChromaDB
    document-content filtering ($contains).

    Returns chunks with a fixed distance of 0.30 (~70% confidence) since
    keyword matching does not produce a similarity distance.

    Useful for log-style queries where embedding similarity may miss exact
    keyword matches like 'ERROR', 'Exception', 'Traceback'.
    """
    collection = _get_or_create_collection(namespace)
    if collection.count() == 0:
        return []

    seen_ids: set = set()
    chunks: List[RetrievedChunk] = []

    for kw in keywords:
        try:
            results = collection.get(
                where_document={"$contains": kw},
                include=["documents", "metadatas"],
                limit=top_k,
            )
        except Exception:
            continue

        if not results or not results.get("ids"):
            continue

        for doc_id, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            chunks.append(RetrievedChunk(
                text=doc,
                source_file=meta.get("source_file", "unknown"),
                line_start=int(meta.get("line_start", 0)),
                line_end=int(meta.get("line_end", 0)),
                source_type=meta.get("source_type", "code"),
                distance=0.30,
                file_name=meta.get("file_name", ""),
                severity=meta.get("severity", "info"),
                detected_timestamp=meta.get("detected_timestamp"),
                language=meta.get("language", ""),
                function_name=meta.get("function_name", ""),
                class_name=meta.get("class_name", ""),
                chunk_kind=meta.get("chunk_kind", ""),
                docstring=meta.get("docstring", ""),
                imports=meta.get("imports", ""),
            ))

    return chunks[:top_k]


def severity_search(
    namespace: str,
    severity: str = "error",
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """
    Retrieve all chunks with a given severity tag from the collection.
    Useful for building cross-file error distribution reports.
    """
    collection = _get_or_create_collection(namespace)
    if collection.count() == 0:
        return []

    try:
        results = collection.get(
            where={"severity": severity},
            include=["documents", "metadatas"],
            limit=top_k,
        )
    except Exception:
        return []

    if not results or not results.get("ids"):
        return []

    chunks: List[RetrievedChunk] = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append(RetrievedChunk(
            text=doc,
            source_file=meta.get("source_file", "unknown"),
            line_start=int(meta.get("line_start", 0)),
            line_end=int(meta.get("line_end", 0)),
            source_type=meta.get("source_type", "code"),
            distance=0.20,  # Severity match = high relevance
            file_name=meta.get("file_name", ""),
            severity=meta.get("severity", "info"),
            detected_timestamp=meta.get("detected_timestamp"),
            language=meta.get("language", ""),
            function_name=meta.get("function_name", ""),
            class_name=meta.get("class_name", ""),
            chunk_kind=meta.get("chunk_kind", ""),
            docstring=meta.get("docstring", ""),
            imports=meta.get("imports", ""),
        ))

    return chunks


# ── Startup initialisation ─────────────────────────────────────────────────────

def init_vector_store() -> None:
    """
    Verify the ChromaDB persistent directory is writable and the client
    initialises without error. Called once at startup from main.py lifespan.
    """
    _get_client()
    persist_dir = os.environ.get("BAKUP_CHROMA_DIR", "vectordb")
    print(f"bakup: vector store OK ({persist_dir})")
