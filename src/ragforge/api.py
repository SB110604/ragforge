"""FastAPI service: upload documents, search, ask with citations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .loaders import LoaderError
from .pipeline import RAGPipeline

MAX_UPLOAD = 25 * 1024 * 1024
ALLOWED = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    doc_ids: list[str] | None = None


def create_app(rag: RAGPipeline, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI(
        title="ragforge",
        version="0.1.0",
        description="Hybrid-search RAG: semantic chunking, BM25 + vectors, reranking, cited answers.",
    )

    def auth(key: str | None) -> None:
        if api_key and key != api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "documents": len(rag.store.documents()),
            "chunks": len(rag.store),
            "embedder": rag.embedder.name,
            "generator": type(rag.generator).__name__,
        }

    @app.get("/documents")
    def documents(x_api_key: str | None = Header(default=None)) -> list[dict[str, Any]]:
        auth(x_api_key)
        return rag.store.documents()

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, f"unsupported file type {suffix!r}")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file larger than 25 MB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                return rag.ingest(path)
            except (LoaderError, ValueError) as e:
                raise HTTPException(422, str(e)) from e

    @app.delete("/documents/{doc_id}")
    def delete(doc_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        n = rag.store.delete_document(doc_id)
        if not n:
            raise HTTPException(404, "document not found")
        return {"deleted_chunks": n}

    @app.post("/search")
    def search(body: AskRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        hits, queries = rag.search(body.question, body.k, body.doc_ids)
        return {
            "queries": queries,
            "hits": [
                {
                    "chunk": rag.store.chunk_dict(h.chunk),
                    "score": round(h.score, 5),
                    "rerank_score": h.rerank_score,
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                }
                for h in hits
            ],
        }

    @app.post("/ask")
    def ask(body: AskRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        return rag.answer_dict(rag.ask(body.question, body.k, body.doc_ids))

    return app


def app_from_env() -> FastAPI:
    """App factory for ``uvicorn --factory ragforge.api:app_from_env`` (see .env.example)."""
    from .cli import build_pipeline

    return create_app(
        build_pipeline(os.getenv("RAGFORGE_INDEX", "data/index.db")), api_key=os.getenv("RAGFORGE_API_KEY") or None
    )
