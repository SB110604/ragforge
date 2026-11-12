"""The RAG pipeline: ingest -> semantic chunks -> index; ask -> rewrite -> hybrid search -> rerank -> cited answer."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import SemanticChunker
from .embeddings import Embedder, HashingEmbedder
from .generate import Answer, ExtractiveGenerator, LLMGenerator
from .llm import LLM
from .loaders import Document, load_document
from .retrieval import Hit, HybridRetriever, OverlapReranker, QueryRewriter, Reranker
from .store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        index_path: str | Path = ":memory:",
        *,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        glossary: dict[str, str] | None = None,
        chunker: SemanticChunker | None = None,
        llm_query_rewrites: bool = False,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = VectorStore(index_path, self.embedder.name)
        self.chunker = chunker or SemanticChunker(self.embedder)
        self.retriever = HybridRetriever(self.store, self.embedder)
        self.reranker = reranker or OverlapReranker()
        self.rewriter = QueryRewriter(glossary, llm if llm_query_rewrites else None)
        self.generator = LLMGenerator(llm) if llm is not None else ExtractiveGenerator()

    def ingest(self, source: str | Path | Document) -> dict[str, Any]:
        doc = source if isinstance(source, Document) else load_document(source)
        chunks = self.chunker.chunk(doc)
        if not chunks:
            raise ValueError(f"{doc.source}: no text to index")
        self.store.add(chunks, self.embedder.embed([c.context_text for c in chunks]))
        return {"id": doc.id, "title": doc.title, "source": doc.source, "pages": len(doc.pages), "chunks": len(chunks)}

    def ingest_dir(
        self, directory: str | Path, patterns: tuple[str, ...] = ("*.pdf", "*.md", "*.txt", "*.html")
    ) -> list[dict]:
        files = sorted({p for pat in patterns for p in Path(directory).rglob(pat)})
        return [self.ingest(p) for p in files]

    def search(self, question: str, k: int = 4, doc_ids: list[str] | None = None) -> tuple[list[Hit], list[str]]:
        queries = self.rewriter.rewrite(question)
        hits = self.retriever.search(queries, k, doc_ids)
        return self.reranker.rerank(question, hits)[:k], queries

    def ask(self, question: str, k: int = 4, doc_ids: list[str] | None = None) -> Answer:
        t0 = time.perf_counter()
        hits, queries = self.search(question, k, doc_ids)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, hits)
        t2 = time.perf_counter()
        answer.queries = queries
        answer.timings_ms = {"retrieve": round((t1 - t0) * 1000, 1), "generate": round((t2 - t1) * 1000, 1)}
        return answer

    @staticmethod
    def answer_dict(a: Answer) -> dict[str, Any]:
        d = asdict(a)
        d["found"] = a.found
        return d
