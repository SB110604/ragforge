"""Query rewriting, hybrid retrieval (vector + BM25 fused with Reciprocal Rank Fusion) and reranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .llm import LLM, LLMError
from .store import VectorStore
from .text import STOPWORDS, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None
    matched_queries: list[str] = field(default_factory=list)


# ---- query rewriting -------------------------------------------------------------------------------------------------
_FILLER = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?(tell me|explain|describe|show me|what is|what are|how do i|"
    r"how does|how to)\s+",
    re.IGNORECASE,
)


class QueryRewriter:
    """Produces the queries to search with.

    Rule-based (always): the original question, a keyword-only form without filler words, synonym/acronym expansions
    from ``glossary``, and sub-questions for comparisons ("A vs B", "difference between A and B").
    With an ``llm``: up to ``max_llm_queries`` extra reformulations from the model (failures are ignored).
    """

    def __init__(
        self, glossary: dict[str, str] | None = None, llm: LLM | None = None, max_llm_queries: int = 3
    ) -> None:
        self.glossary = {k.lower(): v for k, v in (glossary or {}).items()}
        self.llm, self.max_llm = llm, max_llm_queries

    def rewrite(self, question: str) -> list[str]:
        q = question.strip()
        out = [q]
        keywords = " ".join(
            w for w in re.findall(r"[\w'-]+", _FILLER.sub("", q).rstrip("?")) if w.lower() not in STOPWORDS
        )
        out.append(keywords)
        for term, expansion in self.glossary.items():
            if re.search(rf"\b{re.escape(term)}\b", q, re.IGNORECASE):
                out.append(re.sub(rf"\b{re.escape(term)}\b", expansion, keywords, flags=re.IGNORECASE))
        m = re.search(
            r"(?:difference between|compare)\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+?)[?.]*$", q, re.IGNORECASE
        ) or re.search(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.]*$", q, re.IGNORECASE)
        if m:
            out += [m.group(1).strip(), m.group(2).strip()]
        if self.llm is not None:
            out += self._llm_rewrites(q)
        seen, unique = set(), []
        for s in out:
            key = s.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(s.strip())
        return unique

    def _llm_rewrites(self, q: str) -> list[str]:
        prompt = (
            f"Rewrite the question into up to {self.max_llm} short, different queries for a document search engine. "
            f"Reply with a JSON list of strings only.\nQuestion: {q}"
        )
        try:
            reply = self.llm.complete([{"role": "user", "content": prompt}])  # type: ignore[union-attr]
            m = re.search(r"\[.*\]", reply, re.DOTALL)
            items = json.loads(m.group(0)) if m else []
            return [str(s) for s in items if isinstance(s, str)][: self.max_llm]
        except (LLMError, ValueError):
            return []


# ---- reranking -------------------------------------------------------------------------------------------------------
class Reranker(Protocol):
    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]: ...


class OverlapReranker:
    """Lightweight reranker: query-term coverage, term proximity and the fused retrieval score."""

    def __init__(
        self, coverage_weight: float = 0.6, proximity_weight: float = 0.2, retrieval_weight: float = 0.2
    ) -> None:
        self.w = (coverage_weight, proximity_weight, retrieval_weight)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        q = set(tokenize(question))
        if not hits:
            return hits
        top = max(h.score for h in hits) or 1.0
        for h in hits:
            toks = tokenize(h.chunk.context_text)
            present = q & set(toks)
            coverage = len(present) / len(q) if q else 0.0
            proximity = 0.0
            if len(present) > 1:
                pos = [i for i, t in enumerate(toks) if t in present]
                best = len(toks)
                for i in range(len(pos)):  # smallest window holding all present terms
                    seen: set[str] = set()
                    for j in range(i, len(pos)):
                        seen.add(toks[pos[j]])
                        if seen == present:
                            best = min(best, pos[j] - pos[i] + 1)
                            break
                proximity = len(present) / best
            h.rerank_score = round(self.w[0] * coverage + self.w[1] * proximity + self.w[2] * h.score / top, 6)
        return sorted(hits, key=lambda h: -(h.rerank_score or 0))


class CrossEncoderReranker:
    """Neural cross-encoder via fastembed (``pip install ragforge[fastembed]``)."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("install the extra: pip install 'ragforge[fastembed]'") from e
        self._model = TextCrossEncoder(model_name=model)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:  # pragma: no cover - needs model download
        scores = list(self._model.rerank(question, [h.chunk.context_text for h in hits]))
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: -(h.rerank_score or 0))


# ---- hybrid retrieval ------------------------------------------------------------------------------------------------
class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        rrf_k: int = 60,
        candidates: int = 30,
    ) -> None:
        self.store, self.embedder = store, embedder
        self.vw, self.kw, self.rrf_k, self.candidates = vector_weight, keyword_weight, rrf_k, candidates

    def search(self, queries: list[str], k: int = 5, doc_ids: list[str] | None = None) -> list[Hit]:
        n = len(self.store)
        if not n:
            return []
        allowed = np.ones(n, dtype=bool)
        if doc_ids:
            allowed = np.array([c.doc_id in doc_ids for c in self.store.chunks])
        fused: dict[int, Hit] = {}
        vecs = self.embedder.embed(queries)
        for q, vec in zip(queries, vecs, strict=True):
            for kind, scores, weight in (
                ("vector", self.store.vector_scores(vec), self.vw),
                ("keyword", self.store.bm25.scores(q), self.kw),
            ):
                scores = np.where(allowed, scores, -np.inf)
                order = [i for i in np.argsort(-scores)[: self.candidates] if np.isfinite(scores[i]) and scores[i] > 0]
                for rank, i in enumerate(order, start=1):
                    h = fused.setdefault(int(i), Hit(self.store.chunks[i], 0.0))
                    h.score += weight / (self.rrf_k + rank)
                    if kind == "vector":
                        h.vector_rank = rank if h.vector_rank is None else min(h.vector_rank, rank)
                    else:
                        h.keyword_rank = rank if h.keyword_rank is None else min(h.keyword_rank, rank)
                    if q not in h.matched_queries:
                        h.matched_queries.append(q)
        hits = sorted(fused.values(), key=lambda h: -h.score)
        return hits[: max(k, 1) * 4]
