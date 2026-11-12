"""Semantic chunking: split where the topic shifts, within a token budget.

Consecutive sentences are embedded; a boundary is placed where the similarity between a sentence and the
preceding window falls into the lowest ``breakpoint_percentile`` of the document — but only once the chunk has at
least ``min_tokens``. Chunks never exceed ``max_tokens`` (a single over-long sentence becomes its own chunk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .loaders import Document
from .text import count_tokens, split_sentences


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    source: str
    page: int
    heading: str | None
    index: int
    text: str

    @property
    def context_text(self) -> str:
        """Text used for embedding and keyword search: document title and section heading give context."""
        head = f"{self.doc_title} — {self.heading}" if self.heading else self.doc_title
        return f"{head}\n{self.text}"


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        *,
        max_tokens: int = 180,
        min_tokens: int = 40,
        breakpoint_percentile: float = 25.0,
        window: int = 2,
        overlap_sentences: int = 1,
    ) -> None:
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        self.embedder, self.max_tokens, self.min_tokens = embedder, max_tokens, min_tokens
        self.percentile, self.window, self.overlap = breakpoint_percentile, window, overlap_sentences

    def _boundaries(self, sentences: list[str]) -> set[int]:
        """Indices *before which* a topic shift is detected."""
        if len(sentences) < 3:
            return set()
        e = self.embedder.embed(sentences)
        sims = []
        for i in range(1, len(sentences)):
            prev = e[max(0, i - self.window) : i].mean(axis=0)
            n = np.linalg.norm(prev)
            sims.append(float(e[i] @ (prev / n)) if n else 0.0)
        cut = float(np.percentile(sims, self.percentile))
        return {i for i, s in enumerate(sims, start=1) if s <= cut}

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in doc.pages:
            sentences = split_sentences(page.text)
            if not sentences:
                continue
            boundaries = self._boundaries(sentences)
            groups: list[list[str]] = []
            current: list[str] = []
            size = 0
            for i, sent in enumerate(sentences):
                n = count_tokens(sent)
                topic_shift = i in boundaries and size >= self.min_tokens
                too_big = current and size + n > self.max_tokens
                if topic_shift or too_big:
                    groups.append(current)
                    carry = current[-self.overlap :] if self.overlap and not topic_shift else []
                    current, size = list(carry), sum(count_tokens(s) for s in carry)
                current.append(sent)
                size += n
            groups.append(current)
            for group in groups:
                if group:
                    i = len(chunks)  # one sentence per line keeps list items and table rows apart
                    text = "\n".join(group)
                    chunks.append(
                        Chunk(f"{doc.id}:{i}", doc.id, doc.title, doc.source, page.number, page.heading, i, text)
                    )
        return chunks
