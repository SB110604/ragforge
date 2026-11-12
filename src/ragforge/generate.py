"""Answer generation with numbered citations: LLM-based when configured, extractive otherwise."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM, LLMError
from .retrieval import Hit
from .text import tokenize

NOT_FOUND = "I could not find this in the indexed documents."


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_title: str
    source: str
    page: int
    heading: str | None
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "extractive"
    confidence: float = 0.0
    queries: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.answer != NOT_FOUND


def _citation(n: int, h: Hit) -> Citation:
    c = h.chunk
    snippet = c.text if len(c.text) <= 300 else c.text[:297].rsplit(" ", 1)[0] + "..."
    return Citation(n, c.id, c.doc_title, c.source, c.page, c.heading, snippet)


class ExtractiveGenerator:
    """Selects the sentences that best cover the question and cites the chunk each came from. No LLM required."""

    def __init__(self, max_sentences: int = 4, min_coverage: float = 0.34) -> None:
        self.max_sentences, self.min_coverage = max_sentences, min_coverage

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        q = set(tokenize(question))
        scored: list[tuple[float, int, int, str]] = []
        for rank, h in enumerate(hits):
            for pos, sent in enumerate(line for line in h.chunk.text.splitlines() if line.strip()):
                toks = set(tokenize(sent))
                if not toks or not q:
                    continue
                coverage = len(q & toks) / len(q)
                scored.append((coverage - 0.03 * rank, rank, pos, sent))
        scored.sort(key=lambda s: -s[0])
        best = scored[0][0] + 0.03 * scored[0][1] if scored else 0.0
        if best < self.min_coverage:
            return Answer(question, NOT_FOUND, confidence=round(best, 3))
        chosen: list[tuple[int, int, str]] = []
        for score, rank, pos, sent in scored:
            if len(chosen) >= self.max_sentences or score < max(self.min_coverage, best * 0.75):
                break
            if any(len(set(tokenize(sent)) ^ set(tokenize(c[2]))) < 3 for c in chosen):
                continue  # near-duplicate (e.g. from chunk overlap)
            chosen.append((rank, pos, sent))  # kept in score order: the best-supported sentence comes first
        numbers: dict[int, int] = {}
        parts = []
        for rank, _, sent in chosen:
            n = numbers.setdefault(rank, len(numbers) + 1)
            parts.append(f"{sent} [{n}]")
        cites = [_citation(n, hits[rank]) for rank, n in numbers.items()]
        return Answer(question, " ".join(parts), cites, "extractive", round(min(1.0, best), 3))


_PROMPT = """Answer the question using ONLY the numbered sources below.
Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
If the sources do not contain the answer, reply exactly: {not_found}

Sources:
{sources}

Question: {question}"""


class LLMGenerator:
    def __init__(self, llm: LLM, fallback: ExtractiveGenerator | None = None) -> None:
        self.llm, self.fallback = llm, fallback or ExtractiveGenerator()

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(question, NOT_FOUND, mode="llm")
        sources = "\n\n".join(
            f"[{i}] ({h.chunk.doc_title}, p. {h.chunk.page}) {h.chunk.text}" for i, h in enumerate(hits, 1)
        )
        try:
            text = self.llm.complete(
                [{"role": "user", "content": _PROMPT.format(not_found=NOT_FOUND, sources=sources, question=question)}]
            ).strip()
        except LLMError:
            ans = self.fallback.generate(question, hits)
            ans.mode = "extractive (LLM unavailable)"
            return ans
        if NOT_FOUND.rstrip(".") in text:
            return Answer(question, NOT_FOUND, mode="llm")
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
        cites = [_citation(n, hits[n - 1]) for n in used]
        return Answer(question, text, cites, "llm", 1.0 if cites else 0.5)
