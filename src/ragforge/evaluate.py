"""Evaluation of retrieval and answers against a JSONL dataset.

Each line: ``{"question": "...", "evidence": ["text that a relevant chunk contains", ...], "source": "file.md"}``.
Use ``"answerable": false`` for questions the documents cannot answer (the system should abstain).

Metrics: hit rate@k and MRR (retrieval), answer accuracy (answered when answerable, abstained otherwise),
citation precision (cited chunks that contain the evidence) and groundedness (answer sentences whose words are
supported by the cited text).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .pipeline import RAGPipeline
from .text import split_sentences, tokenize


@dataclass
class EvalCase:
    question: str
    evidence: list[str]
    source: str | None = None
    answerable: bool = True


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            ev = d.get("evidence", [])
            cases.append(
                EvalCase(
                    d["question"],
                    [ev] if isinstance(ev, str) else list(ev),
                    d.get("source"),
                    bool(d.get("answerable", True)),
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"{path}:{i}: invalid case ({e})") from e
    return cases


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_relevant(chunk: Chunk, case: EvalCase) -> bool:
    if case.source and chunk.source != case.source:
        return False
    text = _norm(chunk.text)
    return any(_norm(e) in text for e in case.evidence)


def groundedness(answer: str, cited_texts: list[str], threshold: float = 0.6) -> float:
    sentences = [re.sub(r"\[\d+\]", "", s).strip() for s in split_sentences(answer)]
    sentences = [s for s in sentences if tokenize(s)]
    if not sentences:
        return 0.0
    support = set(tokenize(" ".join(cited_texts)))
    ok = sum(1 for s in sentences if len(set(tokenize(s)) & support) / len(set(tokenize(s))) >= threshold)
    return ok / len(sentences)


def evaluate(rag: RAGPipeline, cases: list[EvalCase], k: int = 4) -> dict[str, Any]:
    rows = []
    for case in cases:
        hits, _ = rag.search(case.question, k)
        ranks = [i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, case)]
        ans = rag.ask(case.question, k)
        by_id = {h.chunk.id: h.chunk for h in hits}
        cited = [by_id.get(c.chunk_id) or rag.store.get(c.chunk_id) for c in ans.citations]
        cited = [c for c in cited if c is not None]
        row: dict[str, Any] = {
            "question": case.question,
            "answerable": case.answerable,
            "found": ans.found,
            "answer_correct": ans.found == case.answerable,
        }
        if case.answerable:
            row |= {
                "hit": bool(ranks),
                "rr": 1 / ranks[0] if ranks else 0.0,
                "citation_precision": (sum(is_relevant(c, case) for c in cited) / len(cited)) if cited else 0.0,
                "groundedness": groundedness(ans.answer, [c.text for c in cited]) if ans.found else 0.0,
            }
        rows.append(row)
    ans_rows = [r for r in rows if r["answerable"]]

    def mean(key: str, rs: list[dict[str, Any]]) -> float:
        return round(sum(float(r[key]) for r in rs) / len(rs), 3) if rs else 0.0

    summary = {
        "cases": len(rows),
        "k": k,
        f"hit_rate@{k}": mean("hit", ans_rows),
        "mrr": mean("rr", ans_rows),
        "answer_accuracy": mean("answer_correct", rows),
        "citation_precision": mean("citation_precision", ans_rows),
        "groundedness": mean("groundedness", ans_rows),
    }
    return {"summary": summary, "rows": rows}


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s.items()]
    lines += ["", "| Question | Hit | RR | Answered | Correct |", "|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(
            f"| {r['question']} | {r.get('hit', '-')} | {round(r.get('rr', 0), 2) if r['answerable'] else '-'} | "
            f"{r['found']} | {r['answer_correct']} |"
        )
    return "\n".join(lines)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = ["EvalCase", "load_dataset", "evaluate", "groundedness", "is_relevant", "to_markdown", "save_report"]
