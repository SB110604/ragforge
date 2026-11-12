"""Text utilities shared by chunking, BM25, reranking and evaluation."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her here hers him
    his how
    i if in into is it its itself just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your""".split()
)


def stem(word: str) -> str:
    """A light suffix stripper (enough to match 'orders'/'ordered'/'ordering')."""
    for suf in ("ing", "edly", "ed", "ies", "es", "s"):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    words = _WORD.findall(text.lower())
    return [stem(w) for w in words if keep_stopwords or w not in STOPWORDS]


_ITEM = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")


def split_sentences(text: str, max_words: int = 60) -> list[str]:
    """Split text into sentences, respecting its line structure.

    Blank lines end a paragraph, list items and table rows are their own units (wrapped continuation lines are
    joined back), paragraphs are split into sentences, and anything longer than ``max_words`` is cut.
    """
    units: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            units.extend(_SENT.split(" ".join(para)))
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) or not c for c in cells):
                units.append(" — ".join(c for c in cells if c))
        elif _ITEM.match(line):
            flush()
            para.append(_ITEM.sub("", line))
        else:
            para.append(line)
    flush()
    out: list[str] = []
    for u in units:
        words = " ".join(u.split()).split(" ")
        if words != [""]:
            out += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
    return out


def count_tokens(text: str) -> int:
    """Approximate token count (words); good enough for chunk sizing."""
    return len(text.split())
