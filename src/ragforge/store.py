"""Persistent index: chunk metadata + embeddings in SQLite, exact cosine search in NumPy, and a BM25 keyword index."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .text import tokenize


class BM25:
    """Okapi BM25 over tokenised chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()

    def build(self, texts: list[str]) -> None:
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.df = Counter(w for d in self.docs for w in d)

    def scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        out = np.zeros(n, dtype=np.float32)
        if not n:
            return out
        avg = sum(self.lengths) / n or 1.0
        for w in set(tokenize(query)):
            df = self.df.get(w, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(self.docs):
                tf = d.get(w, 0)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / avg))
        return out


class VectorStore:
    def __init__(self, path: str | Path = ":memory:", embedder_name: str = "") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, doc_id TEXT, doc_title TEXT, source TEXT, "
                "page INTEGER, heading TEXT, idx INTEGER, text TEXT, embedding BLOB)"
            )
        stored = self._db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if stored and embedder_name and stored[0] != embedder_name:
            raise ValueError(
                f"index was built with {stored[0]!r}, not {embedder_name!r}; rebuild it or use the same embedder"
            )
        if embedder_name and not stored:
            with self._db:
                self._db.execute("INSERT INTO meta VALUES ('embedder', ?)", (embedder_name,))
        self._reload()

    def _reload(self) -> None:
        rows = self._db.execute(
            "SELECT id, doc_id, doc_title, source, page, heading, idx, text, embedding FROM chunks ORDER BY doc_id, idx"
        ).fetchall()
        self.chunks = [Chunk(*r[:8]) for r in rows]
        self.matrix = np.vstack([np.frombuffer(r[8], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 0))
        self.bm25 = BM25()
        self.bm25.build([c.context_text for c in self.chunks])

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        with self._lock, self._db:
            for doc_id in {c.doc_id for c in chunks}:
                self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.doc_id,
                        c.doc_title,
                        c.source,
                        c.page,
                        c.heading,
                        c.index,
                        c.text,
                        np.asarray(e, dtype=np.float32).tobytes(),
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
            self._reload()

    def delete_document(self, doc_id: str) -> int:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount
            self._reload()
        return n

    def documents(self) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT doc_id, doc_title, source, COUNT(*), MAX(page) FROM chunks GROUP BY doc_id ORDER BY doc_title"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "source": r[2], "chunks": r[3], "pages": r[4]} for r in rows]

    def vector_scores(self, query_vec: np.ndarray) -> np.ndarray:
        if not len(self.chunks):
            return np.zeros(0, dtype=np.float32)
        return self.matrix @ np.asarray(query_vec, dtype=np.float32)

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def chunk_dict(c: Chunk) -> dict[str, object]:
        return asdict(c)
