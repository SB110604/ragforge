"""Embedding back-ends. All return L2-normalised float32 matrices.

* :class:`HashingEmbedder` — offline default; lexical (word + character n-grams), no model download.
* :class:`FastEmbedEmbedder` — real semantic embeddings via ONNX (``pip install ragforge-ai[fastembed]``), no GPU/torch.
* :class:`OpenAICompatibleEmbedder` — any ``/embeddings`` endpoint (OpenAI, Ollama, vLLM, ...).
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim, self.name = dim, f"hashing-{dim}"
        self._words = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None, ngram_range=(1, 2), stop_words="english"
        )
        self._chars = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None, analyzer="char_wb", ngram_range=(3, 5)
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        w = self._words.transform(texts).toarray()
        c = self._chars.transform(texts).toarray()
        return _normalise(np.log1p(w) * 2.0 + np.log1p(c))


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("install the extra: pip install 'ragforge-ai[fastembed]'") from e
        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dim = int(self.embed(["dimension probe"]).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        return _normalise(np.vstack(list(self._model.embed(texts))))


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model or os.getenv("RAGFORGE_EMBED_MODEL", "")
        if not self.model:
            raise ValueError("no embedding model configured (RAGFORGE_EMBED_MODEL)")
        self.base_url = (base_url or os.getenv("RAGFORGE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("RAGFORGE_LLM_API_KEY", "")
        self._client, self.batch_size = client or httpx.Client(timeout=60), batch_size
        self.name = f"openai-compatible:{self.model}"
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for i in range(0, len(texts), self.batch_size):
            r = self._client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts[i : i + self.batch_size]},
            )
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        m = _normalise(np.array(out)) if out else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = m.shape[1] if len(m) else self.dim
        return m
