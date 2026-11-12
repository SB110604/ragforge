"""Command line: ``ragforge ingest|ask|search|eval|serve``. Configuration comes from environment variables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder, OpenAICompatibleEmbedder
from .evaluate import evaluate, load_dataset, save_report, to_markdown
from .llm import LLM, OpenAICompatibleLLM
from .pipeline import RAGPipeline
from .retrieval import CrossEncoderReranker, OverlapReranker, Reranker


def build_pipeline(index: str) -> RAGPipeline:
    kind = os.getenv("RAGFORGE_EMBEDDER", "hashing").lower()
    embedder: Embedder = (
        FastEmbedEmbedder(os.getenv("RAGFORGE_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        if kind == "fastembed"
        else OpenAICompatibleEmbedder()
        if kind == "openai"
        else HashingEmbedder()
    )
    llm: LLM | None = OpenAICompatibleLLM() if os.getenv("RAGFORGE_LLM_MODEL") else None
    reranker: Reranker = (
        CrossEncoderReranker() if os.getenv("RAGFORGE_RERANKER") == "cross-encoder" else OverlapReranker()
    )
    glossary_path = os.getenv("RAGFORGE_GLOSSARY")
    glossary = json.loads(Path(glossary_path).read_text(encoding="utf-8")) if glossary_path else None
    return RAGPipeline(
        index,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        glossary=glossary,
        llm_query_rewrites=os.getenv("RAGFORGE_LLM_REWRITES", "").lower() in {"1", "true", "yes"},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragforge", description="Hybrid-search RAG with citations.")
    p.add_argument("--index", default=os.getenv("RAGFORGE_INDEX", "data/index.db"), help="index file (SQLite)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="index files or folders")
    s.add_argument("paths", nargs="+")
    for name in ("ask", "search"):
        s = sub.add_parser(name)
        s.add_argument("question")
        s.add_argument("-k", type=int, default=4)
        s.add_argument("--json", action="store_true")
    s = sub.add_parser("eval", help="evaluate against a JSONL dataset")
    s.add_argument("dataset")
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--out", help="write the JSON report here")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles: never crash on characters such as arrows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    rag = build_pipeline(a.index)
    if a.cmd == "ingest":
        for path in a.paths:
            results = rag.ingest_dir(path) if Path(path).is_dir() else [rag.ingest(path)]
            for r in results:
                print(f"indexed {r['source']}: {r['pages']} page(s), {r['chunks']} chunks")
        print(f"index: {len(rag.store)} chunks in {len(rag.store.documents())} documents -> {a.index}")
    elif a.cmd == "ask":
        ans = rag.ask(a.question, a.k)
        if a.json:
            print(json.dumps(rag.answer_dict(ans), indent=2, ensure_ascii=False))
        else:
            print(ans.answer)
            for c in ans.citations:
                where = f"p. {c.page}" + (f", {c.heading}" if c.heading else "")
                print(f"  [{c.n}] {c.doc_title} ({c.source}, {where})")
    elif a.cmd == "search":
        hits, queries = rag.search(a.question, a.k)
        print("queries:", " | ".join(queries))
        for i, h in enumerate(hits, 1):
            print(
                f"{i}. {h.chunk.doc_title} p.{h.chunk.page} score={h.rerank_score} "
                f"(vector #{h.vector_rank}, keyword #{h.keyword_rank})\n   {h.chunk.text[:160]}"
            )
    elif a.cmd == "eval":
        report = evaluate(rag, load_dataset(a.dataset), a.k)
        print(to_markdown(report))
        if a.out:
            save_report(report, a.out)
    elif a.cmd == "serve":  # pragma: no cover - starts a server
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(rag, api_key=os.getenv("RAGFORGE_API_KEY") or None), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
