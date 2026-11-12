import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ragforge import (
    BM25,
    NOT_FOUND,
    Document,
    HashingEmbedder,
    LoaderError,
    Page,
    QueryRewriter,
    RAGPipeline,
    SemanticChunker,
    evaluate,
    load_dataset,
    load_document,
)
from ragforge.api import create_app
from ragforge.cli import main
from ragforge.evaluate import groundedness
from ragforge.generate import LLMGenerator
from ragforge.llm import LLMError
from ragforge.text import split_sentences, tokenize

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

COFFEE = """# Coffee guide

## Brewing
Use water at 92 to 96 degrees Celsius for pour-over coffee. Grind the beans medium-fine, like table salt.
A ratio of one gram of coffee to sixteen grams of water gives a balanced cup.

## Storage
Store beans in an airtight container away from light. Whole beans stay fresh for about four weeks after roasting.
Do not keep coffee in the fridge because it absorbs odours.
"""

TEA = """# Tea guide

Green tea should be brewed at 75 to 80 degrees Celsius for two minutes. Black tea needs boiling water and four minutes.
"""


def make_pdf(path: Path, pages: list[str]) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "coffee.md").write_text(COFFEE, encoding="utf-8")
    (tmp_path / "tea.txt").write_text(TEA, encoding="utf-8")
    make_pdf(tmp_path / "warranty.pdf", ["Warranty terms.", "The grinder has a two year warranty covering the motor."])
    return tmp_path


@pytest.fixture
def rag(corpus):
    r = RAGPipeline()
    r.ingest_dir(corpus)
    return r


# ---- text and loading ------------------------------------------------------------------------------------------------
def test_tokenize_and_sentences():
    assert tokenize("The orders were ordered") == ["order", "order"]
    assert split_sentences("One. Two!\n\n- item a\n- item b\n| x | y |\n|---|---|\n| 1 | 2 |") == [
        "One.",
        "Two!",
        "item a",
        "item b",
        "x — y",
        "1 — 2",
    ]
    assert all(len(s.split()) <= 60 for s in split_sentences("word " * 200))


def test_load_markdown_pdf_html_and_errors(corpus, tmp_path):
    md = load_document(corpus / "coffee.md")
    assert md.title == "Coffee guide" and [p.heading for p in md.pages] == ["Brewing", "Storage"]
    pdf = load_document(corpus / "warranty.pdf")
    assert [p.number for p in pdf.pages] == [1, 2] and "two year warranty" in pdf.pages[1].text
    (tmp_path / "x.html").write_text(
        "<html><head><title>Hi</title><script>evil()</script></head><body><p>Hello &amp; welcome</p></body></html>",
        encoding="utf-8",
    )
    h = load_document(tmp_path / "x.html")
    assert h.title == "Hi" and "Hello & welcome" in h.text and "evil" not in h.text
    (tmp_path / "x.docx").write_bytes(b"x")
    with pytest.raises(LoaderError, match="unsupported"):
        load_document(tmp_path / "x.docx")
    with pytest.raises(LoaderError, match="not found"):
        load_document(tmp_path / "missing.md")


def test_mermaid_blocks_become_text(tmp_path):
    (tmp_path / "d.md").write_text(
        "# D\n\n```mermaid\nflowchart LR\n  A[Stock<br/>monitor] --> B{{Decide}}\n  X->>Y: send PO e-mail\n```\n",
        encoding="utf-8",
    )
    text = load_document(tmp_path / "d.md").text
    assert "Stock, monitor." in text and "Decide." in text and "send PO e-mail." in text and "-->" not in text


# ---- chunking, embeddings, BM25 --------------------------------------------------------------------------------------
def test_semantic_chunker_respects_budget_and_keeps_metadata():
    body = " ".join(f"Sentence number {i} talks about coffee brewing temperature." for i in range(40))
    doc = Document("d1", "Doc", "doc.md", [Page(3, body, "Brewing")])
    chunks = SemanticChunker(HashingEmbedder(), max_tokens=60, min_tokens=20).chunk(doc)
    assert len(chunks) > 3 and all(len(c.text.split()) <= 60 for c in chunks)
    assert {c.page for c in chunks} == {3} and chunks[0].heading == "Brewing" and chunks[1].id == "d1:1"
    with pytest.raises(ValueError):
        SemanticChunker(HashingEmbedder(), max_tokens=10, min_tokens=20)


def test_hashing_embedder_is_normalised_and_similarity_makes_sense():
    e = HashingEmbedder()
    m = e.embed(["brew coffee at 94 degrees", "coffee brewing temperature", "warranty for the grinder motor"])
    assert m.shape == (3, 1024) and np.allclose(np.linalg.norm(m, axis=1), 1, atol=1e-5)
    assert m[0] @ m[1] > m[0] @ m[2]


def test_bm25_ranks_matching_text_first():
    b = BM25()
    b.build(["coffee grind size", "tea brewing time", "coffee coffee storage"])
    s = b.scores("coffee storage")
    assert int(np.argmax(s)) == 2 and s[1] == 0


# ---- retrieval, rewriting, answers -----------------------------------------------------------------------------------
def test_query_rewriter_rules_and_glossary():
    qs = QueryRewriter({"PO": "purchase order"}).rewrite("What is the difference between a PO and an invoice?")
    assert qs[0].startswith("What is") and "difference PO invoice" in qs
    assert any("purchase order" in q for q in qs) and "a PO" in qs and "an invoice" in qs


def test_query_rewriter_uses_llm_and_survives_failure():
    class Good:
        def complete(self, m):
            return 'Sure: ["coffee water temperature", "pour over heat"]'

    class Bad:
        def complete(self, m):
            raise LLMError("down")

    assert "pour over heat" in QueryRewriter(llm=Good()).rewrite("how hot should water be?")
    assert QueryRewriter(llm=Bad()).rewrite("how hot?") == ["how hot?", "hot"]


def test_hybrid_search_finds_pdf_and_markdown(rag):
    hits, queries = rag.search("How long is the grinder warranty?", k=3)
    assert hits[0].chunk.source == "warranty.pdf" and hits[0].chunk.page == 2
    assert hits[0].keyword_rank == 1 and hits[0].rerank_score is not None and len(queries) >= 2
    tea_only = rag.search(
        "water temperature", k=3, doc_ids=[d["id"] for d in rag.store.documents() if d["source"] == "tea.txt"]
    )
    assert {h.chunk.source for h in tea_only[0]} == {"tea.txt"}


def test_extractive_answer_cites_sources_and_abstains(rag):
    a = rag.ask("What water temperature should I use for pour-over coffee?")
    assert a.found and "92 to 96 degrees" in a.answer and "[1]" in a.answer
    assert a.citations[0].source == "coffee.md" and a.citations[0].heading == "Brewing"
    assert a.timings_ms["retrieve"] >= 0 and a.queries
    assert rag.ask("Who won the cricket world cup in 2011?").answer == NOT_FOUND


def test_llm_generator_parses_citations_and_falls_back(rag):
    hits, _ = rag.search("how long do whole beans stay fresh", k=3)

    class LLM:
        def complete(self, m):
            assert "[1]" in m[0]["content"]
            return "About four weeks after roasting [1]."

    class Down:
        def complete(self, m):
            raise LLMError("timeout")

    a = LLMGenerator(LLM()).generate("how long do whole beans stay fresh", hits)
    assert a.mode == "llm" and [c.n for c in a.citations] == [1]
    b = LLMGenerator(Down()).generate("how long do whole beans stay fresh", hits)
    assert b.mode.startswith("extractive") and "four weeks" in b.answer


# ---- persistence and evaluation --------------------------------------------------------------------------------------
def test_index_persists_reingest_replaces_and_embedder_must_match(corpus, tmp_path):
    db = tmp_path / "idx" / "index.db"
    r = RAGPipeline(db)
    first = r.ingest(corpus / "coffee.md")
    n = len(r.store)
    r.ingest(corpus / "coffee.md")
    assert len(r.store) == n
    r2 = RAGPipeline(db)
    assert len(r2.store) == n and r2.store.documents()[0]["id"] == first["id"]
    assert r2.store.delete_document(first["id"]) == n and len(r2.store) == 0
    with pytest.raises(ValueError, match="built with"):
        RAGPipeline(db, embedder=HashingEmbedder(dim=256))


def test_groundedness():
    assert groundedness("Beans stay fresh four weeks [1].", ["Whole beans stay fresh for about four weeks"]) == 1.0
    assert groundedness("The moon is made of cheese [1].", ["Whole beans stay fresh"]) == 0.0


def test_evaluation_on_bundled_example_corpus():
    rag = RAGPipeline()
    rag.ingest_dir(EXAMPLES / "docs")
    report = evaluate(rag, load_dataset(EXAMPLES / "eval.jsonl"), k=4)
    s = report["summary"]
    assert s["cases"] == 12 and s["hit_rate@4"] >= 0.9 and s["answer_accuracy"] == 1.0 and s["groundedness"] >= 0.9


def test_bad_dataset_line(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"evidence": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid case"):
        load_dataset(tmp_path / "d.jsonl")


# ---- API and CLI -----------------------------------------------------------------------------------------------------
def test_api_upload_search_ask_delete(corpus):
    c = TestClient(create_app(RAGPipeline(), api_key="k"))
    h = {"X-API-Key": "k"}
    assert c.get("/documents").status_code == 401
    with open(corpus / "coffee.md", "rb") as f:
        doc = c.post("/documents", files={"file": ("coffee.md", f, "text/markdown")}, headers=h).json()
    assert doc["chunks"] >= 1
    assert (
        c.post("/documents", files={"file": ("a.exe", b"MZ", "application/octet-stream")}, headers=h).status_code == 415
    )
    assert c.post("/documents", files={"file": ("e.md", b"   ", "text/markdown")}, headers=h).status_code == 422
    ans = c.post("/ask", json={"question": "Should I keep coffee in the fridge?"}, headers=h).json()
    assert ans["found"] and ans["citations"][0]["source"] == "coffee.md"
    hits = c.post("/search", json={"question": "airtight container", "k": 2}, headers=h).json()["hits"]
    assert hits and hits[0]["chunk"]["heading"] == "Storage"
    assert c.get("/health").json()["chunks"] >= 1
    assert c.delete(f"/documents/{doc['id']}", headers=h).json()["deleted_chunks"] >= 1
    assert c.delete(f"/documents/{doc['id']}", headers=h).status_code == 404


def test_cli_ingest_ask_eval(corpus, tmp_path, capsys):
    idx = str(tmp_path / "cli.db")
    assert main(["--index", idx, "ingest", str(corpus)]) == 0
    assert "3 documents" in capsys.readouterr().out
    main(["--index", idx, "ask", "How long is the grinder warranty?", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["found"] and out["citations"][0]["source"] == "warranty.pdf"
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        json.dumps({"question": "grinder warranty", "evidence": "two year warranty"}) + "\n", encoding="utf-8"
    )
    main(["--index", idx, "eval", str(ds), "--out", str(tmp_path / "r.json")])
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["hit_rate@4"] == 1.0
