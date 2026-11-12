# HTTP API

Start with `ragforge serve` (uses the index from `--index` / `RAGFORGE_INDEX`) or
`uvicorn --factory ragforge.api:app_from_env`. Interactive docs: `/docs`. When `RAGFORGE_API_KEY` is set, every
endpoint except `/health` needs the header `X-API-Key`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | document and chunk counts, embedder, generator |
| GET | `/documents` | — | `[{id, title, source, chunks, pages}]` |
| POST | `/documents` | multipart `file` (.pdf .md .txt .html, max 25 MB) | `{id, title, source, pages, chunks}` · 415 wrong type · 413 too big · 422 unreadable |
| DELETE | `/documents/{id}` | — | `{deleted_chunks}` · 404 |
| POST | `/search` | `{"question": "...", "k": 4, "doc_ids": null}` | `{queries, hits: [{chunk, score, rerank_score, vector_rank, keyword_rank}]}` |
| POST | `/ask` | same as `/search` | `{answer, citations: [{n, source, page, heading, snippet}], found, mode, confidence, queries, timings_ms}` |

Re-uploading a file with the same name and content replaces its chunks instead of duplicating them.

## Example

```bash
curl -F "file=@contract.pdf" localhost:8000/documents
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is the notice period?", "k": 4}'
```

```json
{
  "answer": "Either party may terminate with thirty days written notice. [1]",
  "citations": [{"n": 1, "source": "contract.pdf", "page": 3, "heading": null, "snippet": "..."}],
  "found": true,
  "mode": "extractive",
  "queries": ["What is the notice period?", "notice period"]
}
```

The response above is illustrative (contract.pdf is not bundled). `found` is `false` and `answer` is
"I could not find this in the indexed documents." when nothing relevant is retrieved.
