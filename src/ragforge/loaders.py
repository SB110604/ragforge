"""Load PDF, Markdown, text and HTML files into page-aware documents."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Page:
    number: int  # 1-based; plain-text formats are a single page
    text: str
    heading: str | None = None


@dataclass
class Document:
    id: str
    title: str
    source: str
    pages: list[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class LoaderError(Exception):
    pass


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section"}:
            self.parts.append("\n\n" if tag in {"p", "h1", "h2", "h3", "h4", "section"} else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _doc_id(source: str, content: bytes) -> str:
    return hashlib.sha1(source.encode() + b"\0" + content).hexdigest()[:12]


_LABEL = re.compile(r"\[\(?\"?([^\]\[]+?)\"?\)?\]|\{\{([^}]+)\}\}|\(\(([^)]+)\)\)|\|([^|]+)\|")


def _mermaid_to_text(src: str) -> str:
    """Turn a Mermaid diagram into sentences: node/edge labels for flowcharts, message text for sequence diagrams."""
    out: list[str] = []
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith(("%%", "classDef", "class ", "style ", "end", "direction")):
            continue
        if re.match(r"^\S+\s*-[->x)]+\+?\s*\S+\s*:", line):  # sequence message  A->>B: text
            out.append(line.split(":", 1)[1].strip())
            continue
        if line.startswith(("participant", "actor")):
            continue
        for m in _LABEL.finditer(line):
            label = next(g for g in m.groups() if g)
            out.append(re.sub(r"<br\s*/?>", ", ", label).strip())
    return "\n".join(f"- {s.rstrip('.')}." for s in out if s and not s.startswith(("subgraph", "flowchart")))


def _clean_markdown(text: str) -> str:
    """Convert Mermaid blocks to text, drop fence markers and inline formatting (links, emphasis, code ticks)."""
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: _mermaid_to_text(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"^```\w*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?!\w)", r"\1", text)
    return text.replace("`", "")


def _markdown_pages(text: str) -> list[Page]:
    """Keep the whole file as one page but remember the nearest heading for each section."""
    sections, heading, buf = [], None, []
    for line in _clean_markdown(text).splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return [Page(1, body, h) for h, body in sections] or [Page(1, text.strip())]


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")


def load_document(path: str | Path) -> Document:
    p = Path(path)
    if not p.is_file():
        raise LoaderError(f"file not found: {p}")
    raw = p.read_bytes()
    suffix = p.suffix.lower()
    title = p.stem.replace("_", " ").replace("-", " ")
    if suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(str(p))
            pages = [Page(i + 1, (pg.extract_text() or "").strip()) for i, pg in enumerate(reader.pages)]
            meta_title = (reader.metadata.title if reader.metadata else None) or ""
        except PdfReadError as e:
            raise LoaderError(f"cannot read PDF {p.name}: {e}") from e
        pages = [pg for pg in pages if pg.text]
        if not pages:
            raise LoaderError(f"{p.name} has no extractable text (scanned PDF? run OCR first)")
        title = meta_title.strip() or title
    elif suffix in {".md", ".markdown"}:
        text = _decode(raw)
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = m.group(1).strip() if m else title
        pages = _markdown_pages(text)
    elif suffix in {".txt", ".rst"}:
        pages = [Page(1, _decode(raw).strip())]
    elif suffix in {".html", ".htm"}:
        parser = _HTMLText()
        parser.feed(_decode(raw))
        text = html.unescape(re.sub(r"\n{3,}", "\n\n", "".join(parser.parts))).strip()
        title = parser.title.strip() or title
        pages = [Page(1, text)]
    else:
        raise LoaderError(f"unsupported file type: {suffix or '(none)'} — use .pdf, .md, .txt or .html")
    if not any(pg.text for pg in pages):
        raise LoaderError(f"{p.name} is empty")
    return Document(_doc_id(p.name, raw), title, p.name, pages)
