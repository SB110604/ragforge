"""Render a Markdown file to a simple multi-page PDF (used to build the sample corpus).

python examples/make_sample_pdf.py SOURCE.md examples/docs/setup-guide.pdf
"""

import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def main(src: str, out: str) -> None:
    styles = getSampleStyleSheet()
    story = []
    for block in re.split(r"\n\s*\n", open(src, encoding="utf-8").read()):
        block = block.strip()
        if not block or block.startswith("---"):
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", block)
        if m:
            story.append(Paragraph(escape(re.sub(r"[`*]", "", m.group(2))), styles[f"Heading{len(m.group(1))}"]))
            continue
        text = re.sub(r"[`*]", "", block).replace("\n", " ")
        story.append(Paragraph(escape(text), styles["BodyText"]))
        story.append(Spacer(1, 4))
    SimpleDocTemplate(out, pagesize=A4, title="SmartStock RPA Setup and Run Guide").build(story)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
