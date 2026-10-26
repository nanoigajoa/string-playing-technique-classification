"""Export bilingual editable drafts and an anonymous English review preview.

Requires local pandoc; PDF printing is a separate local browser command.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "연구과정"
OUT = ROOT / "results" / "phase1" / "canonical" / "manuscript"

CSS = """
@page { size: A4; margin: 22mm; }
body { font: 11pt/1.45 Arial, Helvetica, sans-serif; color: #111; max-width: 170mm; margin: auto; }
h1 { font-size: 19pt; line-height: 1.2; }
h2 { font-size: 14pt; } h3 { font-size: 12pt; }
h1, h2, h3 { break-after: avoid; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin-top: 1em; break-inside: avoid; break-after: avoid; }
th, td { padding: 5px; border-bottom: 0.5pt solid #aaa; }
thead { border-top: 1pt solid #333; border-bottom: 1pt solid #333; }
tr { break-inside: avoid; }
figure { margin: 1em 0; break-inside: avoid; }
figure img { max-width: 100%; max-height: 175mm; object-fit: contain; }
figcaption { font-size: 9pt; margin-top: 8px; }
.references p { font-size: 10pt; line-height: 1.3; margin: 0.7em 0; break-inside: avoid; }
code { font: 9pt monospace; overflow-wrap: anywhere; }
a { color: #111; text-decoration: none; overflow-wrap: anywhere; }
.page-break { break-before: page; }
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for language in ("국문", "영문"):
        path = DOCS / f"주법분류_논문초안_{language}_v1_canonical.md"
        subprocess.run(["pandoc", str(path), f"--resource-path={DOCS}", "-o", str(path.with_suffix(".docx"))], check=True)
    source = (DOCS / "주법분류_논문초안_영문_v1_canonical.md").read_text(encoding="utf-8")
    # Internal editing status is omitted; no author/affiliation fields are supplied.
    preview = re.sub(r"^> Internal manuscript draft.*\n", "", source, flags=re.MULTILINE)
    preview = preview.replace("**Keywords:**", '<div class="page-break"></div>\n\n**Keywords:**', 1)
    (OUT / "review_preview.md").write_text(preview, encoding="utf-8")
    (OUT / "review.css").write_text(CSS, encoding="utf-8")
    subprocess.run(["pandoc", str(OUT / "review_preview.md"), f"--resource-path={DOCS}",
                    "--standalone", "--embed-resources", f"--css={OUT / 'review.css'}",
                    "--metadata", "pagetitle=String Playing-Technique Evaluation Audit",
                    "--metadata", "lang=en", "-o", str(OUT / "review_preview.html")], check=True)
    html_path = OUT / "review_preview.html"
    html = html_path.read_text(encoding="utf-8")
    # Put each numbered caption inside its figure so printing keeps them together.
    html = re.sub(r"<figcaption[^>]*>.*?</figcaption>", "", html, flags=re.DOTALL)
    html, captions = re.subn(r"</figure>\s*<p>(Figure [123]\..*?)</p>",
                            r"<figcaption>\1</figcaption></figure>", html, flags=re.DOTALL)
    assert captions == 3, captions
    html = html.replace('<h2 id="references">References</h2>', '<div class="references"><h2 id="references">References</h2>', 1)
    html = html.replace("</body>", "</div>\n</body>", 1)
    html_path.write_text(html, encoding="utf-8")
    abstract = source.split("## Abstract\n\n", 1)[1].split("**Keywords:**", 1)[0]
    count = len(abstract.split())
    assert count <= 250, count
    assert len(source.split()) <= 8000
    print(f"Exported two DOCX drafts and anonymous HTML preview; abstract {count} words; manuscript {len(source.split())} whitespace words.")


if __name__ == "__main__":
    main()
