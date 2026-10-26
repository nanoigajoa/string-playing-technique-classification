"""Export the supervisor review summary without changing manuscript artifacts."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle

ROOT = Path.cwd()
SOURCE = ROOT / "paper/supervisor_review/투고검토요약_20261007.md"
fontManager.addfont("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
FONT = FontProperties(family=["AppleGothic", "DejaVu Sans"])
WIDTH, HEIGHT = 595.28, 841.89
MARGIN = 43
INK, BLUE = "#202b36", "#234761"


def clean(text):
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text).replace("**", "")


def draw_page(chunk, page_number, output):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y = HEIGHT - MARGIN
    text_width = WIDTH - 2 * MARGIN
    width_cache = {}

    def char_width(char, size):
        key = (char, size)
        if key not in width_cache:
            font = FONT.copy()
            font.set_size(size)
            width_cache[key] = renderer.get_text_width_height_descent(char, font, False)[0] * 72 / fig.dpi
        return width_cache[key]

    def wrap(text, size, width):
        lines, line, occupied = [], "", 0
        for char in text:
            char_size = char_width(char, size)
            if line and occupied + char_size > width:
                # Prefer a space boundary while allowing Korean line breaks.
                boundary = line.rfind(" ")
                if boundary > len(line) * 0.65:
                    lines.append(line[:boundary].rstrip())
                    line = line[boundary + 1:]
                    occupied = sum(char_width(c, size) for c in line)
                else:
                    lines.append(line.rstrip())
                    line, occupied = "", 0
            line += char
            occupied += char_size
        if line.strip():
            lines.append(line.rstrip())
        return lines

    def paragraph(text, size=9.7, line_height=14.2, after=7, color=INK):
        nonlocal y
        matches = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
        url = matches[0][1] if matches else None
        for line in wrap(clean(text), size, text_width):
            ax.text(MARGIN, y, line, fontproperties=FONT, fontsize=size,
                    color=color, va="top", url=url)
            y -= line_height
        y -= after

    lines = chunk.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            paragraph(line[2:], size=16, line_height=23, after=13, color=BLUE)
        elif line.startswith("## "):
            y -= 3
            paragraph(line[3:], size=11.3, line_height=17, after=5, color=BLUE)
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [clean(cell.strip()) for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-:]+", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            columns = len(rows[0])
            widths = [text_width * value for value in ([0.32, 0.34, 0.34] if columns == 3 else [0.29, 0.71])]
            size, row_leading = 9.0, 13
            for row_number, row in enumerate(rows):
                wrapped = [wrap(cell, size, width - 14) for cell, width in zip(row, widths)]
                row_height = max(len(cell) for cell in wrapped) * row_leading + 12
                ax.add_patch(Rectangle((MARGIN, y - row_height), text_width, row_height,
                                      facecolor="#e9f0f5" if row_number == 0 else "white",
                                      edgecolor="#c7d0d7", linewidth=0.45))
                x = MARGIN
                for cell_lines, width in zip(wrapped, widths):
                    for n, cell_line in enumerate(cell_lines):
                        ax.text(x + 7, y - 6 - n * row_leading, cell_line,
                                fontproperties=FONT, fontsize=size, color=INK, va="top")
                    x += width
                y -= row_height
            y -= 10
            continue
        elif re.match(r"\[[1-3]\]", line):
            paragraph(line, size=8.0, line_height=11.3, after=5)
        elif line.startswith("2026-"):
            paragraph(line, size=8.5, line_height=13, color="#586773")
        else:
            paragraph(line)
        i += 1
    assert y > MARGIN + 10, f"Page {page_number} overflow: final y={y}"
    ax.plot([MARGIN, WIDTH - MARGIN], [30, 30], color="#c7d0d7", linewidth=0.5)
    ax.text(MARGIN, 21, "현악 주법 분류 · 투고 방향 검토 · 2026-10-07",
            fontproperties=FONT, fontsize=7.7, color="#586773", va="top")
    ax.text(WIDTH - MARGIN, 21, f"{page_number} / 2", ha="right",
            fontproperties=FONT, fontsize=7.7, color="#586773", va="top")
    output.savefig(fig)
    plt.close(fig)


def export_docx(text):
    page_break = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
    temporary = Path("/private/tmp/phase1-supervisor-brief-docx.md")
    temporary.write_text(text.replace("<!-- PAGE_BREAK -->", page_break))
    target = SOURCE.with_suffix(".docx")
    subprocess.run(["pandoc", str(temporary), "--metadata", "lang=ko-KR", "-o", str(target)], check=True)
    with zipfile.ZipFile(target) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns = {"w": w}
    styles = ET.fromstring(files["word/styles.xml"])
    for fonts in styles.findall(".//w:rFonts", ns):
        for attribute in list(fonts.attrib):
            if "theme" in attribute.lower():
                del fonts.attrib[attribute]
        for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(f"{{{w}}}{attribute}", "AppleGothic")
    document = ET.fromstring(files["word/document.xml"])
    for section in document.findall(".//w:sectPr", ns):
        size = section.find("w:pgSz", ns)
        if size is not None:
            size.set(f"{{{w}}}w", "11906")
            size.set(f"{{{w}}}h", "16838")
    files["word/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)
    files["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    return target


text = SOURCE.read_text()
chunks = text.split("<!-- PAGE_BREAK -->")
assert len(chunks) == 2
pdf = SOURCE.with_suffix(".pdf")
with PdfPages(pdf, metadata={"Title": "현악 주법 분류 연구 — 원고 및 투고 검토 요약"}) as output:
    for number, chunk in enumerate(chunks, 1):
        draw_page(chunk, number, output)
docx = export_docx(text)
manifest = {"source": str(SOURCE.relative_to(ROOT)), "pdf_pages": 2,
            "docx_pages": "not checked in Word", "files": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [SOURCE, pdf, docx]}}
(SOURCE.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"pdf": str(pdf.relative_to(ROOT)), "docx": str(docx.relative_to(ROOT)), "pages": 2}, ensure_ascii=False))
