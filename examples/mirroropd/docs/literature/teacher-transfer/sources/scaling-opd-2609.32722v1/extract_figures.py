# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render selected original PDF figures without modifying plot elements.

Requires pymupdf. Coordinates are PDF points, top-left origin, pages one-based.
The original PDF stays outside Git; the manifest binds all crops to its hash.
"""
import argparse
import hashlib
import json
from pathlib import Path
import fitz

EXPECTED_SHA = "1444fcc17452c608925c7429c4846dfaf8a6886db68541baa179ce0b0e11e53e"
CROPS = [
    (2, 4, [68, 60, 544, 216], "S4.F2"),
    (4, 5, [68, 60, 544, 213], "S4.F4"),
    (5, 8, [175, 60, 430, 236], "S5.F5"),
    (6, 8, [68, 295, 544, 420], "S5.F8"),
    (7, 8, [330, 515, 544, 689], "S6.F9"),
    (8, 9, [319, 118, 544, 281], "S6.F10"),
    (9, 9, [345, 518, 544, 661], "S6.F11"),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError("PDF identity differs; review crop coordinates before rendering")
    args.output.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(args.pdf)
    figures = []
    for number, page, rectangle, html in CROPS:
        path = args.output / f"figure-{number:02d}.png"
        pix = pdf[page-1].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(rectangle), alpha=False)
        pix.save(path)
        figures.append(dict(pdf_figure=number, pdf_page=page, html_anchor=html,
                            crop_points=rectangle, file=path.name, width=pix.width, height=pix.height,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(source="https://arxiv.org/pdf/2609.32722v1", revision="v1",
                    author="Yuntai Bao et al.", pdf_sha256=sha, pages=len(pdf), dpi=216,
                    transform="PDF figure-region rendering; original axes, lines and legends preserved; captions outside crop",
                    numbering="PDF figure numbers; HTML conversion renumbers some subfigures",
                    figures=figures)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

if __name__ == "__main__":
    main()
