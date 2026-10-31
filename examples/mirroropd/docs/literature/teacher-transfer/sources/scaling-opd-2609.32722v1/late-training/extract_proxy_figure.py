# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract PDF Figure 10 as an attributed figure, without digitizing proxy values."""
import argparse
import hashlib
import json
from pathlib import Path
import fitz

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
assert sha == "1444fcc17452c608925c7429c4846dfaf8a6886db68541baa179ce0b0e11e53e"
args.output.mkdir(parents=True, exist_ok=True)
doc = fitz.open(args.pdf)
clip = [70, 60, 545, 571]
out = args.output/"original-figure-10.png"
doc[22].get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(clip),alpha=False).save(out)
(args.output/"original-figure-10-source.json").write_text(json.dumps(dict(
    source="https://arxiv.org/pdf/2609.32722v1", pdf_sha256=sha, pdf_page=23,
    pdf_figure=10, html_figure=12, crop_points=clip, dpi=216,
    transform="Figure-only PDF rendering; original plot and colors retained",
    sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2)+"\n")
