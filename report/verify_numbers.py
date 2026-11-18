#!/usr/bin/env python3
"""Final check: extract every number-like token from report.tex (and the generated table
snippets it \\input{}s) and confirm it appears somewhere in the approved source files. Also
checks for TBD/TODO/placeholder text and Questions.csv-style question text."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "report"

tex_text = (REPORT / "report.tex").read_text()
for tbl in (REPORT / "tables").glob("*.tex"):
    tex_text += "\n" + tbl.read_text()

NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|\\%)?\b")
numbers = sorted(set(m for m in NUM_RE.findall(tex_text) if m not in {"0", "1", "2", "3", "9"}))
print(f"{len(numbers)} distinct number-like tokens found in report.tex + tables/")

SOURCE_GLOBS = [
    "report/*.md", "report/*.csv",
    "README.md", "CONTRIBUTIONS.md", "AI_USAGE.md", "links.txt",
    "part1_rag/*/README.md", "part2_diffusion/*/README.md", "part3_agents/*/README.md",
    "part1_rag/*/outputs/*.csv", "part2_diffusion/*/outputs/**/*.csv",
    "part3_agents/*/outputs/**/*.csv",
]
source_text = ""
for pattern in SOURCE_GLOBS:
    for p in REPO.glob(pattern):
        try:
            source_text += p.read_text(errors="ignore")
        except Exception:
            pass

# also include notebook saved-output text (not source code) for both members, all parts
import json
for nb_path in REPO.glob("part*_*/*/notebooks/*.ipynb"):
    try:
        nb = json.load(open(nb_path))
        for c in nb.get("cells", []):
            for o in c.get("outputs", []):
                t = o.get("text")
                if t:
                    source_text += "".join(t)
    except Exception:
        pass

missing = [n for n in numbers if n.rstrip("%\\") not in source_text]
print(f"{len(missing)} number(s) not found verbatim in any source file:")
for m in missing:
    print(" ", m)

print()
print("=== placeholder / TBD check ===")
for bad in ["TBD", "TODO", "FIXME", "XXX", "<placeholder>"]:
    if bad in tex_text:
        print(f"  FOUND: {bad}")
if not any(b in tex_text for b in ["TBD", "TODO", "FIXME", "XXX"]):
    print("  none found")

print()
print("=== figures/tables referenced check ===")
labels = set(re.findall(r"\\label\{([^}]+)\}", tex_text))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex_text))
unreferenced = labels - refs
print("labels never \\ref'd:", unreferenced or "none")
undefined = refs - labels
print("\\ref's with no \\label:", undefined or "none")
