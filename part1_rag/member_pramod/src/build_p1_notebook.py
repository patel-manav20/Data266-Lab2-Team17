import json
import re
import sys

import nbformat as nbf

src, out, golden = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
parts = re.split(r"^# %%(.*)\n", text, flags=re.M)
cells = []
for marker, body in zip(parts[1::2], parts[2::2]):
    body = body.rstrip("\n")
    if "markdown" in marker:
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n")]
        cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
    else:
        body = body.replace("GOLDEN = __GOLDEN__", "GOLDEN = " + json.dumps(json.load(open(golden)), indent=1))
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                "language_info": {"name": "python"}})
nbf.write(nb, out)
print(f"wrote {out}: {len(cells)} cells")
