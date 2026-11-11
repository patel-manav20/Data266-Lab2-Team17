# AI Usage

## Manav

| Tool | Used for | What I checked myself |
|---|---|---|
| Claude (chat) | Planning Part 1's design: the fact store and resolver, the routing order, which conventions to test, what to measure before adopting a change. | I set the measurement rules myself (pre-set composite and per-set bars) before any run, and only accepted a change when the numbers cleared them. |
| Claude Code (in Cursor) | Writing and editing `p1_rag.ipynb` and `p1_eval.ipynb`, debugging failing tests, building the Kaggle probe notebooks used to measure a change before adopting it, running the Mac regression (`RUN_DIAGNOSTICS=True`, model off) before every push. | I read every diff before it went into a commit, read the printed test and fuzz-test output after every run, and re-ran the leak check and the push-readiness scan myself before any commit that touched the notebook. |
| Kaggle CLI (`kaggle kernels push` / `status` / `output`) | Pushing a notebook as a new kernel version, polling it to a terminal status, downloading its log and `submission.csv`. | I downloaded and read the log myself (health line, errors, status counts, the validator line) and diffed the new `submission.csv` against the previous version row by row before calling a change correct. |
| `lab2-offline-wheels` (a Kaggle dataset I built) | Bundling the extra pip packages the Kaggle kernel can't fetch online (`enable_internet: false`): chromadb, rank_bm25, pdfplumber, pypdfium2, pydantic_settings. | I checked the kernel's own install log each run to confirm the right wheel versions were used and nothing fell back to a network install. |

Every commit on `member_manav` made with Claude Code carries a `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer, so the full list is in `git log`.

## Pramod

| Tool | Used for |
|---|---|
