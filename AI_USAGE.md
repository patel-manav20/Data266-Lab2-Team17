# AI Usage

## Manav

| Tool | Used for | What I checked myself |
|---|---|---|
| Claude (chat) | Planning Part 1's design: the fact store and resolver, the routing order, which conventions to test, what to measure before adopting a change. | I set the measurement rules myself (pre-set composite and per-set bars) before any run, and only accepted a change when the numbers cleared them. |
| Claude Code (in Cursor) | Writing and editing `p1_rag.ipynb` and `p1_eval.ipynb`, debugging failing tests, building the Kaggle probe notebooks used to measure a change before adopting it, running the Mac regression (`RUN_DIAGNOSTICS=True`, model off) before every push. | I read every diff before it went into a commit, read the printed test and fuzz-test output after every run, and re-ran the leak check and the push-readiness scan myself before any commit that touched the notebook. |
| Kaggle CLI (`kaggle kernels push` / `status` / `output`) | Pushing a notebook as a new kernel version, polling it to a terminal status, downloading its log and `submission.csv`. | I downloaded and read the log myself (health line, errors, status counts, the validator line) and diffed the new `submission.csv` against the previous version row by row before calling a change correct. |
| `lab2-offline-wheels` (a Kaggle dataset I built) | Bundling the extra pip packages the Kaggle kernel can't fetch online (`enable_internet: false`): chromadb, rank_bm25, pdfplumber, pypdfium2, pydantic_settings. | I checked the kernel's own install log each run to confirm the right wheel versions were used and nothing fell back to a network install. |
| Claude Code (CLI) | Part 2: writing and editing `p2_finetune.ipynb` and `p2_generate_eval.ipynb`, the LoRA training wrapper, the KID/CLIP/LPIPS evaluation cells, the blinded rater-pack HTML, and the dataset/memorization-check cells. Part 3: writing and editing all five agents in `src/`, the orchestrator (including the shared-state `run_both()` fairness fix), `test_stub.py`'s 162 stub tests, the judging-pack HTML, and `p3_agents.ipynb`. Driving the `colab` CLI (`new`, `drivemount`, `upload`/`download`, `exec -f`, `status`/`usage`/`stop`) for every Colab T4 run in both parts. | I read every diff before it went into a commit, read every notebook's printed cell outputs after each GPU run (losses, timing, GPU memory, scenario pass/fail tables, the determinism and fairness checks) before treating a change as correct, and re-ran the leak check and `colab stop`/`colab usage` myself after every session, including after failures. |
| Claude Code (CLI) | Part 1, `exp/p1-push` branch: designing and measuring the A2 rulebook answerer (`READER_A2`) -- the system prompt, the JSON output schema, the blend with the extractive route's own sentences, the citation-inheritance design, the gate-probe harness used to measure it (a stratified-sample mode and a full-sweep mode), and the Kaggle timing/submission runs. | With the instructor's explicit permission (kept in writing by me), another team's notebook was read as a design reference for two ideas only -- a rulebook-style prompt for a text answerer, and per-question chart reading by the vision model -- never its code or prompt text, which are entirely our own. I read every diff before committing, read the gate-probe output (composite/per-set numbers, error counts) on two non-overlapping samples before adopting the change, and read the Kaggle run log and diffed the new `submission.csv` against the previous version row by row before calling it correct. |

Every commit on `member_manav` made with Claude Code carries a `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer, so the full list is in `git log`.

## Pramod

| Tool | Used for |
|---|---|
