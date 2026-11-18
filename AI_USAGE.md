# AI Usage

## Manav

I used Generative AI throughout this project as a helper whenever I got stuck. It helped me plan the design, helped me with the code and tests when I got stuck, and cleared my doubts along the way. I still made the main decisions and checked every result myself before keeping it.

| Part | How AI helped | What I did myself |
|---|---|---|
| Part 1 planning | Helped me plan the fact store and resolver, the routing order, which conventions to test, and what to measure before adopting a change. | I set the measurement rules myself (the composite and per-set bars) before any run, and only accepted a change when the numbers cleared them. |
| Part 1 code (`p1_rag.ipynb`, `p1_eval.ipynb`) | Helped me with the code when I got stuck, debugging failing tests, the Kaggle probe notebooks, and the Mac regression run before each push. | I read every diff before committing, read the test and fuzz-test output after every run, and re-ran the leak check and push-readiness scan myself. |
| Part 1 rulebook answerer (`READER_A2`, `exp/p1-push` branch) | Helped me with the system prompt, the JSON output schema, the blend with the extractive route, the citation design, and the gate-probe harness. | With the instructor's written permission, I read another team's notebook as a design reference for two ideas only: a rulebook-style prompt for a text answerer, and per-question chart reading by the vision model. I did not use their code or prompt text. I checked the gate-probe numbers on two separate samples before adopting the change, and diffed the new `submission.csv` against the previous one row by row. |
| Part 2 (`p2_finetune.ipynb`, `p2_generate_eval.ipynb`) | Helped me with the LoRA training wrapper, the KID/CLIP/LPIPS evaluation cells, the blinded rater pack, the memorization check, and running the Colab T4 sessions. | I read every cell output after each GPU run (losses, timing, GPU memory) before treating a change as correct, and stopped and checked Colab usage myself after every session. |
| Part 2 failure modes (P11, P16) | Helped me draft the image descriptions from the saved generated images. | I looked at both images myself and checked every sentence against what is actually visible. The notebook and README label these notes as AI-drafted and checked by me. |
| Part 3 (`src/`, `p3_agents.ipynb`) | Helped me with the five agents, the orchestrator (including the shared-state `run_both()` fairness fix), the 162 stub tests in `test_stub.py`, and the judging pack. | I read every diff, checked the scenario pass/fail tables and the determinism and fairness checks before calling a change correct. |

Tools that are not AI:

- **Kaggle CLI**: I used it to push notebooks, check their status, and download logs and `submission.csv`. I read every log myself (health line, errors, status counts, validator line).
- **`lab2-offline-wheels`**: a Kaggle dataset I built with the pip packages the offline kernel cannot download (chromadb, rank_bm25, pdfplumber, pypdfium2, pydantic_settings). I checked the install log each run to confirm the right versions were used.

Every commit on `member_manav` made with AI help carries a `Co-Authored-By` trailer, so the full list is in `git log`.

## Pramod

(to be confirmed by Pramod -- drafted from what his commits and README show, not from a direct
statement by him)

| Tool | Used for |
|---|---|
| Claude Opus 5.5 | 50 of his 52 commits on `member_pramod` carry a `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer (`git log member_pramod`), across all three parts. |
| Codex | One commit's subject line is explicitly marked "(changes by Codex)" (`8dc1570`, Part 2: Drive persistence, input validation, CPU tests). His own `part2_diffusion/member_pramod/README.md` states under "AI tools used": "Codex reviewed and fixed the training/evaluation workflow and added CPU regression checks." |
