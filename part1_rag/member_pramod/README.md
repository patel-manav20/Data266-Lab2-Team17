# Part 1: Multimodal RAG over a read-only knowledge base (Pramod)

The notebook answers multi-turn questions about three economics PDFs (Blanchard chapters 1–2, an outturn supplement, a policy briefing). It reads text, tables and charts. Each answer comes with a status, citations and figure references, as the Kaggle competition requires. Only open models are used, on a single T4.

## Pipeline

1. **Ingestion** (PyMuPDF): text chunks with document and page, tables read cell by cell, figure crops.
2. **Figures**: `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) transcribes every trusted chart into text; chart values are paired with their years.
3. **Fact store**: every table value and every value stated in prose becomes a fact (entity, variable, year, value, forecast or outturn, source, trust). Low-trust pages (DOC3 Appendix C) produce no facts.
4. **Retrieval**: ChromaDB with `BAAI/bge-base-en-v1.5` embeddings plus BM25, merged by reciprocal rank fusion; table rows are indexed individually, and a named figure or table is looked up exactly.
5. **Reconciliation engine** (plain Python): applies the manifest's precedence policy to the facts a question asks about. It handles a revision notice, an outturn replacing a forecast, and a conflict between sources, where the higher-trust value is preferred. For "change in" questions it adds the previous year and the actual change.
6. **Answer**: `unsloth/Qwen3-8B-unsloth-bnb-4bit` (thinking off) returns JSON with facts, a check and an answer. Code then:
   - verifies every value against the evidence;
   - evaluates any arithmetic;
   - sets the status by fixed rules (conflict, false premise, session override for user-supplied values, not found);
   - builds the citations.
7. **Conversation state**: values the user supplies ("assume…", "for this chat…") are kept per conversation and never written to the index. Requests to change the knowledge base are refused, and follow-up questions are rewritten into standalone form.

Citation format, settled on the leaderboard: tables are cited by label only (`DOC2:Table A-2`), and figures by label plus page (`DOC2:Figure A-4,DOC2:p5`).

## How to run

1. Open `notebooks/p1_rag.ipynb` on Kaggle with a T4 GPU, attached to the competition data. Run all.
2. The notebook transcribes the figures from scratch (`USE_FIG_CACHE = False`). It writes `submission.csv`, validates it with the competition's `validate_submission.py`, and then evaluates the golden set.
3. A full run takes about 100 minutes:
   - installing packages and loading models;
   - figure transcription, about 9 minutes;
   - the 40 competition questions, about 42 minutes;
   - the 25-question golden set, run twice (baseline and final).

The notebook is generated from `src/p1_rag_cells.py`, which uses `# %%` cell markers:

```sh
python part1_rag/member_pramod/src/build_p1_notebook.py part1_rag/member_pramod/src/p1_rag_cells.py part1_rag/member_pramod/notebooks/p1_rag.ipynb tests/p1_golden_pramod.json
```

## Results (Kaggle run v20, commit `69e8de2`; the executed notebook is `notebooks/p1_rag.ipynb`, results in `outputs/`)

Golden set: `tests/p1_golden_pramod.json`, 25 questions written before tuning and never edited.

| System | Correct (of 25) | Status right | Recall@5 | Citation accuracy | Median s / question |
|---|---|---|---|---|---|
| Baseline (dense retrieval only) | 17 | 22 | 0.958 | 0.593 | 57.2 |
| Final (hybrid retrieval + reconciliation engine) | **22** | **25** | **1.000** | **0.723** | 58.0 |

Remaining misses:
- G10 gives the right fitted equation for Figure B-2 but leaves out the 2.9% growth threshold printed with it.
- G18 and G19 are a two-turn conversation about Greece. G18 compares a user-assumed value with Spain from a dense country chart and gets the gap wrong. G19 ("its rate two years earlier") needs Figure A-6 and is answered from the wrong chart.

Public leaderboard for the notebook's own submissions: v6 0.728 → v9 0.721 → v13 0.857 → v16 0.857 → **v18 0.868 → v19 0.868**. Format tests on the same answers:
- table citations with pages: 0.857;
- tables by label only: 0.868;
- figures also without pages: 0.862.

## Design rules we kept

- Every rule is stated in terms of the manifest's policy or the documents' structure. There is no question-specific code, and a scan checks that no public question text appears in the source.
- Prompt examples use made-up places and numbers, not knowledge-base content.
- A failing question produces a valid `not_found` row instead of stopping the run.
