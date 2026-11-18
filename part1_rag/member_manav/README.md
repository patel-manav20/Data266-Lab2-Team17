# Part 1: Multimodal RAG (Manav)

This part answers questions about three economics PDFs (a textbook chapter, a data supplement, and a
policy briefing). It reads the text, the tables and the charts, and every answer carries its sources. The
documents are read-only: a user can ask "what if this value were X" for one chat, and the system remembers
that for the rest of the chat, but it never changes what the documents say, and it never agrees to "update"
them. No language model writes any part of an answer; a model is used once, at ingestion time, to name the
series in a few charts whose colors or legend can't be told apart from the image alone.

## Architecture

```
PDFs + kb_manifest.json
        |
        v
Ingestion: text chunks (doc_id, page, section) + figure/table objects (label, caption, crop, trust level)
        |                              |
        |                      Chart values: read from the PDF's vector paths where the chart has them
        |                      (DOC1), color tracing for the raster charts (DOC2, DOC3); Qwen3-VL names
        |                      which color is which series only where the legend or caption can't say so
        v                              v
Fact store (every table cell, keyed by entity + variable + period)   Figure/table objects + chart readings
        |                                                                         |
        v                                                                         |
Resolver: picks between a conflicting, revised or forecast-vs-outturn value       |
        |                                                                         |
        +-----------------------------------+--------------------------------------+
                                             v
                      Hybrid text retrieval: bge-base dense + BM25, fused by rank,
                      reranked by Qwen3-Reranker-4B (bge-small + BM25, no reranker, when the GPU is off)
                                             |
                                             v
        Question -> conversation memory -> routes (in order): a request to change the documents is
        refused; an assumed value stays in that chat only; a named figure or table is answered from that
        object; a change/gap question uses a small calculator; then the resolver's outcome, with a check
        of any number the question itself claims; growth-series and comparison routes for questions that
        name an entity and a variable but no period; an explanatory question gets a short text answer plus
        one fact-store sentence; everything else falls to the text route or gets not_found
                                             |
                                             v
                      Every number in an answer is checked (grounded) against the cited evidence
                                             |
                                             v
                 JSON answer (answer, status, citations, figure_references) -> submission.csv
```

Appendix C of Document 3 is marked low-trust. It is never cropped, opened by a chart reader, sent to the
naming model, or cited -- see the Safety section in `p1_eval.ipynb` for the counters. Any instruction found
inside a document or a chart image is treated as plain text to look for, never as something to obey.

## Conventions

A handful of small, measured rules decide exactly what goes in `citations` and `figure_references`, and
what `status` an answer gets. Each was tested as a one-field change against the real submission, scored by
the board, before it went into the notebook as a general rule (see "citation and status conventions from
the board tests" in the git history, and the Kaggle history table in `p1_eval.ipynb`, section (e)):

- A chart-route answer cites only the figure label(s) it used, never a table, even when it also quotes a
  table value for a forecast/outturn cross-check.
- An answer that revises a value, or replaces a forecast with an outturn, cites and lists in
  `figure_references` only the source of the value given as current -- not the one it replaced.
- A figure with a corrected sibling (for example "Figure B-5" and "Figure B-5 (corrected)") lists both
  labels in `figure_references` when the question names the plain label; this was tested as a change
  (listing only the cited version) and scored the same as the current behaviour, so it was kept as is.
- A refusal of a request to change the documents gets status `session_override`, not `answered`.

## How to run

On Kaggle:
1. Open `p1_rag.ipynb` as a Kaggle notebook with a T4 GPU.
2. Attach the competition data and the two models (`qwen-3-reranker/4b`, `qwen-3-vl/4b-instruct`) and the
   four datasets (`lab2-offline-wheels`, `bge-small-en-v1-5-baai`, `bge-base-en-v1-5-baai`,
   `bge-reranker-v2-m3-baai`) as inputs -- see `links.txt`.
3. `RUN_DIAGNOSTICS` defaults to `False` (set it to `True`, or set the environment variable
   `RUN_DIAGNOSTICS=True`, to also run every test in section 13 -- this is most of the run time).
4. Run All. The notebook reads `QUESTIONS_PATH` (default: the competition's `Questions.csv`) and writes
   `submission.csv`, then `validate_submission.py` checks it.

On a Mac or another machine with no GPU:
1. `pip install -r requirements.txt`
2. Put the three PDFs, `kb_manifest.json` and `Questions.csv` in one folder; the notebook searches upward
   from its own folder (or `$DATA_DIR`) to find it.
3. Set the environment variable `RERANK_MODEL=off` (the reranker and the chart-naming model both need a
   GPU; with this set, the text route falls back to bge-small + BM25 with no reranking, and chart reading
   falls back to vector paths and color tracing only -- a few charts lose their series names).
4. Run `p1_rag.ipynb`, then `p1_eval.ipynb` the same way (it executes `p1_rag.ipynb`'s own cells, so it
   needs the same environment variables).

Citations use the PDF page index, so `DOC1:p3` means the third page of that PDF.

## Live demo

To answer a grader's own questions on the spot (not from `Questions.csv`), use `demo_answer()` in
`notebooks/p1_rag.ipynb`, section 12b ("Demo: answer any question on the spot"), right after the
submission-format check and before the test/diagnostics section. It calls the same `build_answer()`/
`Conversation` the real pipeline uses, unchanged -- nothing in this cell is new answer logic.

**Setup, once, before the demo cell can run.** Run the notebook up through section 12 (Run All, or Run
All Above from the demo cell) -- this ingests the PDFs, builds the fact store, and loads the embedder
and reranker. Measured on v35's real Kaggle run: about **10 minutes** (594 of the run's 942.7 total
seconds; the rest is the regular 40-question answering pass, which the demo doesn't need to wait for).
After that one-time setup, re-run just the demo cell with new questions as many times as needed --
each answer takes under a second unless it reaches the A2 answerer (about 24 seconds, for an
explanatory question the extractive route alone can't resolve).

**Answering 3 grader-chosen questions, including a 2-turn conversation:**
```python
# a 2-turn conversation: the second question can refer back to the first
demo_answer(["<grader's first question>", "<grader's follow-up to it>"], conversation_id="demo-1")
# a separate, single-turn question in its own conversation
demo_answer(["<grader's third question>"], conversation_id="demo-2")
```
Each call prints, per question: the answer, its status, its citations and its figure references --
the same fields `Questions.csv` answers get, for any question typed in live.

**Phrasing a follow-up turn, from testing this live**: a follow-up that drops the entity/variable and
changes only the period or entity ("What about 2009?", "What about China?") threads back to the first
turn reliably. A follow-up that reintroduces a bare noun instead ("How does that compare with *the
rate* in 2009?") can resolve to the wrong variable -- found directly while testing this cell (see
below), not assumed. Prefer the elliptical form for a reliable follow-up.

**If the GPU is slow** (a queued or unavailable Kaggle T4 session): run the exact same `demo_answer()`
cell on a Mac or any machine with no GPU instead, with `RERANK_MODEL=off` set (see "On a Mac or another
machine with no GPU" above) -- it gives the same deterministic fact-store/chart/label answers
immediately; only `route=="text"` explanatory questions answer slightly differently (bge-small + BM25,
no reranker, no A2) while still being correct and grounded. This is a genuine fallback, not a
workaround: Team Rule 3 requires the pipeline to run correctly with no GPU or model attached at all,
and this is exercised by every Mac regression run in this project's history.

Tested on a Mac in model-off mode with three made-up questions (not from `Questions.csv`): two turns of
one conversation ("What was the unemployment rate in the United States in 2010?" then "What about
2009?") and one standalone question ("Why did world output growth slow down in 2012?"). All three
answered correctly with citations -- the follow-up correctly carried the entity and variable forward
from turn 1 and reported 9.3% for 2009 with both documents' sources. A first attempt at the follow-up,
phrased as "How does that compare with the rate in 2009?", resolved to the wrong variable (inflation,
not unemployment) -- kept here as the reason for the phrasing guidance above, not hidden.

## Results

Full tables and the code that produced them are in `part1_rag/member_manav/notebooks/p1_eval.ipynb`
(run once on a Mac, no GPU, `RUN_DIAGNOSTICS=True`; its outputs are saved in
`part1_rag/member_manav/outputs/`). Summary:

**Golden set (25 items, never used to choose any setting).** Measured in two modes, named in each table.
Swapping the text route's retrieval for dense-only (`baseline_retrieve`) gave the identical score on every
one of the 25 items in both modes, because none of them actually reach the text route -- they all resolve
through the fact store, the chart route or the label route.

*Mac mode (`p1_eval.ipynb`, no GPU, reranker and chart-naming model off):*

| | answer | status | figure F1 | citations | **score** |
|---|---|---|---|---|---|
| overall | 0.577 | 0.880 | 0.733 | 0.703 | **0.694** |

| category | score | n |
|---|---|---|
| unanswerable, injection, false_premise, multi_turn_user_value | 1.000 | 1, 1, 1, 2 |
| outturn_over_forecast | 0.889 | 3 |
| two_source | 0.894 | 3 |
| **figure** | **0.550** | 6 |
| **fact_table** | **0.540** | 5 |
| **multi_turn** | **0.333** | 3 |

*Production mode (Kaggle, `manavpatel2025/lab2-golden-prod-probe`, Qwen3-Reranker-4B + Qwen3-VL chart
naming attached, as v31):*

| | answer | status | figure F1 | citations | **score** |
|---|---|---|---|---|---|
| overall | 0.637 | 0.960 | 0.733 | 0.783 | **0.750** |

| category | score | n |
|---|---|---|
| unanswerable, injection, false_premise, multi_turn_user_value | 1.000 | 1, 1, 1, 2 |
| outturn_over_forecast | 0.950 | 3 |
| two_source | 0.894 | 3 |
| fact_table | 0.700 | 5 |
| **figure** | **0.553** | 6 |
| **multi_turn** | **0.467** | 3 |

Production mode scores higher overall (0.750 vs 0.694) and on every component; **figure and multi_turn are
still the two weakest categories in both modes** (see Limitations). The local scorer formula is the local
copy of the Kaggle formula: 0.40 answer facts + 0.20 status + 0.20 figure F1 + 0.20 citations.

The golden expectations (`citations` and `figure_references` in `eval/p1_golden_manav*.json`) were written
before the board tests, and for the three `outturn_over_forecast` items they use the earlier citation
convention (both the current value's source and the one it replaced). The conventions adopted from the
board tests (C2) now cite only the current source, so those three items' citations and figure-F1
components score lower than they would against the old convention -- this is the adopted behaviour working
as intended, not a regression. The golden items themselves were not changed.

**Golden set on Kaggle, v35 (A2 on), four configurations -- the reranker bonus.** Measured on
`manavpatel2025/lab2-p1-finaleval` version 4 (single T4, the same sources as v35), read directly from its
log and saved CSVs, not retyped from memory. (Version 3's "dense-only baseline" had a real bug -- it left
the strong reranker loaded, so `text_answer()` took the reranked path regardless of which retrieval
function a global name was swapped to; `text_route()` only consults that name, and `text_route()` only
runs when no reranker is loaded. Fixed as an evaluation-harness change only, not a pipeline-code change --
confirmed in version 4's own log that no reranker was loaded for the no-reranker configurations below.)

| configuration | score | answer | status | figure F1 | citations | A2 used |
|---|---|---|---|---|---|---|
| (a) dense-only baseline, no reranker (A2 off) | 0.7693 | 0.6433 | 0.96 | 0.8133 | 0.7867 | 0/25 |
| (d) hybrid dense + BM25, no reranker at all | 0.7693 | 0.6433 | 0.96 | 0.8133 | 0.7867 | 0/25 |
| (b) final minus the strong reranker (BAAI/bge-reranker-v2-m3) | **0.7967** | 0.7033 | 1.00 | 0.8267 | 0.75 | 0/25 |
| (c) final v35, current default (Qwen3-Reranker-4B) | 0.7693 | 0.6433 | 0.96 | 0.8133 | 0.7867 | 0/25 |

A2 never fires on this set in any configuration (none of the 25 items reach `route=="text"` explanatory);
for (a) and (d) it is additionally structurally impossible, since `build_answer()`'s A2 hook requires a
reranker to be loaded and both deliberately run without one. **(a), (d) and (c) are not just equal in
score -- they produce byte-identical answer text on all 25 items**, verified directly, not inferred: three
different retrieval paths (dense-only; dense+BM25 unranked; dense+BM25 reranked by Qwen3) converge on the
same final answer every time on this set.

**The lab's bonus, (d) vs (c), reported plainly: zero improvement.** Adding Qwen3-Reranker-4B over no
reranker at all changes nothing here -- same score, same answer on every item, same per-category
breakdown. **No bonus is claimed.** Per-item and per-category numbers are in `p1_eval.ipynb` section (h)
and `outputs/p1_eval_golden_kaggle_3config_by_category.csv`.

**(b) vs (c) -- reranker *choice* (not reranker vs none) does matter here, and it is negative for the
strong reranker**: 0.7967 (b) vs 0.7693 (c), measurable on only 4/25 items (G01, G05, G09, G22 -- the
only ones reaching the text route); the weak reranker wins 3 of those 4. This and the (d) vs (c) result
are not a contradiction -- no reranker does as well as the strong one (zero bonus), while the *other*
reranker happens to do better than the strong one, on this specific 25-item set.

**Context, a different measure, not the golden-set bonus**: the Recall@k/MRR retrieval metrics just below
favor hybrid retrieval generally, on far more items (25, 100) than the 4 that are retrieval-sensitive in
the golden-set score comparison above -- that measures whether the right source is retrieved at all, not
whether the final score changes, and the two are not in tension.

**Two pipeline bugs found while reviewing golden items, not fixed because the code was frozen at the
selected Kaggle version (v35)** (documented here and in Limitations): a label-route entity mismatch (G10,
G11) and an Okun's-law parameter precision gap (G07). Golden is not treated as a clean held-out measure
for exactly this reason -- see Limitations for the detail on each.

**Baseline (a) vs final (c), v35/A2 -- the report's baseline-vs-final row.** Same corrected configuration
(a) as the table above, next to (c):

| system | score | answer | status | figure F1 | citations |
|---|---|---|---|---|---|
| (a) dense-only baseline, no reranker | 0.7693 | 0.6433 | 0.96 | 0.8133 | 0.7867 |
| (c) final v35, current default | 0.7693 | 0.6433 | 0.96 | 0.8133 | 0.7867 |

Identical on every component and every category (`p1_eval.ipynb` section (i)) -- the same finding as
above, stated as the baseline-vs-final comparison specifically.

**The four Part 1 measurement types, in one table** (`p1_eval.ipynb` section (i),
`outputs/p1_eval_four_measurement_types.csv`):

| measurement type | metric | value |
|---|---|---|
| primary | correct answers out of 25 golden (score >= 0.8) | 15/25 (60%) |
| secondary | Recall@5 (golden, hybrid retrieval) | 0.92 |
| secondary | citation accuracy (citations component, v35 config) | 0.7867 |
| human | groundedness rating, manav (n=25, 1-5) | mean 4.56 -- see "(d) Human rating of groundedness" |
| efficiency | median seconds per question (v35 real 40-question run) | 0.14 s |

**Human rating of groundedness** (`p1_eval.ipynb` section (d), `outputs/p1_eval_groundedness_results.csv`).
manav rated all 25 golden answers on a 1-5 groundedness scale against their own cited evidence: mean
**4.56** (16 fives, 8 fours, 1 two). Raters in: manav. Missing: pramod, outside rater 1 -- neither has
rated yet. Exact/within-1 agreement and the weighted Cohen's kappa between raters are not computable with
only one rater in; both will be added once a second rater's sheet is filled in.

**Retrieval (Recall@k, MRR).** On the golden set, hybrid retrieval beats dense-only (MRR 0.844 vs 0.778,
R@1 0.76 vs 0.68); on the auto-generated drop-clause questions (100, made from the corpus's own sentences)
hybrid is also ahead (MRR 0.873 vs 0.795, R@1 0.80 vs 0.73). On dev A and dev B (hand-written formal,
colloquial and conceptual phrasings) the two are closer, and dense-only is slightly ahead on dev A's R@1
(0.83 vs 0.79) -- these sets are small (24 and 12 questions), so this is noise, not a pattern.

**Safety.** The one low-trust object (Figure C-1) is never cropped, never opened by a chart reader or the
naming model, and never cited. The "low-trust text never used" check passed 5 of 5 on this run (the corpus
has 5 low-trust sentences with a number or an instruction in it). A separate, earlier, test-only probe (the
model-based anchor-picking feature, since measured and turned off) appended a planted instruction to 36
real dev questions: the planted word appeared in the model's answer 0 of 36 times, and the model's raw
reply failed to be a clean choice 0 of 36 times, in two separate runs of that probe.

**Safety, re-run with A2 on (v35 config).** All 16 named test groups in section 13.1-13.13 passed 100%
with `READER_A2=True` -- including test (j), "low-trust text never used" (5/5), and every fact-store,
session-override, false-premise and calculator test. No failures. (Same probe as the three-configuration
table above.)

**Kaggle history.** v2 (first submission) 0.637 -> v17 (refusal policy, explanatory questions, chart
values) 0.751 -> v21/v22 0.784 -> v29/v30 0.808 -> the four board-tested conventions (0.808-0.819) -> v31
(three of those four, adopted as general rules) 0.818 -> v34 0.828 (byte-identical to v32) -> **v35 0.853
(current): an A2 rulebook answerer (`READER_A2`, on by default) blends a short Qwen3-8B-generated
passage (4-bit; Qwen3-VL-4B-Instruct text-only fallback) ahead of the extractive route's own chosen
sentences, for explanatory questions that fall through to the raw sentence-splice route -- status,
citations and figure_references are always inherited from the extractive route's own output, never
taken from the model. Gate-tested on two non-overlapping 120-item samples before adoption (+0.0209 and
+0.0431 composite, zero per-set drops either time). 11 of 40 real rows changed, all the same trigger
class, zero metadata changes.** A second piece (per-question chart reading, `READER_A2_CHARTS`) was
built and gate-tested the same way but showed no measurable gain (the gate's own test sets have only
1-2 chart-reading-eligible questions), so it stays off by default. Full table with what changed at each
version in `p1_eval.ipynb`, section (e).

**Run time and efficiency.** Kaggle, version 35 (current, A2 on), diagnostics off: 942.7s (~15.7 min)
total for the real 40-question run, **median 0.14 s per question** overall (dominated by the many fast
deterministic answers; the 11/40 questions that reach the A2 answerer cost 23.57 s/call, 259.3s total),
0 errors, validator passes. On the golden-set probe above (25 items, mostly fact-store/chart routes,
zero A2 calls on this set): median 0.05 s per turn for every configuration. Version 31's numbers (all
datasets and the vision model, diagnostics off): about 12.7 minutes total (490s building the index and
the fact store, 170s loading the models, 92s answering the 40 questions and writing the submission, 9s
for the (skipped) diagnostics check). On a Mac with no GPU and every diagnostic on, loading the pipeline
and answering the golden set once more took about 8 minutes.

## Limitations

- **Figure and multi_turn score noticeably lower than the rest** on the golden set, in both the Mac and
  the production run (figure 0.550/0.553, multi_turn 0.333/0.467) -- both ask for a specific number, from
  a chart reading or across turns, which is harder than the structured fact-store lookup most other
  categories use. fact_table scores much better in production (0.700) than on a Mac (0.540), since several
  fact_table items depend on the reranker to find the right prose sentence.
- **Three raster figures' series names can't be fully verified from the image alone:** Figure B-1 and
  Figure B-2 each have one name but three colors (no caption or legend to tell them apart), and Figure B-5
  only matches the table on 3 of its 4 printed labels. The chart reader still gives a value in each case,
  just with a lower confidence the name is right.
- **Two pipeline bugs, found while reviewing golden items on the v35/A2 Kaggle probe. Not fixed because
  the code was frozen at the selected Kaggle version (v35).** (1) Label-route entity mismatch: a question naming a figure and asking for a specific
  entity's own value (golden G10 "According to Figure 1-6, what was China's population..."; G11 the same
  for the Euro area) gets the figure's subject right ("Figure 1-6 (DOC1) shows: China") but then reports
  an unrelated fact for the wrong entity and variable in both cases -- the identical filler "4.0% for the
  world output growth for World in 2010" regardless of which figure or entity was actually asked about.
  (2) Okun's-law parameter precision: golden G07 expects slope -0.42 and threshold 3.1% (presumably read
  off Figure B-1's own fitted line); `okun_parameters()`'s regex extraction from body-text mentions gives
  -0.4 and 3% -- one decimal place less precise than the chart. The arithmetic that follows from those
  inputs is internally correct; the gap is upstream, in how much precision the extraction regex carries
  over from prose into the calculation. Both are exactly why the golden set is not used to choose
  anything in this pipeline -- see `p1_eval.ipynb` section (h) for the full comparison these were found in.
- **No GPU on a Mac** means no reranker (text retrieval falls back to bge-small + BM25 only) and no chart
  naming model, so a Mac run of `p1_eval.ipynb` understates what the same questions score on Kaggle, where
  both are on. The Kaggle numbers in this README (the history table, the v31 run time) come from the real
  Kaggle run, not the Mac eval.

## Licenses of every model and dataset used

| Tool, model or dataset | Use | License |
|---|---|---|
| pdfplumber | PDF text, words, positions and table extraction | MIT |
| pypdfium2 | PDF page rendering and figure/table crops | Apache-2.0 / BSD-3 |
| rank_bm25 | Keyword (BM25) search | Apache-2.0 |
| ChromaDB | Vector database for dense retrieval | Apache-2.0 |
| BAAI/bge-small-en-v1.5 (sentence-transformers) | Dense text embeddings (Mac / baseline retrieval) | MIT |
| BAAI/bge-base-en-v1.5 (sentence-transformers) | Dense text embeddings (hybrid retrieval, Kaggle) | MIT |
| Qwen/Qwen3-Reranker-4B (transformers) | Text-route reranking (Kaggle, GPU only) | Apache-2.0 |
| Qwen/Qwen3-VL-4B-Instruct (transformers) | Names chart series/categories where the legend or caption can't (Kaggle, GPU only); fallback for the A2 answerer below if Qwen3-8B can't load; per-question chart reading (`READER_A2_CHARTS`, off by default) | Apache-2.0 |
| Qwen/Qwen3-8B (transformers, 4-bit) | A2 rulebook answerer (`READER_A2`, on by default): a short grounded passage for explanatory questions, blended with the extractive route's own sentences -- never the sole source of status/citations/figure_references | Apache-2.0 |
| `lab2-offline-wheels` (my own Kaggle dataset) | Offline pip wheels for the packages above, since the Kaggle kernel has no internet access | same as each wrapped package |

No hosted API is used. Seed is 266. Links to the Kaggle notebook, datasets and models are in the repo
root's `links.txt`.

## Notebooks

- `notebooks/p1_rag.ipynb`: the Kaggle notebook. It builds the full index, answers every question and
  writes `submission.csv`. All code is inside the notebook. **Its saved outputs are from Kaggle version
  35** (`manavpatel2025/lab2-part1-run`, `READER_A2=True`, `READER_A2_CHARTS=False`), public score
  **0.853** -- a single T4 (`CUDA_VISIBLE_DEVICES=0`, the reranker never targets a second
  GPU; `GPU count: 1`), 942.7s total, 11 of 40 rows changed vs. version 34's `submission.csv` (0.828, itself
  byte-identical to version 32's), all 11 the same single trigger class (an A2 answer) with unchanged
  status/citations/figure_references.
- `notebooks/p1_eval.ipynb`: runs the frozen pipeline (by executing `p1_rag.ipynb`'s own cells, not a copy
  of its code) and produces every number in the Results section above, plus the blank groundedness-rating
  sheets in `outputs/` -- each as a CSV and as a simple standalone HTML page
  (`p1_eval_groundedness_{manav,pramod}.html`) with a 1-5 dropdown per item and a "Download CSV" button,
  so each member can rate without editing a CSV by hand (same pattern as the Part 2/3 rating packs).
