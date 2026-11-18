# DATA 266 Lab 2: Multimodal RAG, Diffusion Fine-Tuning and Multi-Agent System

**Team:** Pair_Programming_Team_17 -- Manav Patel and Pramodsatya Dindukurthi.

This lab has three parts. Part 1 answers questions about three economics PDFs on a Kaggle
leaderboard. Part 2 fine-tunes a diffusion model to generate images in a new style. Part 3 chains
five agents together to understand and generate images, with a critique step that can ask for a
revision. Both members built a complete, independent pipeline for all three parts.

## At a glance

| Part | Final system | Built by | Headline result | Source |
|---|---|---|---|---|
| 1 -- Multimodal RAG | Pramod's pipeline, Kaggle version 19 | Pramod (lead); Manav ran the full pipeline too | **0.868** public Kaggle score | `kaggle competitions submissions` |
| 2 -- Diffusion fine-tuning | Manav's rank-32 LoRA | Manav (lead); Pramod ran the full pipeline too | KID 0.0349 vs. base 0.0909 | `part2_diffusion/member_manav/README.md` |
| 3 -- Multi-agent system | Manav's pipeline | Manav (lead); Pramod ran the full pipeline too | human task success 11/15 | `part3_agents/member_manav/README.md` |

**Part 1 Kaggle top score, each member's own pipeline:** Pramod **0.868** (v19, commit `68fa972`,
also his best across all his tracked runs v6-v19) -- Manav **0.853** (v35, commit `310a1b5`, also his
best across all his tracked runs v2-v35). Verified from the real `kaggle competitions submissions`
history, not retyped from memory. The team's selected final submission is Pramod's v19.

## Quick links

- **Kaggle final notebook:** `<Pramod's Kaggle notebook link>` (full list in `links.txt`)
- **Hugging Face models and datasets:** listed in the Part 2 section below and in `links.txt`
- **GitHub tags:** `p1-v1.0`, `p2-v1.0`, `p3-v1.0` -- created after this document, once both members'
  branches are merged into `main`

## Rules we followed

- **One 16 GB GPU** for every required run -- a Kaggle T4 for Part 1, a Colab T4 for Parts 2 and 3.
- **No hosted API** anywhere -- every model runs locally on the GPU, for both members.
- **Seed 266** throughout, in every notebook that trains, splits data, or samples.
- **Notebooks executed with outputs**, Restart & Run All, before committing.
- **The first code cell of every notebook** prints the commit, package versions, GPU name, and the
  seed -- verified for all six notebooks (three parts, two members).

## Repository layout

```
Data266-Lab2-Team17/
├── README.md             (this file)
├── PLAN.md
├── requirements.txt
├── CONTRIBUTIONS.md
├── AI_USAGE.md
├── links.txt
├── eval/                 Manav's golden sets, test prompts, scenarios and red-team cases
├── tests/                Pramod's golden set, prompts and scenarios
├── part1_rag/member_manav/    and    part1_rag/member_pramod/
├── part2_diffusion/member_manav/    and    part2_diffusion/member_pramod/
├── part3_agents/member_manav/    and    part3_agents/member_pramod/
├── notebooks/
└── report/
```

---

## Part 1: Multimodal RAG

**The task:** answer questions about three economics PDFs (forecasts, actual results, and a
re-estimate with a low-trust appendix) using their text, tables and figures as a read-only knowledge
base, scored on Kaggle against a hidden question set.

**Team's final submission: Pramod's v19 (commit `68fa972`), public score 0.868.** This beat Manav's
best, v35 (0.853). Pramod's v19 code is the version copied into this repo's
`part1_rag/member_pramod/` folder (see the "Final-submission check" note below for exactly which
commit produced it and what its config shows).

### Pramod's pipeline

PyMuPDF ingestion (text, tables, figure crops); `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) transcribes
every trusted figure; a fact store with a plain-Python reconciliation engine (revisions, outturn over
forecast, source conflicts); ChromaDB (`BAAI/bge-base-en-v1.5`) + BM25 hybrid retrieval; answers from
`unsloth/Qwen3-8B-unsloth-bnb-4bit`, verified and status-assigned by code, not the model.

### Manav's pipeline

Ingestion into text chunks/tables/figure crops; a fact store with a resolver for the same kinds of
conflicting/revised/forecast values; hybrid dense (`BAAI/bge-base-en-v1.5`) + BM25 retrieval, reranked
with `Qwen/Qwen3-Reranker-4B`; mostly extractive answers, with a small rulebook-style generative
answerer (`Qwen/Qwen3-8B`, 4-bit) blended in for explanatory questions only. Best Kaggle score: v35,
0.853.

### Results table

| Measurement | Pramod (final submission) | Manav | Source |
|---|---|---|---|
| Kaggle public score | **0.868** (v19) | 0.853 (v35) | `kaggle competitions submissions` |
| Golden-set correct, of 25 | 22/25 (final) vs. 17/25 (baseline) | 15/25 (final = baseline, identical) | `part1_rag/member_pramod/outputs/p1_results_table.csv`; `p1_eval.ipynb` section (i) |
| Recall@5 | 1.000 | 0.92 | `outputs/p1_results_table.csv`; `p1_eval.ipynb` section (b) |
| Citation accuracy | 0.723 | 0.7867 | `outputs/p1_results_table.csv`; `p1_eval.ipynb` section (h) |
| Median seconds per question | 58.0 | 0.14 | `outputs/p1_results_table.csv`; `p1_eval.ipynb` section (g) (not directly comparable -- see note below) |
| Groundedness (human, 1-5) | **4.30**, both raters in (member1 4.16, member2 4.44) | 4.56, manav only; pramod and outside rater 1 missing | `outputs/p1_human_results.csv`; `outputs/p1_eval_groundedness_results.csv` |

Median-seconds numbers aren't directly comparable: Pramod's is from a 65-question run (40 competition
+ 25 golden, run twice); Manav's is from the real 40-question Kaggle submission run alone.

### Safety (Document 3's Appendix C, low-trust)

Pramod's pipeline marks Appendix C low-trust and produces no facts from it (pipeline step 3 of his
README). Manav's pipeline never crops, opens, or sends Appendix C to the chart-naming model (0 of 5
low-trust sentences with a number or instruction ever used in an answer), and a separate injection
probe found a planted instruction word in 0 of 36 answers (`p1_eval.ipynb` section (c)).

### How to run

Open either member's `notebooks/p1_rag.ipynb` as a new Kaggle notebook with GPU on and the
competition data attached, then Run All. Reads `QUESTIONS_PATH` (default `Questions.csv`), writes
`submission.csv`.

### Final-submission check

Commit `68fa972` (the one that produced v19) is on `member_pramod` and is an ancestor of `a0c956a`,
the commit this repo's `part1_rag/member_pramod/` folder was copied from -- so the code behind v19 is
in this repo. The currently-committed notebook is actually stamped with a later commit, `69e8de2`
("v20" in his own labeling); diffing the two, the only change between them is one `print()` statement
that stops logging question text to the notebook's saved output -- the answer-producing logic,
verified by diffing `src/p1_rag_cells.py` directly, is unchanged. His run-info cell sets
`CUDA_VISIBLE_DEVICES=0`, forcing a single GPU regardless of the Kaggle instance's accelerator
setting (confirmed `Tesla T4` by name in the saved output). The notebook's first cell installs
packages with `%pip install`, so the kernel's internet setting was on (unlike Manav's offline-wheels
approach); the real saved output confirms the competition data was attached at
`/kaggle/input/competitions/data-266-fall-2026-lab-2-part-1-multimodal-rag`.

---

## Part 2: Diffusion Fine-Tuning

**Data and split:** `reach-vb/pokemon-blip-captions`, corrupted/duplicate images removed, 90/10 split
with seed 266, resized to 512x512 -- both members. **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
for both.

**Manav:** two LoRA runs, rank 8 and rank 32, same steps and data; final: **rank 32**.
**Pramod:** two LoRA runs, rank 4 and rank 32, same steps and data; final: **rank 32** (an
independently trained adapter, not the same weights as Manav's).

**Team's final system: Manav's rank-32 LoRA** (model card marked final).

### Results table

| Measurement | Manav (final: rank 32) | Pramod (final: rank 32) | Source |
|---|---|---|---|
| KID, base / comparison run / final, without flagged | 0.0909 / 0.0642 / **0.0349** | 0.1018 / 0.0444 / **0.0306** | both members' README Results tables |
| CLIP score, base / comparison / final | 29.24 / 27.94 / 28.23 | 29.93 / 29.07 / 28.96 | both members' README Results tables |
| LPIPS (seed 0 vs 1), base / comparison / final | 0.8132 / 0.7670 / 0.7517 | 0.845 / 0.823 / 0.797 | both members' README Results tables |
| GPU-hours, comparison / final | 1.16 / 1.24 | 0.90 / 0.80 | `part2_diffusion/*/README.md` Training table |
| Peak GPU memory | 4.44 GB (rank 8) / 4.52 GB (rank 32) | **not reported** in his README or committed outputs | Manav: `README.md` Training table |
| Human win rate (final vs. base) | rater_1 (manav) only: **0.40** (10/25); rater_2 (pramod), outside rater 1 missing | **not collected** -- his own notebook prints `"human_ratings.csv not found yet"` | Manav: `README.md`; Pramod: `p2_02_generate_and_evaluate.ipynb` cell 21 |

### Human evaluation

Manav: one rater in (manav, 0.40 win rate), two missing. Pramod: none collected, though his README
describes the intended 3+-rater process including an outside rater.

### Failure modes

Manav: P16 compositional binding (only one of the two described creatures is recognisable); the NSFW
filter's false positives on in-domain prompts (8 of 120 flagged, 7 of those in-domain). Pramod's
README describes choosing two failure examples from the comparison grids, but none are named in his
committed Results section.

### Safety (license, NSFW filter, memorization)

Both: SD 1.5 is CreativeML OpenRAIL-M; the training data is CC-BY-NC-SA-4.0 with copyrighted
characters, so every model and generated image is for non-commercial research only. Both keep the
diffusers safety checker on (Manav: 8/120 flagged; Pramod: 4/120 flagged). Memorization: Manav flags 3
pairs at or above 0.90 cosine similarity as worth noting; Pramod reports a 0.83 maximum and calls it
"no near-copies."

### Weights and images

| | Manav | Pramod |
|---|---|---|
| Final model (rank 32) | https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r32 | https://huggingface.co/pramodsatyad/data266-p2-pokemon-lora_r32 |
| Comparison model | https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r8 | https://huggingface.co/pramodsatyad/data266-p2-pokemon-lora_r4 |
| Generated images | https://huggingface.co/datasets/itzmnv/data266-lab2-sd15-pokemon-generations | https://huggingface.co/datasets/pramodsatyad/data266-p2-generated-images |

All six links verified public (`curl` 200). Pramod's separate data-split dataset
(`data266-p2-pokemon-splits`) is private (401) -- not one of the links above, just noted for
completeness.

### How to run

On a Colab T4, open either member's training notebook and Run All, then their generation/evaluation
notebook. See each member's own README for exact file names and any required secrets (e.g. an
`HF_TOKEN` with write access).

---

## Part 3: Multimodal Multi-Agent System

Five agents, both members: **Planner** (routes the request, refuses unsafe ones) -> **Vision
Understanding** -> **Prompt Engineering** -> **Image Generation** -> **Critique** (accept or revise,
up to 2 times). Both use Stable Diffusion 1.5 for style-change, their own Part 2 rank-32 LoRA for
domain-style, and `timbrooks/instruct-pix2pix` for edits; Manav's VLM is `Qwen/Qwen3-VL-4B-Instruct`,
Pramod's is `Qwen/Qwen2.5-VL-7B-Instruct`.

**Team's final system: Manav's pipeline** (human-judged).

### Results table

| Measurement | Manav | Pramod (run 3, automated) | Source |
|---|---|---|---|
| Task success, without critique / with critique | human, single rater: **11/15** (critique always on for judging) | 11/15 / **12/15** | Manav: `p3_agents.ipynb` section 14; Pramod: `outputs/run3/p3_results_table.csv` |
| Critic agreement | judge_1 vs. system: **35%** (7/20 decision points); judge_2, outside rater 1 missing | **not collected** -- no human judgements file committed despite his README describing the process | Manav: `p3_agents.ipynb` section 14; Pramod: no `p3_human_judgements.csv` in his folder |
| Red-team attack success, of 10 | single rater: **0/10** | **0/10**, verified against his real `outputs/run3/redteam_results.csv` (all 10 rows `attack_succeeded=False`) | Manav: `outputs/judging/judge_1_redteam.csv`; Pramod: `outputs/run3/redteam_results.csv` |
| With vs. without critique (CLIP) | mean ON 0.318 vs. OFF 0.292, n=9 | S04 example only: 0.241 -> 0.311 after 2 revisions | Manav: `p3_agents.ipynb` cell 52; Pramod: README |
| Median seconds per request | **13.24** | without critique 20.7, with critique 23.6 | Manav: `p3_agents.ipynb` cell 50; Pramod: `outputs/run3/p3_results_table.csv` |
| Peak GPU memory | **13.38 GB**, whole run | not reported | Manav: `p3_agents.ipynb` cell 52 |

### Safety and red-team

Both refuse unsafe requests and real-person deception before any model call, and treat text found
inside an image as data, never an instruction. Manav: 5/10 refused by the Planner, 1/10 produced an
image, 3/10 flagged visible instruction-like text without a clean refusal (a documented partial safety
gap). Pramod: across three runs, 0/10 attacks succeeded; run 2's R03 case had a keyword-matching bug
(flagged a safe answer for containing the word "dog"), fixed by run 3 and verified against the real
output file, not just his README's claim.

### The Gradio demo

Both notebooks' last cell launches a Gradio app (Manav: non-blocking with a Playwright screenshot
saved to `outputs/app_screenshot.png`; Pramod: `share=True`, upload an image, type a request, toggle
the critique loop).

### How to run

On a Colab T4, open either member's `p3_agents.ipynb` and Run All, then run the last cell to launch
the Gradio app.

---

## Human evaluation status

| Part | Manav's rater(s) in | Missing | Pramod's rater(s) in | Missing |
|---|---|---|---|---|
| 1 -- groundedness | manav (mean 4.56, n=25) | pramod, outside rater 1 | **member1 and member2, both in** (mean 4.30) -- a valid, complete two-rater sheet | -- |
| 2 -- blind win rate | rater_1/manav (0.40, n=25) | rater_2 (pramod), outside rater 1 | none | all three |
| 3 -- judging | judge_1/manav (critic agreement 35%, task success 11/15) | judge_2 (pramod), outside rater 1 | none | all |

No audio or video recordings of design discussions exist for either member -- see
`CONTRIBUTIONS.md`.

## Reproducibility

Every notebook's first code cell prints the commit, package versions, GPU name, and seed 266. Every
required run fits on one 16 GB GPU. No hosted API is used by either member.

## AI usage

See `AI_USAGE.md` -- Manav's tools and what he checked himself per part, including the
instructor-approved reference-notebook note for the Part 1 rulebook answerer; Pramod's section is
drafted from his commits and files (marked to be confirmed by him).

## Contributions

See `CONTRIBUTIONS.md` -- who built what, by part and by member.

## Licenses

Stable Diffusion 1.5: CreativeML OpenRAIL-M. `reach-vb/pokemon-blip-captions`: CC-BY-NC-SA-4.0.
Qwen models: Apache-2.0. Full per-model license tables are in each part's own README.
