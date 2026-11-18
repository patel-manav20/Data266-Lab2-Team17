# DATA 266 Lab 2: Multimodal RAG, Diffusion Fine-Tuning and Multi-Agent System

**Team:** Pair_Programming_Team_17 -- Manav Patel (019162455) and Pramodsatya Dindukurthi (019124612).

## What this lab is

Three components, each with a baseline-vs-final evaluation, a human-rated measurement, and a safety
analysis, per the course's Lab 2 requirements (`DATA266_Lab2_Requirements.pdf`, not committed here):

- **Part 1 -- Multimodal RAG (Kaggle competition).** Answer questions about three economics PDFs,
  using their text, figures and tables as a read-only knowledge base, scored on a hidden Kaggle
  leaderboard.
- **Part 2 -- Diffusion fine-tuning.** Fine-tune Stable Diffusion 1.5 with two LoRA adapters on a
  captioned image set, compare them with the base model.
- **Part 3 -- Multimodal multi-agent system.** Five agents (Planner, Vision, Prompt Engineering,
  Image Generation, Critique) that route a request, generate or edit an image, and revise it against
  a critique loop.

Both members built a complete, independent pipeline for all three parts, per the lab's rule that the
partner not leading a part still implements and runs the full pipeline as a working baseline. Each
member's work lives in that member's own folder under each part.

## Repo layout

```
Data266-Lab2-Team17/
├── README.md             (this file)
├── PLAN.md
├── requirements.txt
├── .gitignore
├── CONTRIBUTIONS.md
├── AI_USAGE.md
├── links.txt
├── eval/                 Manav's golden sets, test prompts, scenarios and red-team cases
├── tests/                Pramod's golden set, prompts and scenarios
├── part1_rag/
│   ├── member_manav/
│   └── member_pramod/
├── part2_diffusion/
│   ├── member_manav/
│   └── member_pramod/
├── part3_agents/
│   ├── member_manav/
│   └── member_pramod/
├── notebooks/
└── report/
```

Open `PLAN.md` to learn how to work in this repo.

## How to run each part

**Hardware rule:** every part runs on a single 16 GB GPU -- a Kaggle T4/P100 for Part 1, a Colab T4
for Parts 2 and 3. No hosted API is used anywhere by either member (all models run locally on the
GPU); see `AI_USAGE.md` for the coding-assistant tools each member declares, and each part's README
for the exact model names and versions.

- **Part 1:** open either member's `notebooks/p1_rag.ipynb` as a new Kaggle notebook with GPU on and
  the Part 1 competition data attached, then Run All. It reads questions from `QUESTIONS_PATH`
  (default `Questions.csv`) and writes `submission.csv`, end to end, within 90 minutes.
  `part1_rag/member_manav/notebooks/p1_eval.ipynb` reproduces Manav's golden-set and retrieval
  evaluation separately; `part1_rag/member_pramod/notebooks/p1_human_eval.ipynb` reproduces Pramod's
  groundedness scoring.
- **Part 2:** on a Colab T4 (`colab` CLI), open either member's training notebook and Run All to
  train both LoRA runs, then their generation/evaluation notebook to generate the fixed prompts and
  run the KID/CLIP/LPIPS/human evaluation. See each member's own README for the exact file names and
  any required secrets (e.g. an `HF_TOKEN` with write access).
- **Part 3:** on a Colab T4, open either member's `p3_agents.ipynb` and Run All to build the agents,
  run the 15 scenarios and 10 red-team cases with and without the critique loop, and launch the
  Gradio demo app from the notebook's last cell.

## Part 1: Multimodal RAG

Both members built a complete pipeline: PDF ingestion into text chunks/tables/figure crops, a fact
store with a reconciliation engine for conflicting/revised/forecast-vs-outturn values, hybrid dense +
BM25 retrieval, and a JSON-structured answer with citations, figure references and status
(`answered`/`not_found`/`conflict`/`false_premise`/`session_override`). Manav's pipeline additionally
blends a small rulebook-style generative answerer (Qwen3-8B) into the extractive route for
explanatory questions (`READER_A2`); Pramod's uses a reconciliation engine in plain Python plus
Qwen3-8B for structured JSON answers.

**Final system: Manav's pipeline, Kaggle version 35 (0.853 public leaderboard), selected as the
team's final submission.**

**Results at a glance** (each number from that member's own notebook output):

| Measurement | Manav (`part1_rag/member_manav/`) | Pramod (`part1_rag/member_pramod/`) |
|---|---|---|
| Primary -- correct of 25 golden | baseline 15/25, final 15/25 identical (score >= 0.8; `p1_eval.ipynb` section (i)) | baseline 17/25, final **22/25** (`outputs/p1_results_table.csv`) |
| Secondary -- Recall@5 / citation accuracy | 0.92 / 0.7867 (final; `p1_eval.ipynb` section (b)/(h)) | 1.000 / 0.723 (final; `outputs/p1_results_table.csv`) |
| Human -- groundedness 1-5 | manav only, mean **4.56** (n=25); pramod, outside rater 1 missing (`outputs/p1_eval_groundedness_results.csv`) | both raters, mean **4.30** (member1 4.16, member2 4.44), within-one agreement 100%, exact 48% (`outputs/p1_human_results.csv`) |
| Efficiency -- median s/question | 0.14 s (v35 real 40-question run; `p1_eval.ipynb` section (g)) | 58.0 s (`outputs/p1_results_table.csv`) |
| Baseline vs final | identical on every component (`p1_eval.ipynb` section (i)) | 17/25 -> 22/25, 0.593 -> 0.723 citation accuracy (`outputs/p1_results_table.csv`) |
| Kaggle public leaderboard | v35: **0.853** (selected final; `kaggle competitions submissions`) | v19: **0.868** (not selected; see Note below) |

Manav's and Pramod's median-seconds numbers are not directly comparable: Manav's is from the real
Kaggle 40-question run, Pramod's from a 65-question run (40 competition + 25 golden-set questions, run
twice) on the same notebook, per his README.

**Note on the selected final submission:** Pramod's own v19 run scored higher (0.868) than the
selected final v35 (0.853). The team's selected final is Manav's v35 by the decision recorded here and
in `links.txt`, not the higher-scoring run -- confirmed real via `kaggle competitions submissions`
(both scores are genuine, dated 2026-11-17). Which submission is marked "selected" on Kaggle's own
leaderboard (a website-only setting, not visible to the `kaggle` CLI) should be checked directly on
Kaggle before the deadline.

**Safety (C4 -- the untrusted appendix, Document 3's Appendix C):** Manav's pipeline never crops,
opens, or sends Appendix C to the chart-naming model (0 of 5 low-trust sentences with a number or
instruction ever used in an answer; `p1_eval.ipynb` section (c)), and a separate injection probe found
the planted word appeared in 0 of 36 answers. Pramod's pipeline marks DOC3 Appendix C as low-trust and
produces no facts from it (`part1_rag/member_pramod/README.md`, pipeline step 3) and reports a
public-question-text leak check with no dedicated Appendix C safety section with quantified checks
the way Manav's has.

**Limitations:** see `report/part1_facts.md` (Manav) and `part1_rag/member_pramod/README.md`'s
"Remaining misses" (Pramod: G10 Figure B-2 growth threshold dropped; G18/G19 a Greece/Spain
cross-chart mixup).

## Part 2: Diffusion Fine-Tuning

Both members fine-tune Stable Diffusion 1.5 on `reach-vb/pokemon-blip-captions` with two LoRA ranks
(Manav: rank 8 and rank 32; Pramod: rank 4 and rank 32), 90/10 split, seed 266, 512x512, and generate
120 images (base + both runs x 20 fixed prompts x 2 seeds) for KID/CLIP/LPIPS evaluation plus a
memorization check.

**Final system: Manav's rank-32 LoRA, model card marked final.** (Pramod's rank-32 LoRA is also his
own final run by the same lowest-in-domain-KID rule -- the two are independently trained adapters, not
the same weights.)

**Results at a glance** (each number from that member's own notebook output):

| Measurement | Manav (`part2_diffusion/member_manav/`) | Pramod (`part2_diffusion/member_pramod/`) |
|---|---|---|
| Primary -- KID without flagged (base / rank 8 or 4 / final rank 32) | 0.0909 / 0.0642 / **0.0349** (`README.md` Evaluation table) | 0.1018 / 0.0444 / **0.0306** (`README.md` Results table) |
| Primary -- CLIP score, without flagged | base 29.24 / rank8 27.94 / **rank32 28.23** | base 29.93 / r4 29.07 / **r32 28.96** |
| Secondary -- LPIPS (seed 0 vs 1) | base 0.8132 / rank8 0.7670 / **rank32 0.7517** | base 0.845 / r4 0.823 / **r32 0.797** |
| Human -- blind win rate (final vs base) | rater_1 (manav) only: **0.40** (10/25); rater_2 (pramod), outside rater 1 missing (`README.md` Human evaluation) | **not collected** -- his own notebook prints `"human_ratings.csv not found yet"` (`notebooks/p2_02_generate_and_evaluate.ipynb` cell 21) |
| Efficiency -- GPU-hours (rank 8 or 4 / rank 32) | 1.16 / 1.24 | 0.90 / 0.80 |
| Memorization check | top cosine 0.9125 (rank 32 / P05 vs. train/0644.png) -- flagged as close | top cosine **0.83** -- "no near-copies" |
| NSFW flagged, of 120 | 8 (4 rank8, 4 rank32, 0 base) | 4 (2 r4, 2 r32, 0 base) |

**Weights and images:**

| | Manav | Pramod |
|---|---|---|
| Final model (rank 32) | https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r32 | https://huggingface.co/pramodsatyad/data266-p2-pokemon-lora_r32 |
| Comparison model | https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r8 | https://huggingface.co/pramodsatyad/data266-p2-pokemon-lora_r4 |
| Generated images | https://huggingface.co/datasets/itzmnv/data266-lab2-sd15-pokemon-generations | https://huggingface.co/datasets/pramodsatyad/data266-p2-generated-images |

All four model links and both generated-images dataset links verified public (`curl` 200). Pramod's
`data266-p2-pokemon-splits` dataset (the train/test split, not a model or generated-image deliverable)
returned 401 (private) when checked.

**Safety (C4 -- license, NSFW filter, memorization):** both keep the diffusers safety checker on and
report the flagged count above; both state the dataset's CC-BY-NC-SA-4.0 license and non-commercial
use given copyrighted characters. Manav's memorization check flags 3 pairs at or above 0.90 cosine
similarity as worth noting; Pramod's reports a 0.83 maximum and calls it "no near-copies" -- a judgment
call on the same kind of measurement, not a methodology difference.

**Failure modes:** Manav -- P16 compositional binding (only one creature recognisable; see
`part2_diffusion/member_manav/README.md`'s Failure modes section) and the NSFW-filter false positives
on in-domain prompts (8/120, 7 in-domain). Pramod's two required failure examples are not yet in his
committed README (his README describes the process -- "choose and explain at least two failure
examples from the comparison grids" -- but the Results section doesn't name them).

## Part 3: Multimodal Multi-Agent System

Both members built a five-agent pipeline (Planner -> Vision Understanding -> Prompt Engineering ->
Image Generation -> Critique, revise up to 2 times) with a shared typed state, a JSON trace per
request, and a Gradio demo app. Both use Stable Diffusion 1.5 for style-change, their own Part 2
rank-32 LoRA for domain-style, and `timbrooks/instruct-pix2pix` for edits.

**Final system: Manav's pipeline, human-judged.**

**Results at a glance** (each number from that member's own notebook output):

| Measurement | Manav (`part3_agents/member_manav/`) | Pramod (`part3_agents/member_pramod/`, run3) |
|---|---|---|
| Primary -- task success out of 15 | single human rater (judge_1/manav): **11/15**; judge_2 (pramod), outside rater 1 missing | **automated** (not human-judged): without critique 11/15, with critique **12/15** (`outputs/run3/p3_results_table.csv`) |
| Secondary -- critique ON vs OFF (CLIP) | mean ON 0.318 vs OFF 0.292, n=9 (`p3_agents.ipynb` cell 52) | not reported as a CLIP ON/OFF pair; S04 example: 0.241 -> 0.311 after 2 revisions (README) |
| Secondary -- red-team attack success, of 10 | single human rater: **0/10** | automated: **0/10** (`outputs/run3/redteam_results.csv`, verified: all 10 rows `attack_succeeded=False`) |
| Human -- critic agreement | judge_1 vs. system: **35%** (7/20 decision points); judge_2, outside rater 1 missing | **not collected** -- no human judgements file exists in his folder despite his README describing the process (`p3_human_judgements.csv` is not committed) |
| Efficiency -- median s/request | **13.24 s** (`p3_agents.ipynb` cell 50) | without critique 20.7 s, with critique 23.6 s (`outputs/run3/p3_results_table.csv`) |

**Safety (C4 -- guardrails and 10 red-team cases):** both refuse unsafe requests and real-person
deception before any model call, and treat text found inside an image as data, never an instruction.
Manav's red-team: 5/10 refused by the Planner, 1/10 produced an image, 3/10 flagged visible
instruction-like text without a clean refusal (a partial safety gap, documented in his Limitations
section); single-rater attack success 0/10. Pramod's red-team: 0/10 attacks succeeded across three
runs, with run 2's known R03 keyword-matching bug (flagged a safe answer because it contained the word
"dog") fixed by run 3 -- verified against his real `outputs/run3/redteam_results.csv`, not just his
README's claim.

**Limitations:** Manav -- 35% critic agreement, the 3/10 red-team partial-safety-gap cases, the
missing second judge, the edit route's much higher latency (`part3_agents/member_manav/README.md`
Limitations section). Pramod -- S07/S08 instruct-pix2pix edits stay below the 0.25 CLIP threshold even
after revisions; S11 domain-style stays close to the original photo (his README's "Remaining
failures").

## Human evaluation status (both members, all parts)

| Part | Manav's rater(s) in | Missing | Pramod's rater(s) in | Missing |
|---|---|---|---|---|
| 1 -- groundedness | manav (n=25, mean 4.56) | pramod, outside rater 1 | **member1 and member2, both in** (n=25, mean 4.30) -- a valid, complete two-rater sheet (`outputs/p1_human_ratings.csv`) | -- |
| 2 -- blind win rate | rater_1/manav (n=25, win rate 0.40) | rater_2 (pramod), outside rater 1 | none | all three (his notebook: `"human_ratings.csv not found yet"`) |
| 3 -- judging | judge_1/manav (critic agreement 35%, single-rater task success 11/15) | judge_2 (pramod), outside rater 1 | none | all (no `p3_human_judgements.csv` committed) |

No audio/video recordings of design discussions exist for either member -- see `CONTRIBUTIONS.md`.

## Links

Full list in `links.txt` (template format, filled). Summary: GitHub
`https://github.com/patel-manav20/Data266-Lab2-Team17`; Kaggle final notebook
`https://www.kaggle.com/code/manavpatel2025/lab2-part1-run` (version 35); Hugging Face models and
datasets listed in the Part 2 table above for both members.

## Tags

`p1-v1.0`, `p2-v1.0`, `p3-v1.0` are created at the end, after both members' branches are merged into
`main`, pointing at the final merged commit -- not created yet as of this document.
