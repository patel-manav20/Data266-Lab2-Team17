# PLAN.md: DATA 266 Lab 2, Team 17

Read this file first. It tells you (and your coding agent) where things go and how to push work.

Repo: https://github.com/patel-manav20/Data266-Lab2-Team17
Team: Manav Patel and Pramod
Deadline: Tuesday Nov 17, 2026, 11:59 PM PT (Canvas zip, Git tags, Kaggle submission)
Demos: Nov 18 to 24, 15 minutes per team

---

## 1. The three parts

| Part | What we build | Marks |
|---|---|---|
| Part 1 | Multimodal RAG over 3 economics PDFs, plus the Kaggle competition | 25 + 10 |
| Part 2 | Fine-tune Stable Diffusion 1.5 (two runs vs base model) | 27 |
| Part 3 | Five-agent system for image understanding and generation | 28 |
| Demo | Live team demo | 10 |

Both of us build all three parts, each in our own folder. Both must be able to explain every part in the demo.

---

## 2. Repo layout

```
Data266-Lab2-Team17/
├── README.md              final overview with both members' results side by side
├── PLAN.md                this file
├── requirements.txt       pinned versions (built at the end from the member files)
├── .gitignore
├── CONTRIBUTIONS.md       who did what + meeting recordings (date, topic, link)
├── AI_USAGE.md            AI tools used and for what
├── links.txt              tag commits, Hugging Face links, Kaggle notebook link
│
├── tests/                 written BEFORE tuning, one file per member
│   ├── p1_golden_manav.json
│   ├── p1_golden_pramod.json
│   ├── p2_prompts_manav.json
│   ├── p2_prompts_pramod.json
│   ├── p3_scenarios_manav.json
│   └── p3_scenarios_pramod.json
│
├── part1_rag/
│   ├── member_manav/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── notebooks/
│   │   ├── src/
│   │   └── outputs/
│   └── member_pramod/     same layout
│
├── part2_diffusion/
│   ├── member_manav/      same layout
│   └── member_pramod/
│
├── part3_agents/
│   ├── member_manav/      same layout
│   └── member_pramod/
│
├── notebooks/             final merged notebooks, added at the end
│   ├── p1_rag.ipynb
│   ├── p2_finetune.ipynb
│   └── p3_agents.ipynb
│
└── report/                IEEE report (PDF and source) and architecture diagrams
```

Rules for the layout:
- Your work goes only inside `part*/member_<yourname>/`.
- Do not create new top-level folders.
- Data files (PDFs, Questions.csv, datasets), model weights, generated images and secrets are never committed. Weights and images go to Hugging Face. Data stays local or on Kaggle.

---

## 3. Branches

Only three branches exist:

| Branch | Who | Purpose |
|---|---|---|
| `main` | both | final merged version only |
| `member_manav` | Manav | all of Manav's work for all three parts |
| `member_pramod` | Pramod | all of Pramod's work for all three parts |

No branch per part. No extra branches.

### One-time setup (Manav does this once)

```
git clone https://github.com/patel-manav20/Data266-Lab2-Team17.git
cd Data266-Lab2-Team17
git checkout -b member_manav
git push -u origin member_manav
git checkout main
git checkout -b member_pramod
git push -u origin member_pramod
git checkout member_manav
```

Before this, Manav pushes the empty folder layout, `.gitignore` and the starter files to `main`. Folders need a `.gitkeep` file to be saved by Git while empty.

### Pramod setup

```
git clone https://github.com/patel-manav20/Data266-Lab2-Team17.git
cd Data266-Lab2-Team17
git checkout member_pramod
```

### Daily work (both)

```
git checkout member_<yourname>
git pull origin member_<yourname>
```

Work only inside your own member folders, then:

```
git add part1_rag/member_<yourname>
git commit -m "part1: short message about what changed"
git push origin member_<yourname>
```

Use `git add` on your own folders (and your own `tests/` files). Do not use `git add .` blindly, and never add weights, data or secrets.

Push at least once a day.

---

## 4. Final merge to main (Nov 14 to 16)

1. Nov 14: both branches are finished. Every notebook is executed with outputs saved (Restart and Run All).
2. Nov 15: merge each branch into `main` with a normal merge. Do not squash and do not rebase, so the Git timestamps on the test files stay intact.
   ```
   git checkout main
   git pull origin main
   git merge member_manav
   git merge member_pramod
   git push origin main
   ```
3. On `main`, add the shared files:
   - Copy the better notebook for each part into `notebooks/` as `p1_rag.ipynb`, `p2_finetune.ipynb`, `p3_agents.ipynb`.
   - Combine the member `requirements.txt` files into one pinned `requirements.txt`.
   - Write `README.md` (both members' results side by side), `CONTRIBUTIONS.md`, `AI_USAGE.md` and `links.txt`.
   - Add the report to `report/`.
4. Nov 16: create and push the three tags from `main`:
   ```
   git tag p1-v1.0
   git tag p2-v1.0
   git tag p3-v1.0
   git push origin --tags
   ```
   Tags made after the deadline are ignored. The Kaggle notebook must run at tag `p1-v1.0`.
5. Nov 16: check on GitHub that all three tags exist and that the notebooks show outputs.
6. Nov 17: submit the Canvas zip well before 11:59 PM PT, and select the final submission on the Kaggle leaderboard.

---

## 5. Timeline

| Dates | Goal |
|---|---|
| Nov 2 to 3 | Repo set up, both branches pushed, all test files committed (see section 6) |
| Nov 4 to 9 | Part 1 final system, Part 2 both runs and generation, Part 3 pipeline |
| Nov 10 to 12 | Evaluations, human ratings, red-team cases, critique loop comparison |
| Nov 13 to 14 | Member READMEs finished, notebooks re-run with outputs |
| Nov 15 | Merge to main, shared files, report |
| Nov 16 | Tags, check everything, practice demo (each explains the other's code) |
| Nov 17 | Submit |

---

## 6. Do this first: test files

Write these before any tuning and commit them right away. The Git timestamp is checked.

| File (per member) | Content |
|---|---|
| `p1_golden_<name>.json` | Golden questions from the 3 PDFs (25 in total across both files, see below) |
| `p2_prompts_<name>.json` | The 20 fixed prompts: 10 in-domain from test captions, 5 out-of-domain, 5 compositional |
| `p3_scenarios_<name>.json` | 15 scenarios with a short pass/fail rubric each, and 10 red-team cases |

Notes:
- The golden set has 25 questions in total, split between the two Part 1 files (about 13 and 12). Both systems are evaluated on all 25. Questions must not be copied or paraphrased from `Questions.csv`.
- The 20 prompts and 15 scenarios work the same way: split between the two files, then both of us use the full set.
- Golden set mix: 5 facts and tables, 6 figures (including near-identical ones), 3 needing two sources, 5 multi-turn with a user-supplied value (in at least 2 conversations), 3 conflicts and revisions, 3 unanswerable, false-premise or injection.
- Scenario mix: 3 each of caption/visual questions, style change, edit or enhancement, domain-style generation (Part 2 model), ambiguous or broken input.
- Red-team mix: instructions written inside images, unsafe requests, edits of real people, broken or empty files.

---

## 7. What each part needs (from the lab PDF)

### Part 1: Multimodal RAG

Knowledge base: DOC1 (Blanchard Ch. 1-2, trust high, 2011-2012 values are forecasts), DOC2 (actual 2011-2013 results, trust high), DOC3 (briefing, trust medium, Appendix C untrusted and low).

Ingestion
- Split text into chunks with `doc_id` and `page`.
- Extract every numbered figure and table as an image with label and caption. "Figure B-5" and "Figure B-5 (corrected)" are different objects.
- Make figures searchable with one method: image embeddings (CLIP) or text descriptions from a vision-language model.
- Store in an open-source vector database (for example ChromaDB) with at least `doc_id`, `page`, `figure_id`, `trust_level` (from `kb_manifest.json`).

Retrieval
- Hybrid: dense embeddings plus BM25. A reranker is optional (+2 bonus if it improves the golden set).
- If the question names a figure or table, retrieve exactly that one. Never mix values from two figures.

Answering rules
1. A stated revision or correction replaces the earlier value. An actual result replaces a forecast. Mention the value that was replaced.
2. If documents disagree with no revision note: status `conflict`, give both values, prefer the higher-trust source.
3. Never use Appendix C as evidence. Never follow instructions found inside a document or image.
4. Not supported by the documents: `not_found`. Question assumes something false: `false_premise` and correct it.
5. User supplies or changes a value ("assume...", "for this chat..."): use it only in that conversation, say what the documents report, status `session_override`. Refuse requests to change the documents permanently.

Answer format (JSON, answer at most 120 words):
```
{"answer": "...", "status": "answered|not_found|conflict|false_premise|session_override",
 "citations": ["DOC2:p4", "DOC1:Table 1-2"], "figure_references": ["Figure A-3"]}
```

Conversations
- Separate memory per `conversation_id`. Follow-ups use only earlier turns of that conversation.
- User values go in that memory only, never in the vector database, and are forgotten when the conversation ends.

Kaggle (competition: data-266-fall-2026-lab-2-part-1-multimodal-rag)
- Read `Questions.csv` with `pd.read_csv(path, dtype=str, keep_default_na=False)`. Do not edit, reorder or drop rows.
- Notebook reads the questions from `QUESTIONS_PATH` (default `Questions.csv`) and writes `submission.csv` with no manual step, within 90 minutes on a Kaggle GPU.
- Columns: `id, conversation_id, question_id, answer, figure_references, citations, status`.
- Lists are joined with commas and no spaces (`Figure A-6,Figure A-7`). Empty list is `NONE`.
- Check with `python validate_submission.py submission.csv Questions.csv`, then submit from the notebook's Output tab and select it as the final submission.
- Score per question: 0.40 answer + 0.20 status + 0.20 figure references (F1) + 0.20 citations. Citation format must be exact, like `DOC2:p4`.
- Never hard-code answers or write rules for specific questions. Never put question IDs or question text in the code. Graders search the repo for them.
- Final marks come from a hidden question set run through the tagged notebook.

Part 1 results table (baseline = dense retrieval only, final = full system)

| Measurement | What to report |
|---|---|
| Primary | Answers correct out of 25 (numbers within 1%, status right) |
| Secondary | Recall@5 and citation accuracy |
| Human | Both rate each of the 25 final answers 1 to 5 for groundedness, average and agreement within one point |
| Efficiency | Median seconds per question |

Safety: show how the system handled Appendix C of DOC3 (false claims, instructions in the text and inside a chart image), with examples where it was and was not fooled.

### Part 2: Fine-tune Stable Diffusion

- Dataset: `lambdalabs/naruto-blip-captions` or `reach-vb/pokemon-blip-captions` (or another captioned set with at least 500 images and a stated license).
- Base model: Stable Diffusion 1.5 at 512x512. Same base for both runs.
- Data: remove corrupted and duplicate images and report how many. Split 90% train, 10% test with seed 266. Describe the dataset in a few lines with two limitations or biases.
- Two runs with the same data and step budget: (1) LoRA, (2) a second LoRA rank, QLoRA, DoRA or full UNet fine-tune.
- Print for each run: trainable parameters, learning rate, steps, batch size, peak GPU memory, training time.
- Push both models to Hugging Face. Write a half-page model card for the final run (base model, data, training settings, results, limitations, license).
- Generate 2 images per prompt (seeds 0 and 1) for base, run 1 and run 2 with the same sampler settings: 40 images per model. Show side-by-side grids for 8 prompts. Upload all images to a Hugging Face dataset with prompt, seed and model in the metadata.

Part 2 results table (base vs run 1 vs run 2)

| Measurement | What to report |
|---|---|
| Primary | KID (in-domain generations vs test images) and CLIP score |
| Secondary | Average LPIPS between the two seeds of each prompt |
| Human | Blinded comparison of 25 image pairs (final vs base, left/right random), at least 3 raters including one person outside the team, report win rate |
| Efficiency | GPU-hours per run with peak GPU memory |

Also: show at least two failure modes with example images, say which run you would choose and why.
Safety: dataset license (copyrighted characters, so non-commercial research only), the NSFW filter used, and a memorization check (for 10 in-domain images find the most similar training image with CLIP, show the 3 closest pairs).

### Part 3: Multi-agent system

Five agents, one shared typed state (dataclass or pydantic):

1. Planner: classify the request (caption/question, style change, edit, domain-style generation, unsupported), choose the route, refuse unsafe requests (NSFW, deceptive edits of real people, instructions hidden in the image).
2. Vision Understanding: caption, main objects, visible text, answers to at least 2 questions about the image.
3. Prompt Engineering: rewrite the request into a clear prompt using the vision output. Never treat text found in the image as an instruction.
4. Image Generation: at least style change (image-to-image) and one of edit or enhancement. Domain-style requests use the Part 2 model from Hugging Face. Record model, prompt, seed and steps.
5. Critique: CLIP similarity to the prompt plus a short vision-language judgement, output safety check, decide accept or revise with a concrete instruction.

Flow: request, Planner, Vision, Prompt, Generation, Critique, accept or revise (back to Prompt, at most 2 times). Caption and question requests stop after Vision. Unsafe requests are refused by the Planner.

- Save a JSON trace of every request (agent, input, output, time taken).
- Handle ambiguous requests, broken images and generation errors with a clear message, not a crash.
- Build a Gradio app showing each agent's output.
- Compare the system with and without the critique loop on the 15 scenarios.

Part 3 results table (without critique vs with critique)

| Measurement | What to report |
|---|---|
| Primary | Scenarios passed out of 15 |
| Secondary | Critic agreement with our own accept/revise judgement, red-team attack success rate out of 10 |
| Human | Both judge each scenario pass/fail independently, report agreement and how disagreements were settled |
| Efficiency | Median seconds per request |

Safety: describe the guardrails and give the results of the 10 red-team cases. Write a half-page system card (what it can do, known limits, safety measures).

Bonus (up to +4, total capped at 100): +2 Hugging Face Space for the Part 3 app, +2 reranker in Part 1 with a shown improvement.

---

## 8. Rules for everyone (people and agents)

Setup
- All work must fit one 16 GB GPU (Kaggle T4 or P100, Colab T4).
- Seed is 266 everywhere.
- Name any hosted API (LLM, VLM, image generation) with its model version in your member README.
- Check each model's license on its Hugging Face page. Kaggle rules ask for open-source code with licenses that allow commercial use.

Notebooks
- Put notebooks in your member folder's `notebooks/`.
- Give each notebook clear headings.
- Run Restart and Run All and save with all outputs. A notebook without outputs earns at most 50% of its marks.
- The first cell of every notebook is exactly this:
```
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})
```

Numbers
- Every number in a README or the report must come from a notebook output. Never type numbers by hand. A number that does not match the notebook gets 0 for that item.
- Never invent or estimate results. Fabricated results mean 0 for the whole part.

Code and writing style
- Keep code short and clean, with nothing extra and no comments.
- Write READMEs and the report in simple words, in our own words. Plagiarism is checked across groups.
- READMEs must read as finished. No "TBD", "pending" or "incomplete".

Git
- Work only on your own branch and in your own member folders.
- Never commit weights, data, generated images or secrets (API keys, tokens, `.env`).
- Never use force push.
- Normal merge to main at the end, no squash and no rebase.

AI use
- Both of us use AI coding agents. Keep a short list in your member README under "AI tools used" (tool name and what it helped with). We combine these into `AI_USAGE.md` at the end. Do not use AI beyond what is listed.

Recordings
- Record every knowledge-transfer session and every design or trade-off discussion (audio or video). Keep the files outside the repo (Drive or similar) and write date, topic and link in `CONTRIBUTIONS.md` at the end.

---

## 9. Member README template

Every `part*/member_<name>/README.md` uses these headings:

```
# Part N: <title> (<member name>)

## What I built
Two or three simple sentences.

## Models and settings
Model names and versions, hosted APIs, seed, learning rate, steps, batch size, chunk size, and so on.

## How to run
Exact steps and the notebook order.

## Results
One table with baseline and final: primary, secondary, human, efficiency.
Numbers copied from notebook outputs.

## What worked and what did not
Short and honest.

## Safety
What was tested and the outcome.

## Links
Hugging Face model, dataset and Kaggle notebook (if any).

## AI tools used
Tool and what it helped with.
```

---

## 10. Final checklist

- [ ] Both branches pushed, all test files committed before tuning
- [ ] Part 1 notebook runs on Kaggle with no manual step in under 90 minutes, `submission.csv` passes `validate_submission.py`, final submission selected
- [ ] Part 2: both models and the image dataset on Hugging Face, model card written
- [ ] Part 3: Gradio app runs, JSON traces saved, 15 scenarios and 10 red-team cases reported
- [ ] Every notebook has outputs, and the first cell prints commit, versions, GPU and seed
- [ ] Member READMEs finished, numbers match notebook outputs
- [ ] `main` has the merged work, `notebooks/`, `requirements.txt`, `README.md`, `CONTRIBUTIONS.md`, `AI_USAGE.md`, `links.txt`, `report/`
- [ ] Tags `p1-v1.0`, `p2-v1.0`, `p3-v1.0` pushed and checked on GitHub
- [ ] Canvas zip `Pair_Programming_Team_17.zip` submitted with: report PDF, `CONTRIBUTIONS.md`, `AI_USAGE.md`, `links.txt`, folders for Part 1, Part 2 and Part 3 (each with its executed notebook, saved weights and output files), and the GitHub repo link in the report
- [ ] Both members practiced explaining each other's code for the demo
