# CONTEXT.md: handover notes for the next session or coding agent

Last updated: Nov 15, 2026 (Pramod's branch `member_pramod`). Read this together with `PLAN.md` (the team plan) and the lab PDF on Canvas.

## 1. The deadline and what still counts

- Final submission is **Tue Nov 17, 2026, 11:59 PM PT**: Canvas zip, Git tags `p1-v1.0`, `p2-v1.0` and `p3-v1.0`, and the final Kaggle submission selected. The Kaggle deadline is Nov 18 07:59 UTC.
- Demos run Nov 18–24. Both members must be able to explain all code.
- Hard rules:
  - Every number in the report must come from a notebook output.
  - Never hard-code answers or write question-specific rules for Part 1. Graders search the repo for question IDs and text.
  - Test sets must be committed before tuning. They already are (see section 4).
  - Executed notebooks must keep their outputs.
  - Everything must run on one 16 GB GPU.
  - No weights, data or secrets in Git.

## 2. Blockers that need people, not code

1. **GitHub push is blocked (403).** Pramod's GitHub account `pramodsatyad` is not a collaborator on `patel-manav20/Data266-Lab2-Team17`. Manav must add him: Settings → Collaborators. Until then every commit on `member_pramod` exists **only locally** on Pramod's Mac. Push with `git push origin member_pramod` as soon as access exists.
2. **Kaggle team.** Team `Pair_Programming_Team_17` (rank 22, public score 0.828 as of Nov 15) currently has only Manav (`manavpatel2025`). Pramod's account (`pramodsatyad`) must join before anything is submitted from it. **Do not submit to the leaderboard from Pramod's account until he is on the team.**
3. **Manav's Part 1 code** (which scored 0.828) is not on GitHub. Both branches were empty when this work started.
4. **Kaggle runs one GPU session at a time** on Pramod's account. A second queued GPU notebook blocks the first.

## 3. Accounts, tools and where secrets live (never print or commit their values)

- Kaggle CLI token: `~/.kaggle/access_token` (user `pramodsatyad`). The CLI is installed in a throwaway venv. Recreate it with:
  `python3 -m venv /tmp/kvenv && /tmp/kvenv/bin/pip install kaggle six`
- Hugging Face write token: `~/.cache/huggingface/token` (user `pramodsatyad`). `huggingface_hub` picks it up automatically.
- Competition data has been downloaded to `data/p1/` (git-ignored):
  `kaggle competitions download data-266-fall-2026-lab-2-part-1-multimodal-rag -p data/p1`
- Kaggle secrets cannot be attached to notebooks pushed by the CLI. For that reason the notebooks work without `HF_TOKEN` and `GITHUB_TOKEN`, and the commit hash is stamped into the pushed copy instead (see section 6).
- GPU plan:
  - Kaggle is used for Part 1. Its final run must be on Kaggle.
  - Colab (Pay As You Go, T4 only) is used for Part 2 training and other interactive work.
  - RunPod was considered and dropped.

## 4. Test sets (committed before tuning; do not edit the questions)

| File | Content |
|---|---|
| `tests/p1_golden_pramod.json` | 25 golden questions: 5 facts/tables, 6 figures, 3 two-source, 5 multi-turn override (2 conversations), 3 conflict/revision, 3 not_found/false_premise/injection. Each has `required_facts`, `status` and `citations`. None is copied from Questions.csv (checked by string similarity). |
| `tests/p2_prompts_pramod.json` | 20 fixed prompts: 10 in-domain (test-set captions copied exactly), 5 out-of-domain, 5 compositional. Seeds 0 and 1. |
| `tests/p3_scenarios_pramod.json` | Recipes for the test images, 15 scenarios with rubrics, and 10 red-team cases |

Judgement call: G17 (refusing a permanent knowledge-base change) expects status `session_override`.

## 5. Decisions already made (do not re-open without a reason)

- **Part 1 models:** open models only, so graders can re-run with no API keys.
  - Qwen2.5-VL-7B-Instruct (4-bit NF4) transcribes every figure into text.
  - `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Apache-2.0, run without thinking mode; 14B ran out of memory on the T4) writes the answers.
  - `BAAI/bge-base-en-v1.5` produces the embeddings.
  - ChromaDB, BM25 (`rank-bm25`) and PyMuPDF are used for storage, keyword search and ingestion.
- **Part 1 retrieval:**
  - Dense and BM25 results are fused with RRF (k=60).
  - Every table row is indexed as its own item and maps back to its whole table.
  - Slots are balanced: the top 4 tables/figures plus the top 4 text chunks.
  - Exact label lookup: any figure or table named in the question (or earlier in the conversation) is pinned, plus its "(corrected)" version if one exists.
  - Appendix C (DOC3 p7, low trust) is stored but filtered out of retrieval.
- **Part 1 citations:**
  - A figure is cited by label and page (`DOC2:Figure A-3,DOC2:p4`), a table by label only, and text by page.
  - If the question names a figure, only that figure (or its corrected version) is referenced, and the corrected version replaces the base one.
  - Labels are checked against the labels that actually exist (`KNOWN_LABELS`).
- **Part 1 status decision order in the prompt:** session_override → false_premise → not_found → conflict → answered.
- **Part 2:** Pokémon BLIP captions (`reach-vb/pokemon-blip-captions`, CC-BY-NC-SA-4.0, 833 images).
  - Data: 1 near-duplicate removed; 749 train / 83 test with seed 266.
  - Base model: SD 1.5. Run 1 is LoRA rank 4; run 2 is LoRA rank 32. Both use 1500 steps, batch 1 with gradient accumulation 4, learning rate 1e-4.
  - Training uses the diffusers example script, downloaded at the same version as the installed diffusers.
- **Part 3:**
  - Plain Python functions sharing one `State` dataclass.
  - Qwen2.5-VL-7B handles the planner, vision, prompt-engineering and critic roles.
  - SD 1.5 img2img does style changes; the same pipeline plus our Part 2 LoRA does domain-style requests; `timbrooks/instruct-pix2pix` does edits.
  - The critic uses CLIP ViT-L/14 similarity; the Stable Diffusion safety checker is the NSFW filter.
  - Domain-style prompts describe *original* creatures, never existing named characters.
- **Packages:** pinned old versions broke on Kaggle (Python 3.13, torch 2.11). The notebooks now use the preinstalled torch and transformers, install only the missing packages, and print every version. Build the final `requirements.txt` from those printed versions.

## 6. How the notebooks are built and run

- **Source of truth:** the builder scripts under `part*/member_pramod/src/build_*_notebook.py`. They write the `.ipynb` files. Edit the builder and rebuild, rather than hand-editing the JSON:
  - `python part1_rag/member_pramod/src/build_p1_notebook.py part1_rag/member_pramod/notebooks/p1_rag.ipynb tests/p1_golden_pramod.json`
  - `python part2_diffusion/member_pramod/src/build_p2_01_notebook.py part2_diffusion/member_pramod/notebooks/p2_01_data_and_training.ipynb`
  - `python part2_diffusion/member_pramod/src/build_p2_02_notebook.py part2_diffusion/member_pramod/notebooks/p2_02_generate_and_evaluate.ipynb tests/p2_prompts_pramod.json part2_diffusion/member_pramod/notebooks/p2_01_data_and_training.ipynb`
  - `python part3_agents/member_pramod/src/build_p3_notebook.py part3_agents/member_pramod/notebooks/p3_agents.ipynb tests/p3_scenarios_pramod.json part2_diffusion/member_pramod/notebooks/p2_01_data_and_training.ipynb`
  - Each needs `nbformat`; the Anaconda Python at `/opt/anaconda3/bin/python` has it.
- **The first code cell:**
  - Detects Kaggle or Colab and clones the repo if `GITHUB_TOKEN` exists.
  - Prints the required run info (commit, versions, GPU, seed 266).
  - Falls back to `STAMP` for the commit. Before a CLI push, replace `STAMP = ""` with the short hash of the current commit, in the pushed copy only:
    `sed 's/STAMP = \\"\\"/STAMP = \\"<hash>\\"/' notebook.ipynb > kernel_dir/notebook.ipynb`
- **Pushing to Kaggle:** use a folder holding the notebook plus `kernel-metadata.json`.
  - Part 1 metadata: `id pramodsatyad/data266-p1-rag`, `code_file p1_rag.ipynb`, `kernel_type notebook`, private, GPU and internet on, `machine_shape NvidiaTeslaT4`, `competition_sources ["data-266-fall-2026-lab-2-part-1-multimodal-rag"]`, `dataset_sources ["pramodsatyad/data266-p1-figcache"]`.
  - Push: `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`
  - Check: `kaggle kernels status pramodsatyad/data266-p1-rag`
  - Download results: `kaggle kernels output pramodsatyad/data266-p1-rag -p <dir>`. This gives the log plus `submission.csv` and `p1_out/`.
  - Live log: `kaggle kernels logs -f pramodsatyad/data266-p1-rag`
- **Development shortcut:** `USE_FIG_CACHE = False` in the repo. Setting it to `True` in a pushed copy reuses `fig_cache.json` from the private Kaggle dataset `pramodsatyad/data266-p1-figcache` and skips about 12 minutes.
  - The cache was refreshed from v4 with full transcriptions (see section 7).
  - Final and graded runs must use `False`.
- **Testing locally first.** Kaggle runs cost 25–50 minutes, so test locally before pushing:
  - Run the CPU parts locally: settings, ingestion, objects, index, retrieval, submission writing and the validator.
  - Use cached figure text and a stub `llm_call`.
  - Needs `pymupdf`, `chromadb`, `rank-bm25`, `sentence-transformers` and `torch`.

## 7. Status of each part

### Part 1 (notebook `part1_rag/member_pramod/notebooks/p1_rag.ipynb`)

**Kaggle run history:**

| Version | Result |
|---|---|
| v1 | Package conflict |
| v2 | Chroma collection name too short (fixed) |
| v3 | Ran end to end: 4.4 minutes for 40 questions, 6.7 s median, validator passed. About 15 answers were wrong (weak 7B answer model, truncated figure cache). |
| v4 (commit `9c770c2`) | Full fresh figure transcription (585 s, all complete and accurate), then GPU out-of-memory on question 7. Qwen3-14B uses 11.6 GB, so long evidence prompts do not fit. |
| v5 (commit `ae950b6`) | Pushed with `USE_FIG_CACHE=True`: compacted transcriptions ("2012=11.4"), evidence capped at 2500 chars per item and 12000 in total, retry on out-of-memory with 5 shorter items, cache freed after each call |

| v6 (commit `cf1b13f`) | **First complete run** with Qwen3-8B: 22.6 min for 40 questions (33.7 s median), validator passed. Golden set: baseline 14/25 correct (Recall@5 0.958, citation accuracy 0.692); final 11/25 (Recall@5 1.000, citation accuracy 0.925). Most public figure questions are now right. Main failures: answers leave out the numbers they rely on, columns misread in unlabelled tables, forecast vs outturn wrongly called a conflict, missed DOC1/DOC3 conflicts. |
| v7 (commit `5131c49`) | Tables rendered as `row: column=value` (year labels used only when the years run in order; Table A-2's bracketed value is labelled "fall-2011 forecast"); prompt rules: forecast vs outturn is not a conflict, DOC3 vs DOC1/DOC2 is a conflict, check the premise first, write per-document values in "work"; numbers in "work" that are missing from the answer are added as "Working: ...". Pushed. |

The figure-cache dataset `pramodsatyad/data266-p1-figcache` now holds the **full, untruncated** v4 transcriptions, so it is safe for development runs.

**Next steps:**
1. Download v4's output and read all 40 answers against the documents.
2. Check the golden results table: baseline vs final correctness out of 25, Recall@5, citation accuracy, median seconds.
3. Fix general weaknesses only, never question-specific rules.
4. Before the final run, set `USE_FIG_CACHE = False` (the repo default) so graders get a full from-scratch run.

**Known facts in the documents** that the hidden questions will probe:
- **Revision:** U.S. 2010 output growth is 3.0, revised to 2.4 (DOC2 p3 revision notice).
- **Outturns replacing forecasts:** DOC1's 2011–2012 values are forecasts, replaced by DOC2 outturns. Example: Euro-area unemployment 2012 was forecast at 9.9; the outturn is 11.4.
- **Conflicts:** U.S. unemployment 2010 (DOC1 9.6 vs DOC3 9.8); China growth 2012 (DOC2 7.7 vs DOC3 7.8).
- **Corrected figure:** Figure B-5 (corrected) shows China's 2012 rate as 4.1; the original showed 4.5.
- **Appendix C:**
  - A false Lehman date (the true date, Sept 15, 2008, is on DOC1 p1).
  - A false claim that China's output contracted in 2009 (DOC1 Table 1-4 shows growth of 9.2%).
  - Injected instructions in the text and inside the Figure C-1 image.

**Still to do:**
- Final Kaggle notebook link and the team's selected submission.
- Share the notebook with the TAs.
- Human groundedness ratings: both members fill `p1_groundedness_sheet.csv` (columns `member1`, `member2`), save it as `p1_human_ratings.csv`, and attach it.
- Optional reranker bonus.

### Part 2 (`part2_diffusion/member_pramod/notebooks/`)

- `p2_01_data_and_training.ipynb`:
  - Data cleaning and split, both LoRA runs, the training table, and adapters saved to `adapters/`, with Hugging Face upload if `HF_TOKEN` is set.
  - On Colab everything goes to Google Drive under `MyDrive/data266_p2/` (runs, adapters, `p2_data`, outputs), and training resumes from checkpoints.
  - **Not yet run on a GPU.** Pramod was asked to run it on Colab (T4). The earlier Kaggle training notebook was deleted, because it blocked the single GPU slot.
- `p2_02_generate_and_evaluate.ipynb`:
  - 3 models × 20 prompts × 2 seeds, an 8-prompt grid, KID (in-domain vs test, subset 20), CLIP score (ViT-L/14), LPIPS diversity, NSFW counts, failure modes and the memorization top-3.
  - Also writes 25 blinded pairs plus a key for human rating (at least 3 raters, one from outside the team; ratings go in `human_ratings.csv`) and uploads the images to a Hugging Face dataset.
  - `FINAL` is the run with the lowest in-domain KID.
  - **Not yet run.**
- Still to do after the runs:
  - Upload both adapters to Hugging Face (`pramodsatyad/data266-p2-pokemon-lora_r4` and `..._r32`).
  - Write the model card: base model, data, settings, results, limitations, non-commercial licence.

### Part 3 (`part3_agents/member_pramod/notebooks/p3_agents.ipynb`)

- Built but **not yet run.**
  - Loads the LoRA from Hugging Face if the `LORA_REPO` environment variable is set; otherwise from attached Kaggle input.
  - Contains the 5 agents, the revise loop (at most 2 revisions), JSON traces and the error handling.
  - Builds the test images from their recipes, runs the 15 scenarios with and without the critique loop and the 10 red-team cases, produces the results table, and builds the Gradio app (`share=True`).
- Needs the Part 2 adapter first.
- Human pass/fail judgements go in `p3_human_judgements.csv` with columns `sid, critique, member1, member2, critic_label`.
- Optional bonus: deploy the app as a Hugging Face Space.

## 8. Final assembly (from PLAN.md)

1. Merge into `main` with normal merges (no squash or rebase).
2. Copy the best notebooks to `notebooks/p1_rag.ipynb`, `p2_finetune.ipynb` and `p3_agents.ipynb`.
3. Write the pinned `requirements.txt`, `README.md`, `CONTRIBUTIONS.md` (with recording links), `AI_USAGE.md` (list Claude Code and what it did) and `links.txt`.
4. Write the IEEE report (at most 12 pages per part) with diagrams, results tables, safety analyses, the model card and the system card.
5. Create and push the tags.
6. Submit the Canvas zip `Pair_Programming_Team_17.zip`.
