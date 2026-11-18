# Part 2 facts: Diffusion Fine-Tuning

All facts below are read from the committed, executed `p2_finetune.ipynb` and `p2_generate_eval.ipynb`,
cell by cell. Each bullet names the cell it comes from. Note: no literal "C4" item exists in Part 2's
history (that labeling is Part 1's board-test convention) -- "safety analysis" below covers the NSFW
filter and memorization check, Part 2's two safety-relevant measurements.

## Architecture (7 steps)

- **1. Clean and split the dataset**: `reach-vb/pokemon-blip-captions`, 833 raw rows -> 0 corrupted, 0
  exact duplicates (pixel MD5), 8 near-duplicates (perceptual hash, Hamming <= 4) removed -> 825 images,
  90/10 split (seed 266) -> 743 train / 82 test, resized to 512x512. -- `p2_finetune.ipynb` cells 17, 19,
  21, 23.
- **2. Train two LoRA adapters**, rank 8 and rank 32, identical other settings (lr 1e-4, cosine, warmup
  220, snr_gamma 5.0, fp16, gradient checkpointing, 2200 steps). -- `p2_finetune.ipynb` cell 29.
- **3. Confirm the run fits the lab's GPU budget**: rank 8 -- 1,594,368 params, peak 4.44 GB (PASS),
  69.7 min, 1.16 GPU-hours; rank 32 -- 6,377,472 params, peak 4.52 GB (PASS), 74.6 min, 1.24 GPU-hours.
  -- `p2_finetune.ipynb` cell 42.
- **4. Generate 120 images**: base, rank 8, rank 32 x 20 fixed prompts (10 in_domain, 5 out_of_domain, 5
  compositional) x seeds 0 and 1, fixed sampler (PNDM, 30 steps, guidance 7.5, fp16). -- `p2_generate_eval.ipynb`
  cell 13, 22.
- **5. Filter for NSFW and exclude flagged images from metrics** before computing anything else, so no
  metric is corrupted by a black placeholder. -- `p2_generate_eval.ipynb` cell 27.
- **6. Score KID (vs. all 82 test images), CLIP (per prompt type), LPIPS (diversity between seeds), and a
  memorization check** (CLIP similarity of 10 in-domain generations per run against all 743 training
  images). -- `p2_generate_eval.ipynb` cells 29, 33, 41.
- **7. Apply the pre-committed final-run rule** and build a blind human rating pack for the chosen run vs.
  the base model. -- `p2_generate_eval.ipynb` cell 31; rater pack cells 35-37.

## Results: baseline (rank 8) vs final (rank 32), and the four measurement types

- **KID, without flagged** (used by the rule): rank 8 = 0.0642 +/- 0.0101 (n=16); rank 32 = 0.0349 +/-
  0.0074 (n=17); base = 0.0909 +/- 0.0137 (n=20). -- `p2_generate_eval.ipynb` cell 29.
- **Final-run rule**: lower in-domain KID wins if the difference exceeds the larger standard deviation.
  |diff| = 0.0293 > spread 0.0101 -> **rank 32 is FINAL**, resolved at step 1, CLIP tie-break never needed.
  -- `p2_generate_eval.ipynb` cell 31.
- **Four measurement types**:
  - Primary -- the pre-committed KID rule's own decision: **rank 32 final**, margin 0.0293 outside noise.
  - Secondary -- CLIP score without flagged (mean, all 20 prompts): rank 8 = 27.94, rank 32 = 28.23, base =
    29.24; LPIPS without flagged (seed 0 vs. seed 1): rank 8 = 0.7670, rank 32 = 0.7517, base = 0.8132. --
    `p2_generate_eval.ipynb` cells 29, 33.
  - Human -- blind win rate, rater_1 (manav) only: **0.40** (10/25 pairs preferred final over base) --
    `p2_generate_eval.ipynb` cell 39. rater_2 (pramod), outside rater 1: missing.
  - Efficiency -- training GPU-hours: rank 8 = 1.16, rank 32 = 1.24. -- `p2_finetune.ipynb` cell 42.

## What worked

- **fp16 + gradient checkpointing** kept both LoRA ranks under 4.52 GB peak memory on a single 16 GB T4,
  well inside the lab's GPU budget. -- `p2_finetune.ipynb` cell 42.
- **The pre-committed, metric-based final-run rule**: decided cleanly at step 1 (KID difference outside
  noise), with no need for the CLIP tie-break or the parameter-count fallback -- exactly the deterministic
  outcome the rule was designed to produce. -- `p2_generate_eval.ipynb` cell 31.
- **Excluding NSFW-flagged black placeholders from "without flagged" metrics** rather than regenerating or
  discarding them from the record: kept the published dataset's metadata honest while preventing 8 black
  images from corrupting KID/CLIP/LPIPS. -- `p2_generate_eval.ipynb` cell 27.

## What did not work

- **The quantitative winner was not the human-preferred one.** Rank 32 won the pre-committed KID rule
  decisively (margin outside noise), but the one rater in so far (manav) preferred the **base model** on
  15 of 25 blind pairs -- a 0.40 win rate for the final run, below chance. The metric that decided "final"
  and the one human rater's preference disagree; this needs rater_2 and outside rater 1 before it can be
  called a real pattern rather than one rater's taste. -- `p2_generate_eval.ipynb` cell 39;
  `part2_diffusion/member_manav/README.md`, Human evaluation section.
- **Caption and background bias limit what the model can learn**: captions are short (6-16 words) and
  colour-heavy (472 of 825 mention a colour word) but rarely mention the background (15 of 825 say "white
  background") even though most training images already have one -- the model has more signal for colour
  than for creature identity, and defaults to a white background even when not asked. --
  `p2_finetune.ipynb` cell 25.
- **Rank 32's own generations lean closest to specific training images**: all 3 of the top memorization
  pairs (CLIP similarity >= 0.90) are rank 32's, not rank 8's or base's. -- `p2_generate_eval.ipynb` cell 41.

## Safety analysis (NSFW filter and memorization check)

- **NSFW filter stays on** (the pipeline's default safety checker). **8 of 120** generated images were
  flagged; **all 8 confirmed pure-black placeholders**, not real content, via the safety checker's own
  standard behavior. Flags are kept in `metadata.jsonl` and the published dataset, never stripped; no
  flagged image was regenerated or swapped. -- `p2_generate_eval.ipynb` cell 27.
- All 8 flagged images are excluded from every "without flagged" KID/CLIP number, and the matching
  (model, prompt) pair is dropped entirely from LPIPS rather than comparing a real image to a black one --
  7 (model, prompt) pairs dropped for LPIPS. -- `p2_generate_eval.ipynb` cells 27, 33.
- **Memorization check**: CLIP embedding similarity between 10 in-domain generations per run and all 743
  training images. Three pairs reach similarity >= 0.90 (not pixel-identical, but high enough to flag):
  rank 32/P05 vs. train/0644.png = 0.9125, rank 32/P02 vs. train/0198.png = 0.9102, rank 32/P02 vs.
  train/0797.png = 0.9062. -- `p2_generate_eval.ipynb` cell 41.
- **License**: SD 1.5 is CreativeML OpenRAIL-M; the training data is CC BY-NC-SA 4.0 and shows copyrighted
  characters, so every model, adapter, and generated image here is for non-commercial research only. --
  `part2_diffusion/member_manav/README.md`, Safety and memorization section.
