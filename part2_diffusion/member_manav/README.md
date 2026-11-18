# Part 2: Diffusion Fine-Tuning (Manav)

Fine-tunes Stable Diffusion 1.5 on the Pokemon BLIP captions dataset with two LoRA adapters (rank 8 and
rank 32, same data, same step budget), then generates and evaluates images from the base model and both
adapters to pick a final run. All required work runs on a single 16 GB GPU (Colab T4) in fp16.

## How to run

1. `uv tool install google-colab-cli` (or `pip install google-colab-cli`), then authenticate once.
2. `colab new -s <session> --gpu T4 --high-mem`
3. `colab drivemount -s <session>` (needs a one-time browser approval the first time).
4. Copy a Hugging Face token file onto the VM -- never through a notebook cell, and never read, printed
   or logged: `colab upload -s <session> ~/.cache/huggingface/token /root/.cache/huggingface/token`
5. Upload and run the two notebooks, in order, with papermill:
   - `colab upload -s <session> notebooks/p2_finetune.ipynb /content/p2_finetune.ipynb`, then
     `papermill /content/p2_finetune.ipynb /content/p2_finetune_out.ipynb -k python3` -- data prep, both
     LoRA runs (detached subprocess, resumes from the latest checkpoint if interrupted), the results
     table, loss curves, and the push of both adapters to Hugging Face.
   - `colab upload -s <session> notebooks/p2_generate_eval.ipynb /content/p2_generate_eval.ipynb`, then
     `papermill /content/p2_generate_eval.ipynb /content/p2_generate_eval_out.ipynb -k python3` -- 120
     generations, KID/CLIP/LPIPS/memorization, the final-run rule, the rater pack, and the dataset push.
6. `colab stop -s <session>` when done; check `colab usage` before and after.

Weights, generated images and the rater pack go to Hugging Face and Drive, not to GitHub. Only the rater
pack's HTML pages and a blank rating sheet are committed here (`outputs/rating/`) -- no answer key, no
filled ratings.

## Data

`reach-vb/pokemon-blip-captions`, 833 raw rows. Cleaning removed 0 corrupted images, 0 exact duplicates
(pixel MD5), and 8 near-duplicates (perceptual hash, Hamming distance <= 4), leaving **825** images, split
90/10 (seed 266) into **743 train / 82 test**, resized to 512x512.

Two limitations carried into training and the model cards:
- **Caption style bias**: captions are short (6-16 words) and generic -- the most common caption openers
  are "a drawing of" (312), "a cartoon character" (89) and "a very cute" (51); 472 of 825 captions mention
  a colour word, so the model has more signal to latch onto colour than onto creature identity or pose.
- **Background bias**: "white background" appears in only 15 captions, but the great majority of training
  images are already on a plain white background, so generations default to one even when not asked for.

## Training

Both LoRA runs use identical settings (seed 266, batch 1, gradient accumulation 4, lr 1e-4, cosine
schedule, snr_gamma 5.0, fp16, gradient checkpointing, 2200 steps, checkpoint every 500 steps) and differ
only in rank.

| rank | trainable params | learning rate | steps | batch size | grad. accum. | effective batch | peak GPU mem | wall-clock | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 8  | 1,594,368 | 1e-4 | 2200 | 1 | 4 | 4 | 4.44 GB (PASS) | 69.7 min | 1.16 |
| 32 | 6,377,472 | 1e-4 | 2200 | 1 | 4 | 4 | 4.52 GB (PASS) | 74.6 min | 1.24 |

Trainable parameter counts are read from the saved LoRA weights, not estimated.

**Loss curve:** both ranks drop from an initial loss of roughly 0.05-0.06 to a noisy plateau around
0.035-0.04 within the first 100-150 steps and stay flat there for the remaining ~2050 steps; rank 8 and
rank 32 are essentially indistinguishable on the smoothed curve. The loss flattening this early says
little about image quality on its own, so the final run is chosen by the evaluation metrics below, not
by the training loss.

## Evaluation

120 images generated (base, rank 8, rank 32 x 20 fixed prompts x seeds 0 and 1), fixed sampler settings
(PNDM scheduler, 30 steps, guidance 7.5, 512x512, fp16) for every model. KID is computed vs. **all 82 test
images** (100 subsets, seed 266), with a single shared `subset_size` across all three models in each pass
-- capped at 20, and shrunk to the smallest generation count any one model has available that pass, so the
three KID numbers are always compared on the same subset size (20 for the "all images" row below, 16 for
"without flagged", since rank 8 has only 16 non-flagged in-domain generations); every number below except
the "all images" KID/CLIP columns excludes images the safety checker returned as a black placeholder (see
Safety below) -- nothing is ever regenerated or replaced, the flag just isn't allowed to corrupt a metric.

| metric | rank 8 | rank 32 | base |
|---|---|---|---|
| KID, without flagged (used by the rule) | 0.0642 +/- 0.0101 (n=16) | 0.0349 +/- 0.0074 (n=17) | 0.0909 +/- 0.0137 (n=20) |
| KID, all images | 0.0896 +/- 0.0084 | 0.0472 +/- 0.0063 | 0.0909 +/- 0.0091 |
| CLIP score, without flagged (mean, all 20 prompts) | 27.94 | 28.23 | 29.24 |
| LPIPS, without flagged pairs (seed 0 vs. seed 1, mean) | 0.7670 | 0.7517 | 0.8132 |
| GPU-hours (training) | 1.16 | 1.24 | -- |

CLIP score by prompt type (mean, without flagged):

| type | rank 8 | rank 32 | base |
|---|---|---|---|
| in_domain | 26.66 | 26.91 | 28.05 |
| out_of_domain | 29.25 | 28.60 | 31.07 |
| compositional | 28.66 | 30.29 | 29.81 |

Lower LPIPS between the two seeds means less diversity (more similar images for different seeds). With the
flagged black placeholders dropped, rank 32 is actually the *least* diverse of the three here -- the
opposite of what the uncorrected numbers suggested, because 3 of its dropped prompts happened to be
unusually high-LPIPS ones.

### Final run

The rule was fixed before any image was looked at: the run with the lower in-domain KID is final; if the
two KID means differ by less than the larger of their standard deviations, the run with the higher mean
CLIP score over all 20 prompts is final instead; if that CLIP difference is also under 0.005, rank 8 is
chosen (fewer parameters). The rule uses the "without flagged" numbers above.

**Result: rank 32 is FINAL.** KID differs by 0.0293 (0.0642 vs. 0.0349), which is outside the larger
standard deviation (0.0101), so the rule resolves at step 1 without needing CLIP or the parameter
tie-break -- the same winner as before the NSFW correction, just with a smaller (and now non-trivial,
since KID is computed against all 82 test images instead of 20) margin. Rank 8 is the comparison run.
Both model cards on Hugging Face state this.

### Safety and memorization

- **License:** SD 1.5 is CreativeML OpenRAIL-M; the training data is CC BY-NC-SA 4.0 and shows copyrighted
  characters, so the models, adapters, and every generated image here are for **non-commercial research
  only**.
- **NSFW filter:** kept on (the pipeline's own default safety checker). **8 of 120** generated images were
  flagged, and **all 8 were confirmed to be pure-black placeholders** (the safety checker's standard
  behavior), not real content -- so they are excluded from every "without flagged" number above, and the
  (model, prompt) pair is dropped entirely from LPIPS rather than comparing a real image to a black one.
  The flags themselves are never removed from `metadata.jsonl` or the published dataset, and no flagged
  image was ever regenerated or swapped out.

  | model | prompt | type | seed |
  |---|---|---|---|
  | rank 8 | P04 | in_domain | 0 |
  | rank 8 | P04 | in_domain | 1 |
  | rank 8 | P05 | in_domain | 1 |
  | rank 8 | P07 | in_domain | 1 |
  | rank 32 | P04 | in_domain | 0 |
  | rank 32 | P09 | in_domain | 1 |
  | rank 32 | P10 | in_domain | 0 |
  | rank 32 | P17 | compositional | 1 |

- **Memorization check:** CLIP image-embedding similarity between 10 in-domain generations per run and
  every one of the 743 training images. The 3 closest pairs overall (shown side by side with their closest
  training image in the notebook):

  | model | prompt | closest train image | similarity |
  |---|---|---|---|
  | rank 32 | P05 | train/0644.png | 0.9125 |
  | rank 32 | P02 | train/0198.png | 0.9102 |
  | rank 32 | P02 | train/0797.png | 0.9062 |

  These are CLIP embedding similarities, not pixel-identical matches, but 0.91 is high enough to flag:
  rank 32's P02 and P05 generations lean closer to specific training images than most other prompts do.

### Human evaluation

A rater pack (`outputs/rating/rater_1.html`, `rater_2.html`, `rater_3.html`, plus a blank
`outputs/rating/blind_rating_sheet.csv`) holds 25 blind pairs of rank 32 vs. the base model, left/right
randomized by seed 266 -- the same 25 pairs in the same order for every rater -- with no filename or model
name anywhere in the page so the comparison stays blind.

**Raters in: rater_1 (manav), 25/25 pairs rated** (`outputs/rating/rater_1_ratings.csv`). Missing: rater_2
(pramod), rater_3 (outside rater 1).

**Win rate, rater_1 (manav) only: 0.40** (10 of 25 pairs preferred the final run over the base model;
`score_blind_ratings()` in `p2_generate_eval.ipynb`, `blind_answer_key.csv` supplied). This single rater
preferred the base model on more pairs than the final run -- rater_2 (pramod) and rater_3 (outside rater
1) are still missing, so this is not yet the overall win rate the lab asks for, only one rater's.

**Inter-rater agreement (Fleiss' kappa, percent agreement): not computable yet** -- needs 3 raters in;
only rater_1 is in so far.

## Hugging Face

- Final model (rank 32): https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r32
- Comparison model (rank 8): https://huggingface.co/itzmnv/data266-lab2-sd15-pokemon-lora-r8
- Generated images (120, with prompt/seed/model/nsfw metadata): https://huggingface.co/datasets/itzmnv/data266-lab2-sd15-pokemon-generations

## Tools and models

| Tool or model | Use | License |
|---|---|---|
| reach-vb/pokemon-blip-captions | Training and test images | CC BY-NC-SA 4.0 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | Base model | CreativeML OpenRAIL-M |
| diffusers `train_text_to_image_lora.py` | LoRA training | Apache-2.0 |
| peft, accelerate | LoRA layers and mixed precision | Apache-2.0 |
| torchmetrics (KID), transformers CLIP | KID and CLIP score | Apache-2.0 / Apache-2.0 |
| openai/clip-vit-large-patch14 | CLIP score | MIT |
| open_clip (ViT-B-32, openai weights) | Memorization check | MIT |
| lpips | Diversity between seeds | BSD-2 |
| diffusers safety checker | NSFW filter on outputs | Apache-2.0 |

No hosted API is used. Seed is 266 throughout.
