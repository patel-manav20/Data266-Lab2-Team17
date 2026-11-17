# Part 2: Diffusion Fine-Tuning (Pramod)

The two notebooks prepare the captioned Pokemon image dataset, train two SD 1.5 LoRA adapters, and compare them with the base model. The evaluation includes KID, CLIP score, LPIPS diversity, blinded rating pairs and nearest-training-image checks.

## Models and settings

- Dataset: `reach-vb/pokemon-blip-captions`; corrupted and duplicate images are removed before the 90/10 split with seed 266.
- Base: `stable-diffusion-v1-5/stable-diffusion-v1-5`, 512×512.
- LoRA ranks: 4 and 32; each uses 1,500 optimizer steps, learning rate 1e-4, batch size 1 and gradient accumulation 4.
- Generation: 20 fixed prompts, seeds 0 and 1, DPM-Solver++ with 30 steps and guidance 7.5, for each of the three models.

## How to run

1. Open `notebooks/p2_01_data_and_training.ipynb` on Colab with a T4 GPU and run all cells. Grant Drive access. Add an `HF_TOKEN` secret with write access to upload the splits and adapters. `GITHUB_TOKEN` is optional for cloning the repository.
2. Open `notebooks/p2_02_generate_and_evaluate.ipynb` and run all cells. It finds the saved Drive artifacts automatically. On Kaggle, attach the training notebook's output; elsewhere set `TRAIN_OUT` to the directory containing `adapters/`, `p2_data/` and `outputs/p2_training_runs.csv`.
3. Give the 25 images in `p2_eval/human_pairs/` to both team members and at least one outside rater. Keep `human_pairs_key.csv` private. Combine their completed copies of `human_ratings_template.csv` into `p2_eval/human_ratings.csv`, with a distinct `rater` name, all 25 pair numbers, A/B choices for `style_winner` and `prompt_winner`, and `is_external` set to true or false. Rerun the ratings cell to save the win rates.
4. Save both executed notebooks with their outputs. Choose and explain at least two failure examples from the comparison grids.

Colab saves the splits, checkpoints, completed adapters, training measurements and evaluation artifacts under `MyDrive/data266_p2/`. Each completed training run is saved before the next starts. Evaluation checks that both adapters and measured training summaries exist, the splits do not overlap, and the in-domain prompts match the original held-out captions.

## Results

Measured in the executed `p2_02_generate_and_evaluate.ipynb` on a Colab T4. KID is computed on in-domain held-out prompts; lower is better.

| Model | KID in-domain (mean ± std) | CLIP all | CLIP in-domain | LPIPS diversity | NSFW flagged | Trainable params | GPU hours |
|---|---|---|---|---|---|---|---|
| Base SD 1.5 | 0.1018 ± 0.0108 | 29.93 | 29.57 | 0.845 | 0 | – | – |
| LoRA r4 | 0.0444 ± 0.0073 | 29.07 | 29.66 | 0.823 | 2 | 797,184 | 0.90 |
| **LoRA r32 (final)** | **0.0306 ± 0.0067** | 28.96 | 28.07 | 0.797 | 2 | 6,377,472 | 0.80 |

- **Final adapter:** both adapters bring generations much closer to the training domain, and r32 has the lowest KID, so it's the final adapter. Its prompt alignment (CLIP) stays close to the base model's.
- **Memorization check:** the highest CLIP cosine between a generated image and its nearest training image is 0.83, so there are no near-copies.
- **Where to find them:**
  - the final adapter and model card: https://huggingface.co/pramodsatyad/data266-p2-pokemon-lora_r32;
  - the 120 generated images: https://huggingface.co/datasets/pramodsatyad/data266-p2-generated-images.

The notebooks write `p2_training_runs.csv`, `p2_data_summary.json`, `p2_metrics.csv`, `p2_results_table.csv`, `memorization.csv` and, after complete human ratings, `p2_human_results.csv`. The final adapter is selected using the unrounded in-domain KID. Its model card is generated from the measured results.

## Safety and limitations

The Stable Diffusion safety checker stays enabled during generation; flagged outputs are counted. The dataset license is CC-BY-NC-SA-4.0 and the images depict copyrighted characters, so the adapters are intended for non-commercial research and coursework. The dataset is concentrated on cartoon creatures, and its BLIP captions may miss visual details. KID based on 20 generated in-domain images is noisy.

## Development checks

Edit the builder scripts under `src/` and regenerate the notebooks from the repository root:

```sh
python part2_diffusion/member_pramod/src/build_p2_01_notebook.py part2_diffusion/member_pramod/notebooks/p2_01_data_and_training.ipynb
python part2_diffusion/member_pramod/src/build_p2_02_notebook.py part2_diffusion/member_pramod/notebooks/p2_02_generate_and_evaluate.ipynb tests/p2_prompts_pramod.json part2_diffusion/member_pramod/notebooks/p2_01_data_and_training.ipynb
python part2_diffusion/member_pramod/src/test_part2_notebooks.py
```

The CPU checks use `nbformat`, `numpy`, `pandas`, `Pillow` and `torch`. They exercise artifact validation, metric aggregation, bounded KID batches, adapter selection, training measurements and rating completeness with stubbed models. Real training and metric computation require a GPU run.

## AI tools used

Codex reviewed and fixed the training/evaluation workflow and added CPU regression checks.
