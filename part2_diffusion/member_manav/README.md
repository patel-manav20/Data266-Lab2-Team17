# Part 2: Diffusion Fine-Tuning (Manav)

This part fine-tunes Stable Diffusion 1.5 on Pokemon images with captions. Two LoRA runs (rank 8 and rank 32) use the same data and the same number of steps. Both are compared with the base model.

## Workflow

```
Pokemon captions dataset
        |
        v
Remove corrupted and duplicate images -> split 90/10 (seed 266) -> resize 512x512
        |
        v
SD 1.5 + LoRA rank 8      SD 1.5 + LoRA rank 32
        |                         |
        +-----------+-------------+
                    v
20 fixed prompts x seeds 0 and 1 -> base, run 1, run 2 (40 images each)
                    |
                    v
KID, CLIP score, LPIPS, blinded human pairs, GPU-hours
                    |
                    v
Models and images on Hugging Face, model card for the final run
```

## Tools and models

| Tool or model | Use | License |
|---|---|---|
| reach-vb/pokemon-blip-captions | Training and test images | CC BY-NC-SA 4.0 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | Base model | CreativeML OpenRAIL-M |
| diffusers `train_text_to_image_lora.py` | LoRA training | Apache-2.0 |
| peft, accelerate | LoRA layers and mixed precision | Apache-2.0 |
| torchmetrics (KID, CLIP score) | Quality and prompt match | Apache-2.0 |
| openai/clip-vit-large-patch14 | CLIP score and memorization check | MIT |
| lpips | Diversity between seeds | BSD-2 |
| diffusers safety checker | NSFW filter on outputs | Apache-2.0 |

The dataset shows copyrighted characters, so the models are for non-commercial research only. No hosted API is used. Seed is 266.

## How to run

1. Use a machine with one NVIDIA GPU (16 GB is enough).
2. `pip install -r requirements.txt`
3. Log in to Hugging Face with `huggingface-cli login`. Never save the token in the repo.
4. Run `notebooks/p2_finetune.ipynb` from top to bottom.

Weights and generated images go to Hugging Face, not to GitHub.
