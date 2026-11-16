import re
import nbformat as nbf
import sys, json

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
prompts = json.load(open(sys.argv[2]))["prompts"]
setup = [c for c in nbf.read(sys.argv[3], 4).cells if c.cell_type == "code"][0].source
setup = re.sub(r'^STAMP = ".*"$', 'STAMP = ""', setup, flags=re.M)

cells = [
md("""# Part 2.3 and 2.4: Image Generation, Evaluation and Safety

Inputs: the processed train/test split and the two LoRA adapters produced by `p2_01_data_and_training.ipynb`. On Kaggle they are attached as the output of that notebook; on Colab they are read from Google Drive (`MyDrive/data266_p2`); elsewhere set `TRAIN_OUT` to the folder that holds `p2_data/` and `adapters/`.

Same sampler settings for every model: DPM-Solver++ multistep, 30 steps, guidance 7.5, 512×512, fp16, seeds 0 and 1. The built-in Stable Diffusion safety checker is the NSFW filter: flagged outputs are blacked out and counted."""),
code(setup),
code("""%pip install -q diffusers peft torchmetrics torch-fidelity lpips accelerate
%pip uninstall -y -q torchao
import importlib.metadata as md
print({p: md.version(p) for p in ["torch", "diffusers", "transformers", "peft", "torchmetrics", "torch-fidelity", "lpips"]})"""),

md("## Settings and fixed prompts\n\nThe 20 prompts are copied from `tests/p2_prompts_pramod.json`, committed before training finished. The 10 in-domain prompts are test-set captions."),
code(f"""import glob, json, time, collections, shutil
import pandas as pd
from PIL import Image

PROMPTS = {json.dumps(prompts, indent=1)}
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
TRAIN_OUT = os.environ.get("TRAIN_OUT") or os.path.dirname(os.path.dirname((glob.glob("/kaggle/input/**/adapters/lora_r4.safetensors", recursive=True) + glob.glob("/content/drive/MyDrive/data266_p2/adapters/lora_r4.safetensors") + [f"{{WORK}}/adapters/x"])[0]))
if not os.path.isfile(f"{{TRAIN_OUT}}/adapters/lora_r4.safetensors"):
    from huggingface_hub import hf_hub_download, snapshot_download
    TRAIN_OUT = f"{{WORK}}/train_out"
    os.makedirs(f"{{TRAIN_OUT}}/adapters", exist_ok=True)
    for name in ["lora_r4", "lora_r32"]:
        src = hf_hub_download(f"pramodsatyad/data266-p2-pokemon-{{name}}", "pytorch_lora_weights.safetensors", token=secret("HF_TOKEN"))
        shutil.copy(src, f"{{TRAIN_OUT}}/adapters/{{name}}.safetensors")
    snapshot_download("pramodsatyad/data266-p2-pokemon-splits", repo_type="dataset", local_dir=f"{{TRAIN_OUT}}/p2_data", token=secret("HF_TOKEN"))
    os.makedirs(f"{{TRAIN_OUT}}/outputs", exist_ok=True)
    shutil.move(f"{{TRAIN_OUT}}/p2_data/outputs/p2_training_runs.csv", f"{{TRAIN_OUT}}/outputs/p2_training_runs.csv")
    print("training artifacts not on this Drive: downloaded adapters, split and training table from Hugging Face")
ADAPTERS = {{"lora_r4": f"{{TRAIN_OUT}}/adapters/lora_r4.safetensors", "lora_r32": f"{{TRAIN_OUT}}/adapters/lora_r32.safetensors"}}
MODELS = ["base", "lora_r4", "lora_r32"]
SEEDS = [0, 1]
STEPS, GUIDANCE, RES = 30, 7.5, 512
EVAL_ROOT = "/content/drive/MyDrive/data266_p2" if ON_COLAB else WORK
GEN = f"{{EVAL_ROOT}}/generated"
OUT = f"{{EVAL_ROOT}}/p2_eval"
os.makedirs(OUT, exist_ok=True)
print(TRAIN_OUT)
print(collections.Counter(p["type"] for p in PROMPTS))"""),

code("""missing = [p for p in ADAPTERS.values() if not os.path.isfile(p)]
for split in ["train", "test"]:
    path = f"{TRAIN_OUT}/p2_data/{split}/metadata.jsonl"
    if not os.path.isfile(path):
        missing.append(path)
if missing:
    raise FileNotFoundError("Run the training notebook first, or set TRAIN_OUT to its saved artifacts: " + ", ".join(missing))

split_records = {}
for split in ["train", "test"]:
    with open(f"{TRAIN_OUT}/p2_data/{split}/metadata.jsonl") as f:
        split_records[split] = [json.loads(line) for line in f if line.strip()]
    for r in split_records[split]:
        if not os.path.isfile(f"{TRAIN_OUT}/p2_data/{split}/{r['file_name']}"):
            raise FileNotFoundError(f"Missing {split} image: {r['file_name']}")
test_captions = {r["file_name"]: r["text"] for r in split_records["test"]}
train_names = {r["file_name"] for r in split_records["train"]}
if train_names & set(test_captions):
    raise ValueError("Train/test images overlap; restore the seed-266 split.")
if len(PROMPTS) != 20 or len({p["pid"] for p in PROMPTS}) != 20 or collections.Counter(p["type"] for p in PROMPTS) != {"in_domain": 10, "out_of_domain": 5, "compositional": 5}:
    raise ValueError("Expected the original 20 fixed evaluation prompts.")
for p in PROMPTS:
    if p["type"] == "in_domain" and test_captions.get(p["test_file"]) != p["prompt"]:
        raise ValueError(f"{p['pid']} does not match its held-out caption; restore the original split rather than changing the fixed prompts.")
if len(test_captions) < 20:
    raise ValueError("KID requires at least 20 held-out images for this evaluation.")
train_csv = glob.glob(f"{TRAIN_OUT}/**/p2_training_runs.csv", recursive=True)
if not train_csv:
    raise FileNotFoundError("p2_training_runs.csv is missing; copy the measured training summary into TRAIN_OUT/outputs.")
train_df = pd.read_csv(train_csv[0])
required = {"run", "trainable_params", "learning_rate", "steps", "batch_size", "grad_accum", "peak_gpu_mem_gb", "gpu_hours", "gpu"}
if not required.issubset(train_df.columns) or train_df.run.duplicated().any() or set(train_df.run) != {"lora_r4", "lora_r32"}:
    raise ValueError("The training summary must contain both completed LoRA runs and their measured settings.")
print("Validated adapters, disjoint splits, held-out prompts and training summary.")"""),

md("## 2.3 Generate 40 images per model"),
code("""from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

meta = []
for m in MODELS:
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    if m != "base":
        pipe.load_lora_weights(ADAPTERS[m])
    os.makedirs(f"{GEN}/{m}", exist_ok=True)
    t0 = time.time()
    for p in PROMPTS:
        for s in SEEDS:
            r = pipe(p["prompt"], num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES,
                     generator=torch.Generator("cuda").manual_seed(s))
            f = f"{m}/{p['pid']}_s{s}.png"
            r.images[0].save(f"{GEN}/{f}")
            meta.append({"file_name": f, "model": m, "pid": p["pid"], "type": p["type"], "prompt": p["prompt"], "seed": s,
                         "steps": STEPS, "guidance": GUIDANCE, "scheduler": "DPMSolverMultistep", "nsfw_flagged": bool(r.nsfw_content_detected and r.nsfw_content_detected[0])})
    print(m, "seconds per image: %.2f" % ((time.time() - t0) / (len(PROMPTS) * len(SEEDS))), flush=True)
meta_df = pd.DataFrame(meta)
meta_df.to_csv(f"{GEN}/metadata.csv", index=False)
print("NSFW flagged per model:", meta_df.groupby("model").nsfw_flagged.sum().to_dict())"""),

code("""import gc
del pipe
gc.collect(); torch.cuda.empty_cache()"""),

md("### Side-by-side grids (seed 0): base vs run 1 (LoRA rank 4) vs run 2 (LoRA rank 32)"),
code("""import matplotlib.pyplot as plt
grid_pids = [p["pid"] for p in PROMPTS if p["type"] == "in_domain"][:4] + [p["pid"] for p in PROMPTS if p["type"] == "out_of_domain"][:2] + [p["pid"] for p in PROMPTS if p["type"] == "compositional"][:2]
PTEXT = {p["pid"]: p["prompt"] for p in PROMPTS}
fig, axes = plt.subplots(len(grid_pids), 3, figsize=(10, 3.4 * len(grid_pids)))
for r, pid in enumerate(grid_pids):
    for c, m in enumerate(MODELS):
        axes[r, c].imshow(Image.open(f"{GEN}/{m}/{pid}_s0.png")); axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(m, fontsize=12)
    axes[r, 0].text(-20, 256, f"{pid}: {PTEXT[pid][:45]}", rotation=90, va="center", ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/grid_8_prompts.png", dpi=80); plt.show()"""),

md("""## 2.4 Evaluation

* **KID** (primary): in-domain generations (10 prompts × 2 seeds = 20 images) against all test images, Inception features, subset size 20. KID with 20 images is noisy, so the mean and standard deviation over subsets are both reported.
* **CLIP score** (primary): `openai/clip-vit-large-patch14`, prompt vs image, all 40 images and per prompt type.
* **LPIPS diversity** (secondary): AlexNet LPIPS between the seed-0 and seed-1 images of each prompt, averaged."""),
code("""import numpy as np
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
import lpips

def load_u8(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB").resize((299, 299)))).permute(2, 0, 1)

test_paths = [f"{TRAIN_OUT}/p2_data/test/{r['file_name']}" for r in split_records["test"]]
clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
lp = lpips.LPIPS(net="alex", verbose=False).to("cuda")
clip.eval(); lp.eval()

@torch.inference_mode()
def update_kid(metric, paths, real, batch_size=8):
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([load_u8(p) for p in paths[start:start + batch_size]]).to("cuda")
        metric.update(batch, real=real)

def to_lpips(path):
    x = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1
    return x.unsqueeze(0).to("cuda")

rows = []
for m in MODELS:
    sub = meta_df[meta_df.model == m]
    ind = sub[sub.type == "in_domain"]
    kid = KernelInceptionDistance(subset_size=min(20, len(ind)), subsets=100, normalize=False).to("cuda")
    update_kid(kid, test_paths, real=True)
    update_kid(kid, [f"{GEN}/{f}" for f in ind.file_name], real=False)
    torch.manual_seed(SEED)
    kid_mean, kid_std = kid.compute()
    del kid
    gc.collect(); torch.cuda.empty_cache()
    clip_scores = []
    for _, r in sub.iterrows():
        img = torch.from_numpy(np.array(Image.open(f"{GEN}/{r.file_name}").convert("RGB"))).permute(2, 0, 1).to("cuda")
        with torch.inference_mode():
            clip_scores.append(float(clip(img, r.prompt)))
        clip.reset()
    sub = sub.assign(clip=clip_scores)
    meta_df.loc[sub.index, "clip"] = clip_scores
    with torch.no_grad():
        lps = [float(lp(to_lpips(f"{GEN}/{m}/{pid}_s0.png"), to_lpips(f"{GEN}/{m}/{pid}_s1.png"))) for pid in sub.pid.unique()]
    rows.append({"model": m, "KID_in_domain_mean": float(kid_mean), "KID_in_domain_std": float(kid_std),
                 "CLIP_all": np.mean(clip_scores), **{f"CLIP_{t}": sub.loc[sub.type == t, "clip"].mean() for t in ["in_domain", "out_of_domain", "compositional"]},
                 "LPIPS_diversity": np.mean(lps), "nsfw_flagged": int(sub.nsfw_flagged.sum())})
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT}/p2_metrics.csv", index=False)
meta_df.to_csv(f"{GEN}/metadata.csv", index=False)
del clip, lp
gc.collect(); torch.cuda.empty_cache()
display(metrics.round(4))"""),

md("### Results table (training cost from the training notebook)"),
code("""results = metrics.merge(train_df[["run", "trainable_params", "peak_gpu_mem_gb", "gpu_hours"]].rename(columns={"run": "model"}), on="model", how="left", validate="one_to_one")
FINAL = min(["lora_r4", "lora_r32"], key=lambda m: metrics.set_index("model").loc[m, "KID_in_domain_mean"])
print("final run (lowest in-domain KID):", FINAL)
results.to_csv(f"{OUT}/p2_results_table.csv", index=False)
display(results.round(4))"""),

md("""### Failure modes

Out-of-domain and compositional prompts for the base model and both runs (seed 0)."""),
code("""fail_pids = [p["pid"] for p in PROMPTS if p["type"] != "in_domain"]
fig, axes = plt.subplots(3, len(fail_pids), figsize=(2.2 * len(fail_pids), 7))
for c, pid in enumerate(fail_pids):
    for r, m in enumerate(MODELS):
        axes[r, c].imshow(Image.open(f"{GEN}/{m}/{pid}_s0.png")); axes[r, c].axis("off")
        axes[r, c].set_title(f"{m}\\n{pid} CLIP {meta_df.loc[(meta_df.model == m) & (meta_df.file_name == f'{m}/{pid}_s0.png'), 'clip'].iloc[0]:.1f}", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/failure_modes.png", dpi=70); plt.show()
for p in PROMPTS:
    if p["type"] != "in_domain":
        print(p["pid"], p["type"], p["prompt"])"""),

md("""## Safety: memorization check

For the 10 in-domain prompts (final run, seed 0), find the most similar training image by CLIP image-embedding cosine similarity, and show the 3 closest pairs."""),
code("""from transformers import CLIPModel, CLIPProcessor
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
cm.eval()
cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@torch.no_grad()
def img_emb(paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = cp(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]], return_tensors="pt").to("cuda")
        e = cm.get_image_features(pixel_values=x.pixel_values.half())
        e = getattr(e, "pooler_output", e)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1))
    return torch.cat(out)

train_paths = [f"{TRAIN_OUT}/p2_data/train/{r['file_name']}" for r in split_records["train"]]
gen_paths = [f"{GEN}/{FINAL}/{p['pid']}_s0.png" for p in PROMPTS if p["type"] == "in_domain"]
sim = img_emb(gen_paths) @ img_emb(train_paths).T
best, idx = sim.max(dim=1)
mem = pd.DataFrame({"generated": [os.path.basename(p) for p in gen_paths], "closest_train": [os.path.basename(train_paths[i]) for i in idx.tolist()], "clip_cosine": best.cpu().numpy().round(4)}).sort_values("clip_cosine", ascending=False)
mem.to_csv(f"{OUT}/memorization.csv", index=False)
print(mem.to_string(index=False))
fig, axes = plt.subplots(3, 2, figsize=(6, 9))
for r, (_, row) in enumerate(mem.head(3).iterrows()):
    axes[r, 0].imshow(Image.open(f"{GEN}/{FINAL}/{row.generated}")); axes[r, 0].set_title(f"generated {row.generated}", fontsize=8)
    axes[r, 1].imshow(Image.open(f"{TRAIN_OUT}/p2_data/train/{row.closest_train}")); axes[r, 1].set_title(f"train {row.closest_train} cos {row.clip_cosine:.3f}", fontsize=8)
    axes[r, 0].axis("off"); axes[r, 1].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/memorization_top3.png", dpi=80); plt.show()"""),

md("""## Human evaluation sheet (blinded)

25 pairs of final run vs base model for the same prompt and seed, left/right randomized with seed 266. Each sheet image shows A and B only; the key is saved separately. Raters copy `human_ratings_template.csv` to `human_ratings.csv`, use a distinct rater name, mark `is_external` true or false, and fill both winner columns with A or B. At least three raters (both members and one outside person) must each judge all 25 pairs before results are reported."""),
code("""import textwrap
rng = np.random.default_rng(SEED)
cands = [(p["pid"], s) for p in PROMPTS for s in SEEDS]
pick = [cands[i] for i in rng.choice(len(cands), 25, replace=False)]
os.makedirs(f"{OUT}/human_pairs", exist_ok=True)
key = []
for n, (pid, s) in enumerate(pick, 1):
    final_left = bool(rng.integers(0, 2))
    left, right = (FINAL, "base") if final_left else ("base", FINAL)
    label = textwrap.fill(f"Pair {n:02d}: {PTEXT[pid]}", width=100)
    header = 32 + 16 * len(label.splitlines())
    canvas = Image.new("RGB", (1044, 512 + header), "white")
    canvas.paste(Image.open(f"{GEN}/{left}/{pid}_s{s}.png"), (0, header)); canvas.paste(Image.open(f"{GEN}/{right}/{pid}_s{s}.png"), (532, header))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas); d.multiline_text((8, 6), label, fill="black"); d.text((8, header - 18), "A", fill="black"); d.text((540, header - 18), "B", fill="black")
    canvas.save(f"{OUT}/human_pairs/pair_{n:02d}.png")
    key.append({"pair": n, "pid": pid, "seed": s, "A": left, "B": right})
pd.DataFrame(key).to_csv(f"{OUT}/human_pairs_key.csv", index=False)
pd.DataFrame({"rater": "", "pair": range(1, 26), "style_winner": "", "prompt_winner": "", "is_external": ""}).to_csv(f"{OUT}/human_ratings_template.csv", index=False)
print("saved 25 pairs to", f"{OUT}/human_pairs")"""),
code("""ratings_path = next(iter(glob.glob("/kaggle/input/**/human_ratings.csv", recursive=True) + glob.glob("human_ratings.csv") + glob.glob(f"{OUT}/human_ratings.csv")), None)
if ratings_path:
    rt = pd.read_csv(ratings_path, keep_default_na=False)
    required = {"rater", "pair", "style_winner", "prompt_winner", "is_external"}
    if not required.issubset(rt.columns):
        raise ValueError("Ratings require rater, pair, style_winner, prompt_winner and is_external columns.")
    rt["rater"] = rt.rater.astype(str).str.strip()
    for c in ["style_winner", "prompt_winner"]:
        rt[c] = rt[c].astype(str).str.strip().str.upper()
    rt["is_external"] = rt.is_external.astype(str).str.strip().str.lower()
    if (rt.rater == "").any() or rt.duplicated(["rater", "pair"]).any() or not rt[["style_winner", "prompt_winner"]].isin(["A", "B"]).all().all():
        raise ValueError("Each rating needs a rater, a unique pair, and A/B choices for both criteria.")
    if not rt.is_external.isin(["true", "false"]).all() or (rt.groupby("rater").is_external.nunique() != 1).any():
        raise ValueError("Each rater must have one consistent true/false is_external value.")
    expected = set(k["pair"] for k in key)
    if rt.rater.nunique() < 3 or any(set(g.pair) != expected for _, g in rt.groupby("rater")):
        raise ValueError("At least three raters must each complete the same 25 pairs.")
    people = rt[["rater", "is_external"]].drop_duplicates()
    if (people.is_external == "false").sum() < 2 or (people.is_external == "true").sum() < 1:
        raise ValueError("Ratings must include both team members and at least one outside person.")
    rt = rt.merge(pd.DataFrame(key), on="pair", validate="many_to_one")
    for crit in ["style_winner", "prompt_winner"]:
        rt[crit + "_model"] = rt.apply(lambda r: r[r[crit]], axis=1)
    summary = {c: round((rt[c + "_model"] == FINAL).mean(), 3) for c in ["style_winner", "prompt_winner"]}
    pd.DataFrame([{"raters": rt.rater.nunique(), "judgements": len(rt), **summary}]).to_csv(f"{OUT}/p2_human_results.csv", index=False)
    print("raters:", rt.rater.nunique(), "judgements:", len(rt), "final-run win rate:", summary)
else:
    print("human_ratings.csv not found yet")"""),

md("## Model card for the final run (written from the computed results)"),
code("""tr = train_df.set_index("run").loc[FINAL] if FINAL in set(train_df.get("run", [])) else None
mr = metrics.set_index("model")
card = f\"\"\"---
license: other
license_name: creativeml-openrail-m-and-cc-by-nc-sa-4.0
base_model: {BASE_MODEL}
tags: [stable-diffusion, lora, text-to-image, diffusers]
---
# DATA 266 Lab 2, Part 2: {FINAL} (Team 17)

LoRA adapter for Stable Diffusion 1.5, fine-tuned on Pokemon-style BLIP-captioned artwork for non-commercial research and coursework.

**Base model:** {BASE_MODEL} (CreativeML OpenRAIL-M).
**Data:** reach-vb/pokemon-blip-captions (CC-BY-NC-SA-4.0), corrupted and duplicate images removed, 90/10 train/test split with seed 266, 512x512.
**Training:** diffusers train_text_to_image_lora.py, {FINAL}{"" if tr is None else f", trainable parameters {int(tr.trainable_params):,}, learning rate {tr.learning_rate}, {int(tr.steps)} steps, batch {int(tr.batch_size)} x gradient accumulation {int(tr.grad_accum)}, peak GPU memory {tr.peak_gpu_mem_gb} GB, {tr.gpu_hours} GPU-hours on {tr.gpu}"}.

**Results** (20 fixed prompts x 2 seeds, DPM-Solver++ 30 steps, guidance 7.5):

| Model | KID in-domain (lower is better) | CLIP score | LPIPS diversity |
|---|---|---|---|
| base | {mr.loc['base','KID_in_domain_mean']:.4f} | {mr.loc['base','CLIP_all']:.2f} | {mr.loc['base','LPIPS_diversity']:.3f} |
| {FINAL} | {mr.loc[FINAL,'KID_in_domain_mean']:.4f} | {mr.loc[FINAL,'CLIP_all']:.2f} | {mr.loc[FINAL,'LPIPS_diversity']:.3f} |

KID with 20 generated images is noisy.

**Limitations:** strongly biased to the flat cartoon-creature style of the training set, weaker on photographic and out-of-domain prompts, weak at multi-object composition and counting; captions are machine-generated (BLIP) and short.

**License and use:** the training images depict copyrighted characters, so this adapter is for non-commercial research and education only. Do not use it to reproduce existing named characters.
\"\"\"
open(f"{OUT}/MODEL_CARD.md", "w").write(card)
print(card)
if secret("HF_TOKEN"):
    from huggingface_hub import HfApi, login
    login(token=secret("HF_TOKEN")); api = HfApi(); user = api.whoami()["name"]
    repo = f"{user}/data266-p2-pokemon-{FINAL}"
    api.create_repo(repo, exist_ok=True)
    api.upload_file(path_or_fileobj=f"{OUT}/MODEL_CARD.md", path_in_repo="README.md", repo_id=repo)
    print("model card pushed to", f"https://huggingface.co/{repo}")"""),

md("## Upload images to a Hugging Face dataset (prompt, seed and model in the metadata)"),
code("""from huggingface_hub import HfApi, login
if secret("HF_TOKEN"):
    login(token=secret("HF_TOKEN")); api = HfApi(); user = api.whoami()["name"]
    repo = f"{user}/data266-p2-generated-images"
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=GEN, repo_id=repo, repo_type="dataset")
    print(f"https://huggingface.co/datasets/{repo}")
else:
    print("no HF_TOKEN: images and metadata.csv are in", GEN)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}})
nbf.write(nb, sys.argv[1])
