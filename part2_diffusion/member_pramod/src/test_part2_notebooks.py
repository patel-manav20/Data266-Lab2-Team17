import ast
import collections
import gc
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch


MEMBER = Path(__file__).resolve().parents[1]
ROOT = MEMBER.parents[1]
TRAIN = MEMBER / "notebooks/p2_01_data_and_training.ipynb"
EVAL = MEMBER / "notebooks/p2_02_generate_and_evaluate.ipynb"


def cell(path, marker):
    return next(c.source for c in nbformat.read(path, 4).cells if c.cell_type == "code" and marker in c.source)


class Part2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.prompts = json.loads((ROOT / "tests/p2_prompts_pramod.json").read_text())["prompts"]
        self.training = pd.DataFrame([
            dict(run=m, trainable_params=100, learning_rate=1e-4, steps=1500, batch_size=1, grad_accum=4, peak_gpu_mem_gb=8.0, gpu_hours=1.0, gpu="test")
            for m in ["lora_r4", "lora_r32"]
        ])
        for split in ["train", "test"]:
            folder = self.base / "p2_data" / split
            folder.mkdir(parents=True)
            records = ([dict(file_name=p["test_file"], text=p["prompt"]) for p in self.prompts[:10]] +
                       [dict(file_name=f"extra{i}.png", text="held out") for i in range(10)]) if split == "test" else [dict(file_name="train.png", text="train")]
            (folder / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            for r in records:
                Image.new("RGB", (32, 32), "blue").save(folder / r["file_name"])
        (self.base / "adapters").mkdir()
        for m in self.training.run:
            (self.base / "adapters" / f"{m}.safetensors").touch()
        (self.base / "outputs").mkdir()
        self.training.to_csv(self.base / "outputs/p2_training_runs.csv", index=False)

    def preflight(self):
        ns = dict(os=os, glob=glob, json=json, collections=collections, pd=pd, TRAIN_OUT=str(self.base),
                  ADAPTERS={m: str(self.base / "adapters" / f"{m}.safetensors") for m in self.training.run}, PROMPTS=self.prompts)
        exec(cell(EVAL, "missing = [p for p in ADAPTERS"), ns)
        return ns

    def test_notebook_syntax_and_builder_consistency(self):
        tasks = [(TRAIN, "build_p2_01_notebook.py", []), (EVAL, "build_p2_02_notebook.py", [str(ROOT / "tests/p2_prompts_pramod.json"), str(TRAIN)])]
        for original, builder, args in tasks:
            notebook = nbformat.read(original, 4)
            nbformat.validate(notebook)
            for c in notebook.cells:
                if c.cell_type == "code":
                    ast.parse("\n".join(l for l in c.source.splitlines() if not l.startswith("%")))
            dest = self.base / original.name
            subprocess.run([os.sys.executable, str(MEMBER / "src" / builder), str(dest), *args], check=True)
            rebuilt = nbformat.read(dest, 4)
            self.assertEqual([(c.cell_type, c.source) for c in notebook.cells], [(c.cell_type, c.source) for c in rebuilt.cells])

    def test_preflight_and_bad_inputs(self):
        self.assertEqual(len(self.preflight()["test_captions"]), 20)
        self.prompts[0]["prompt"] = "changed caption"
        with self.assertRaisesRegex(ValueError, "held-out caption"):
            self.preflight()
        self.prompts[0]["prompt"] = "a red fish with a gold crown on its head"
        path = self.base / "p2_data/train/metadata.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(dict(file_name=self.prompts[0]["test_file"], text="leak")) + "\n")
        Image.new("RGB", (32, 32)).save(path.parent / self.prompts[0]["test_file"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.preflight()

    def test_missing_training_summary(self):
        (self.base / "outputs/p2_training_runs.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "training_runs"):
            self.preflight()

    def test_split_cleanup_and_persistence(self):
        source = cell(TRAIN, "split_idx =")
        data, keep, out = self.base / "fresh", self.base / "drive", self.base / "summary"
        out.mkdir()
        for split in ["train", "test"]:
            (data / split).mkdir(parents=True)
            (data / split / "stale.png").touch()
        records = [dict(src_idx=i, text=f"caption {i}", img=Image.new("RGB", (64, 80))) for i in range(50)]
        ns = dict(np=np, SEED=266, records=records, RES=32, Image=Image, os=os, json=json, shutil=shutil,
                  DATA=str(data), KEEP=str(keep), WORK=str(self.base), OUT=str(out), data_summary={"kept": 50})
        exec(source, ns)
        first = ns["split_idx"]
        exec(source, ns)
        self.assertEqual(ns["split_idx"], first)
        for split, count in [("train", 45), ("test", 5)]:
            self.assertEqual(len(list((data / split).glob("*.png"))), count)
            self.assertEqual(len(list((keep / "p2_data" / split).glob("*.png"))), count)
        self.assertTrue((out / "p2_data_summary.json").is_file())

    def test_metric_batches_aggregates_and_selection(self):
        ns = self.preflight()
        gen, out = self.base / "generated", self.base / "eval"
        out.mkdir()
        meta = []
        models = ["base", "lora_r4", "lora_r32"]
        for m in models:
            (gen / m).mkdir(parents=True)
            for p in self.prompts:
                for seed in [0, 1]:
                    name = f"{m}/{p['pid']}_s{seed}.png"
                    Image.new("RGB", (32, 32), "blue").save(gen / name)
                    meta.append(dict(file_name=name, model=m, pid=p["pid"], type=p["type"], prompt=p["prompt"], nsfw_flagged=False))
        batches = []

        class Metric:
            def to(self, device):
                return self

            def eval(self):
                return self

        class KID(Metric):
            def __init__(self, **kw):
                self.calls = []

            def update(self, batch, real):
                batches.append(len(batch))
                self.calls.append((len(batch), real))

            def compute(self):
                self_test.assertEqual(sum(n for n, real in self.calls if real), 20)
                self_test.assertEqual(sum(n for n, real in self.calls if not real), 20)
                return torch.tensor(0.123456), torch.tensor(0.01)

        class CLIP(Metric):
            def __init__(self, **kw):
                pass

            def __call__(self, image, prompt):
                self_test.assertFalse(torch.is_grad_enabled())
                return torch.tensor(25.0)

            def reset(self):
                pass

        class LPIPS(CLIP):
            def __call__(self, a, b):
                self_test.assertFalse(torch.is_grad_enabled())
                return torch.tensor(0.3)

        self_test = self
        modules = {name: types.ModuleType(name) for name in ["torchmetrics", "torchmetrics.image", "torchmetrics.image.kid", "torchmetrics.multimodal", "torchmetrics.multimodal.clip_score", "lpips"]}
        modules["torchmetrics.image.kid"].KernelInceptionDistance = KID
        modules["torchmetrics.multimodal.clip_score"].CLIPScore = CLIP
        modules["lpips"].LPIPS = LPIPS
        ns.update(torch=torch, np=np, Image=Image, gc=gc, MODELS=models, GEN=str(gen), OUT=str(out), meta_df=pd.DataFrame(meta), SEED=266, display=lambda x: None)
        with patch.dict(os.sys.modules, modules):
            exec(cell(EVAL, "def update_kid").replace('.to("cuda")', '.to("cpu")'), ns)
        self.assertLessEqual(max(batches), 8)
        self.assertEqual(len(ns["metrics"]), 3)
        self.assertTrue((ns["metrics"].CLIP_in_domain == 25).all())
        self.assertEqual(len(pd.read_csv(gen / "metadata.csv")), 120)
        self.assertTrue((pd.read_csv(gen / "metadata.csv")["clip"] == 25).all())
        ns["metrics"].loc[ns["metrics"].model == "lora_r32", "KID_in_domain_mean"] -= 0.000001
        exec(cell(EVAL, "results = metrics.merge"), ns)
        self.assertEqual(ns["FINAL"], "lora_r32")

    def test_training_timing_and_missing_measurements(self):
        tree = ast.parse(cell(TRAIN, "def train(name, rank)"))
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        runs = self.base / "runs"

        def run(cmd, **kw):
            output = next(a.split("=", 1)[1] for a in cmd if a.startswith("--output_dir="))
            (Path(output) / "pytorch_lora_weights.safetensors").touch()
            return types.SimpleNamespace(returncode=0)

        ns = dict(os=os, json=json, threading=threading, time=time, RUNS_DIR=str(runs), SCRIPT="train.py", BASE_MODEL="base", DATA=str(self.base), RES=512, BATCH=1, GRAD_ACCUM=4, STEPS=1500, LR=1e-4, SEED=266,
                  gpu_mem_mb=lambda: 8192, last_progress=lambda p: "", subprocess=types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT),
                  torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_name=lambda i: "test")), load_file=lambda p: {"adapter": torch.zeros(10)})
        exec(compile(ast.Module(body=[train], type_ignores=[]), "train", "exec"), ns)
        result = ns["train"]("lora_r4", 4)
        self.assertEqual(result["peak_gpu_mem_gb"], 8.0)
        self.assertEqual(result["trainable_params"], 10)
        self.assertGreater(json.loads((runs / "lora_r4/timing.json").read_text())["seconds"], 0)
        self.assertEqual(ns["train"]("lora_r4", 4)["peak_gpu_mem_gb"], 8.0)
        (runs / "lora_r4/timing.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "timing.json"):
            ns["train"]("lora_r4", 4)

    def test_clip_embedding_return_formats(self):
        tree = ast.parse(cell(EVAL, "def img_emb"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "img_emb")
        paths = [str(self.base / "p2_data/train/train.png")]

        class Batch:
            pixel_values = torch.ones(1, 3, 32, 32)

            def to(self, device):
                return self

        for structured in [False, True]:
            features = torch.tensor([[3.0, 4.0]])
            value = types.SimpleNamespace(pooler_output=features) if structured else features
            ns = dict(torch=torch, Image=Image, cp=lambda **kw: Batch(), cm=types.SimpleNamespace(get_image_features=lambda **kw: value))
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "embeddings", "exec"), ns)
            torch.testing.assert_close(ns["img_emb"](paths), torch.tensor([[0.6, 0.8]]))

    def test_completed_run_survives_next_run_failure(self):
        source = cell(TRAIN, "def train(name, rank)")
        source = source[source.index("rows = []"):]
        runs, keep, out = self.base / "run_artifacts", self.base / "saved", self.base / "member_outputs"
        out.mkdir()

        def train(name, rank):
            if rank == 32:
                raise RuntimeError("simulated second-run failure")
            folder = runs / name
            folder.mkdir(parents=True)
            (folder / "pytorch_lora_weights.safetensors").touch()
            return dict(run=name, steps=1500)

        ns = dict(train=train, os=os, shutil=shutil, pd=pd, KEEP=str(keep), OUT=str(out), RUNS_DIR=str(runs), RUNS={"lora_r4": 4, "lora_r32": 32})
        with self.assertRaisesRegex(RuntimeError, "second-run failure"):
            exec(source, ns)
        self.assertTrue((keep / "adapters/lora_r4.safetensors").is_file())
        self.assertEqual(pd.read_csv(keep / "outputs/p2_training_runs.csv").run.tolist(), ["lora_r4"])

    def test_human_ratings_completeness(self):
        out = self.base / "ratings"
        out.mkdir()
        key = [dict(pair=i, A="lora_r4", B="base") for i in range(1, 26)]
        ratings = pd.DataFrame([dict(rater=name, pair=i, style_winner="A", prompt_winner="B", is_external=external)
                                for name, external in [("pramod", False), ("manav", False), ("outside", True)] for i in range(1, 26)])
        path = out / "human_ratings.csv"
        ns = dict(glob=glob, pd=pd, OUT=str(out), FINAL="lora_r4", key=key)
        source = cell(EVAL, "ratings_path =")
        ratings.iloc[:-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "same 25 pairs"):
            exec(source, ns)
        ratings.to_csv(path, index=False)
        exec(source, ns)
        summary = pd.read_csv(out / "p2_human_results.csv").iloc[0]
        self.assertEqual(summary.judgements, 75)
        self.assertEqual(summary.style_winner, 1.0)
        self.assertEqual(summary.prompt_winner, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
