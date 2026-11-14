# Part 3: Multi-Agent System (Manav)

A system of five agents. It looks at an input image, routes and safety-checks the
request, turns it into a clear prompt, makes a new image, checks the result and
revises it when needed. Domain-style requests use my Part 2 LoRA
(`itzmnv/data266-lab2-sd15-pokemon-lora-r32`). Every model is open -- no hosted API.

**Status: drafted, not yet run on a GPU.** The routing, safety, prompt-engineering,
critique and orchestration logic in `src/` is fully stub-tested on a laptop (no model
downloads, no GPU -- see "Tests" below). Model loading, the real agent wrappers and the
Gradio app are written and reviewed in `notebooks/p3_agents.ipynb` but have not run yet.

## Workflow

```
User request + image
        |
        v
Planner (route, input safety check) --unsafe/broken--> refuse, clear message
        |
        v
Vision Understanding (caption, objects, visible text, answers) --caption or question--> done
        |
        v
Prompt Engineering <-----------------------------+
        |                                        |
        v                                        |
Image Generation (SD 1.5 img2img, instruct-pix2pix, SD 1.5 + Part 2 LoRA)
        |                                        |
        v                                        |
Critique (CLIP score, VLM judgement, safety) --revise, at most 2 times
        |
        v
accept -> final image + JSON trace
```

All agents share one pydantic `AgentState` (`src/state.py`), passed by plain Python
functions (`src/orchestrator.py`) -- no agent framework. Every agent call is timed into
the trace; every request saves a JSON trace, including failures. Visible text found in
an input image is never treated as an instruction: ordinary text is quoted as image
content in the generation prompt, and text flagged as instruction-like is kept out of
the prompt entirely (see `src/prompt_engineering.py`).

## Memory plan (one 16 GB Colab T4)

| loaded | when | approx. size (fp16) |
|---|---|---|
| Qwen3-VL-4B-Instruct | once, kept resident | ~8 GB |
| CLIP ViT-L/14 | once, kept resident | ~0.6 GB |
| one diffusion pipeline (SD 1.5 img2img/text2img, or instruct-pix2pix) | only while its route is used, `enable_model_cpu_offload()`, freed before loading another | ~2-3 GB resident at a time |

Two diffusion pipelines are never loaded together (`get_pipeline()` in
`notebooks/p3_agents.ipynb` frees the previous one first).

## Tools and models

| Tool or model | Use | License |
|---|---|---|
| Qwen/Qwen3-VL-4B-Instruct | Vision Understanding, the Planner's fallback classifier, Critique's yes/no judgement | Apache-2.0 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | Style change (img2img) and domain style (img2img or text2img) | CreativeML OpenRAIL-M |
| itzmnv/data266-lab2-sd15-pokemon-lora-r32 (my Part 2 LoRA) | Domain-style generation | CreativeML OpenRAIL-M / CC BY-NC-SA 4.0 |
| timbrooks/instruct-pix2pix | Edits | MIT |
| openai/clip-vit-large-patch14 | Critique's CLIP similarity score | MIT |
| OpenCV (bundled Haar cascade) | The Planner's real-person check | Apache-2.0 |
| diffusers safety checker | Output NSFW check, every route | Apache-2.0 |
| Gradio | Demo app | Apache-2.0 |

No hosted API is used. Seed is 266 throughout.

## Tests

`src/test_stub.py` runs the real routing/safety/prompt-engineering/critique/
orchestration logic against all 15 scenarios (`eval/p3_scenarios_manav.json`) and all 10
red-team cases (`eval/p3_redteam_manav.json`) with plain Python stand-ins for every
model call -- no GPU, no downloads. Run it with:

```
python -m src.test_stub --images-dir <path to the images built by src/build_test_images.py>
```

## How to run (once a GPU session is available -- not yet, per the lab stage this was written in)

1. `colab new -s <session> --gpu T4 --high-mem`
2. Upload the code: `colab upload -s <session> src /content/src`,
   `colab upload -s <session> eval/p3_scenarios_manav.json /content/eval/p3_scenarios_manav.json`,
   and the same for `eval/p3_redteam_manav.json`.
3. Upload the notebook and run it with papermill, same pattern as Parts 1 and 2.
4. The last section starts the Gradio app with `share=True`.

Traces are saved under `/content/p3_outputs/traces/` during a run; test images build
into `/content/p3_images/` (reproducible, from `src/build_test_images.py`) and are
published to a public Hugging Face dataset in a later stage, never committed here.
