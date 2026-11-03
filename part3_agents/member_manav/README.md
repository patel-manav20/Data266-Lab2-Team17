# Part 3: Multi-Agent System (Manav)

This part is a system of five agents. It looks at an input image, turns the user's request into a clear prompt, makes a new image, checks the result and revises it when needed. Domain-style requests use my Part 2 LoRA.

## Workflow

```
User request + image
        |
        v
Planner (route, input safety check) --unsafe--> refuse
        |
        v
Vision Understanding (caption, objects, visible text, answers) --caption or question--> done
        |
        v
Prompt Engineering <-----------------------------+
        |                                        |
        v                                        |
Image Generation (img2img style, edit, Part 2 LoRA)
        |                                        |
        v                                        |
Critique (CLIP score, VLM judgement, safety) --revise, at most 2 times
        |
        v
accept -> final image + JSON trace
```

All agents share one dataclass state. Every request saves a JSON trace with each agent's input, output and time. Text found inside an image is never treated as an instruction.

## Tools and models

| Tool or model | Use | License |
|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | Planner, vision and critique judgement | Apache-2.0 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 + my Part 2 LoRA | Style change (img2img) and domain style | CreativeML OpenRAIL-M |
| timbrooks/instruct-pix2pix | Edits | MIT |
| openai/clip-vit-large-patch14 | CLIP similarity | MIT |
| diffusers safety checker | Output safety check | Apache-2.0 |
| Gradio | Demo app | Apache-2.0 |

No hosted API is used. Seed is 266.

## How to run

1. Use a machine with one NVIDIA GPU (16 GB is enough).
2. `pip install -r requirements.txt`
3. Run `notebooks/p3_agents.ipynb` from top to bottom. The last section starts the Gradio app.
4. Traces are saved in `outputs/traces/`.
