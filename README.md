# DATA 266 Lab 2: Multimodal RAG, Diffusion Fine-Tuning and Multi-Agent System

Team 17: Manav Patel and Pramod

## Part 1: Multimodal RAG

We build a system that answers questions about three economics PDFs. It searches both the text and the figures, and it gives its sources. We also use it for the Kaggle competition.

## Part 2: Diffusion fine-tuning

We fine-tune Stable Diffusion 1.5 on a small captioned image set. We train two versions and compare them with the base model using scores and human ratings.

## Part 3: Multi-agent system

We build five agents that work together to understand an image and change or create images. A critique agent checks each result and asks for a fix when needed.

## How to run each part

**Hardware rule:** every part runs on a single 16 GB GPU -- a Kaggle T4/P100 for Part 1, a Colab T4
for Parts 2 and 3. No hosted API is used anywhere (all models run locally on the GPU); see
`AI_USAGE.md` for the coding-assistant tools used during development, and each part's README for
the exact model names and versions.

- **Part 1:** open `part1_rag/member_manav/notebooks/p1_rag.ipynb` as a new Kaggle notebook with
  GPU on and the Part 1 competition data attached, then Run All. It reads questions from
  `QUESTIONS_PATH` (default `Questions.csv`) and writes `submission.csv`, end to end, in under 90
  minutes. `part1_rag/member_manav/notebooks/p1_eval.ipynb` reproduces the golden-set and retrieval
  evaluation separately.
- **Part 2:** on a Colab T4, open `part2_diffusion/member_manav/notebooks/p2_finetune.ipynb` and
  Run All to train both LoRA runs, then `p2_generate_eval.ipynb` to generate the fixed prompts and
  run the KID/CLIP/LPIPS/human evaluation.
- **Part 3:** on a Colab T4, open `part3_agents/member_manav/notebooks/p3_agents.ipynb` and Run All
  to build the agents, run the 15 scenarios and 10 red-team cases with and without the critique
  loop, and launch the Gradio demo app.

## Repo layout

```
Data266-Lab2-Team17/
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── CONTRIBUTIONS.md
├── AI_USAGE.md
├── links.txt
├── eval/              golden sets, test prompts, scenarios and red-team cases (per member)
├── tests/
├── part1_rag/
│   ├── member_manav/
│   └── member_pramod/
├── part2_diffusion/
│   ├── member_manav/
│   └── member_pramod/
├── part3_agents/
│   ├── member_manav/
│   └── member_pramod/
├── notebooks/
└── report/
```

Open PLAN.md to learn how to work in this repo.
