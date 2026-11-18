# DATA 266 Lab 2: Multimodal RAG, Diffusion Fine-Tuning and Multi-Agent System

Team 17: Manav Patel and Pramod

## Part 1: Multimodal RAG

We build a system that answers questions about three economics PDFs. It searches both the text and the figures, and it gives its sources. We also use it for the Kaggle competition.

## Part 2: Diffusion fine-tuning

We fine-tune Stable Diffusion 1.5 on a small captioned image set. We train two versions and compare them with the base model using scores and human ratings.

## Part 3: Multi-agent system

We build five agents that work together to understand an image and change or create images. A critique agent checks each result and asks for a fix when needed.

## Results at a glance

Details, how to run and the build commands are in each folder's README.

| Part | Notebook | Main result |
|---|---|---|
| 1 | `part1_rag/member_pramod/notebooks/p1_rag.ipynb` (Kaggle T4) | Golden set: 22/25 correct and 25/25 statuses, against 17/25 and 22/25 for the dense-only baseline. Kaggle public leaderboard 0.868. |
| 2 | `part2_diffusion/member_pramod/notebooks/p2_0{1,2}_*.ipynb` (Colab T4) | In-domain KID: base 0.102, LoRA r4 0.044, LoRA r32 0.031 (final). Highest nearest-neighbour cosine 0.83, so no memorization. |
| 3 | `part3_agents/member_pramod/notebooks/p3_agents.ipynb` (Colab T4) | 5 agents with a critique loop, JSON traces and a Gradio app. Task success 12/15 with critique against 11/15 without; red-team attacks succeeded 0/10. |

Links to the models, the datasets and the Kaggle notebook are in `links.txt`. Test sets are in `tests/` and were committed before tuning.

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
