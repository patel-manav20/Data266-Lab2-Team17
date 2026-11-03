# DATA 266 Lab 2: Multimodal RAG, Diffusion Fine-Tuning and Multi-Agent System

Team 17: Manav Patel and Pramod

## Part 1: Multimodal RAG

We build a system that answers questions about three economics PDFs. It searches both the text and the figures, and it gives its sources. We also use it for the Kaggle competition.

## Part 2: Diffusion fine-tuning

We fine-tune Stable Diffusion 1.5 on a small captioned image set. We train two versions and compare them with the base model using scores and human ratings.

## Part 3: Multi-agent system

We build five agents that work together to understand an image and change or create images. A critique agent checks each result and asks for a fix when needed.

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
