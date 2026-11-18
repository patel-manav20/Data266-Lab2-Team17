# Contributions

## Part 1

**Manav:** Built the whole multimodal RAG pipeline in `part1_rag/member_manav/notebooks/p1_rag.ipynb` --
reading the three PDFs into text chunks, tables and figure crops; building a fact store from the tables
and a resolver that picks between a conflicting, revised or forecast-vs-outturn value; hybrid retrieval
(bge-base embeddings + BM25, fused, then reranked with Qwen3-Reranker-4B); reading chart values from
vector paths, color tracing or Qwen3-VL-named series; the answer rules that decide status, citations and
figure references with no language model; and the refusal rules (a name not in the corpus, a request to
change the documents, a false premise). Measured and rejected model-written text answers, HyDE/query
expansion, a model-picked anchor sentence and a model-written lead sentence (see "What we tried and why it
is off" in the notebook) before freezing the logic. Built and ran `p1_eval.ipynb` (golden set, retrieval
Recall@k/MRR, safety counters, the groundedness rating sheets) and wrote `part1_rag/member_manav/README.md`.
Pushed and tracked every measured change through the Kaggle CLI (see the version history in `p1_eval.ipynb`).

## Part 2

**Manav:** Built the Part 2 fine-tuning and evaluation pipeline in `part2_diffusion/member_manav/notebooks/`
-- `p2_finetune.ipynb` (dataset cleaning and dedup, the 90/10 seed-266 split, two LoRA fine-tuning
runs of Stable Diffusion 1.5 on Pokemon BLIP captions at rank 8 and rank 32, and pushing both to
Hugging Face with model cards) and `p2_generate_eval.ipynb` (the 20 fixed prompts across in-domain,
out-of-domain and compositional, generation for the base model and both runs at two seeds, KID and
CLIP scoring, LPIPS diversity, the blinded 3-rater human-evaluation pack, the memorization check,
license and NSFW-filter writeups, and the final-run selection rule). Wrote `eval/p2_prompts_manav.json`
and `eval/p2_prompts_manav_ood_comp.json` before training, and `part2_diffusion/member_manav/README.md`.

## Part 3

**Manav:** Built the five-agent system in `part3_agents/member_manav/src/` (Planner, Vision
Understanding, Prompt Engineering, Image Generation using the Part 2 LoRA for domain-style requests,
and Critique) and the orchestration layer in `orchestrator.py` (shared pydantic state, a 2-revision
cap, a JSON trace per request, and a fair with/without-critique comparison that prepares each
scenario's upstream state once and branches into two independent runs). Wrote the 15 test scenarios
and 10 red-team cases in `eval/p3_scenarios_manav.json` and `eval/p3_redteam_manav.json` before
running them, the Gradio demo app, and `part3_agents/member_manav/notebooks/p3_agents.ipynb`
(executed end to end on a Colab T4). Wrote 162 Mac-side stub tests (`src/test_stub.py`, no GPU
required) covering routing, safety refusals, the revision loop, unsafe-attempt discarding, and the
fairness guarantee between the critique-on and critique-off runs.

## Meeting recordings

| Date | Topic | Link |
|---|---|---|
