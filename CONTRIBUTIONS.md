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

## Part 3

## Meeting recordings

| Date | Topic | Link |
|---|---|---|
