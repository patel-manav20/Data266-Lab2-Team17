# Part 1: Multimodal RAG (Manav)

This part answers questions about three economics PDFs. It uses the text, the tables and the charts, and it cites its sources. The documents are read-only. Values a user gives in a chat are kept only for that chat.

## Workflow

```
PDFs + kb_manifest.json
        |
        v
Text chunks (doc_id, page)  +  Figures and tables (label, caption, crop)
        |                              |
        |                      Chart values: vector paths (DOC1),
        |                      color tracing (DOC2, DOC3), VLM if both fail
        v                              v
ChromaDB (dense) + BM25  <---  figure cards
        |
        v
Question -> conversation memory -> hybrid retrieval (exact label lookup first)
        |
        v
Answer rules -> JSON answer (answer, status, citations, figure_references)
        |
        v
submission.csv
```

Appendix C of DOC3 has trust level low. It is never used as evidence, and any instruction inside a document or image is treated as plain text.

## Tools and models

| Tool or model | Use | License |
|---|---|---|
| pdfplumber | Text, words and positions | MIT |
| pypdfium2 | Page rendering and crops | Apache-2.0 / BSD-3 |
| BAAI/bge-small-en-v1.5 (sentence-transformers) | Text embeddings | MIT |
| rank_bm25 | Keyword search | Apache-2.0 |
| ChromaDB | Vector database | Apache-2.0 |
| Qwen/Qwen3-VL-4B-Instruct (transformers) | Chart reading fallback and answer writing | Apache-2.0 |

No hosted API is used. Seed is 266.

## Notebooks

- `notebooks/p1_rag.ipynb`: the Kaggle notebook. It builds the full index, answers every question and writes `submission.csv`. All code is inside the notebook.
- `notebooks/p1_eval.ipynb`: golden set, baseline (dense retrieval only) vs final system, human ratings and safety examples.

## How to run

On Kaggle:
1. Open `p1_rag.ipynb` as a Kaggle notebook with a T4 GPU.
2. Add the competition data and the two models as inputs.
3. Run All. The notebook reads `QUESTIONS_PATH` (default `Questions.csv`) and writes `submission.csv`.
4. Check it with `python validate_submission.py submission.csv Questions.csv`.

On a local machine:
1. `pip install -r requirements.txt`
2. Put the competition files in one folder and point the notebook to it.
3. Run `p1_rag.ipynb`, then `p1_eval.ipynb`.

Citations use the PDF page index, so `DOC1:p3` means the third page of the PDF file.
