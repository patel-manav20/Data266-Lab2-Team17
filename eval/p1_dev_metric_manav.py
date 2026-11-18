"""Score one answer against one item of p1_dev_conceptual_manav.json.

term_recall: share of required_terms found in the answer (lower case, quotes, dashes and punctuation normalized)
page_hit:    1 if a source page (DOCn:pN) is among the cited pages, else 0
status_ok:   1 if the answer status equals the expected status, else 0
"""
import re


def normalize(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐‑–—−]", "-", text).lower()
    text = re.sub(r"[^a-z0-9%$.\-]+", " ", text)
    text = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", text)
    return " " + " ".join(text.split()) + " "


def score_answer(item, answer, citations, status):
    """answer: text; citations: list or comma string like "DOC1:p2,DOC1:Table 1-2"; status: answer status."""
    text = normalize(answer)
    terms = item["required_terms"]
    found = [t for t in terms if normalize(t) in text]
    if isinstance(citations, str):
        citations = [c for c in citations.split(",") if c and c != "NONE"]
    pages = {f'{s["doc"]}:p{s["pdf_page"]}' for s in item["source"]}
    return {"term_recall": len(found) / len(terms), "missing_terms": [t for t in terms if t not in found],
            "page_hit": int(any(c.strip() in pages for c in citations)),
            "status_ok": int(status == item.get("status", "answered"))}
