"""Local copy of the Kaggle row score, used to compare answer designs on dev questions.

score = 0.40 answer + 0.20 status + 0.20 figure_F1 + 0.20 citations
  answer     share of required facts found in the answer: required terms (normalized text match) and
             required numbers (a number in the answer within 1% of the required number)
  status     1 if the status equals the expected status
  figure_F1  F1 between the answer's figure_references and the expected labels (NONE against NONE = 1)
  citations  0.5 x F1 over the documents cited + 0.5 x F1 over the exact citations
             (ASSUMPTION: the Kaggle formula for citations is not public; F1 is our guess)
An expected item: {"required_terms": [...], "required_numbers": [...], "status": "...",
                   "citations": ["DOC1:p22", ...], "figure_references": ["Table 1-2", ...]}
A row: {"answer": "...", "status": "...", "citations": "a,b" or [...], "figure_references": "a,b" or [...]}
"""
import re

ASSUMPTION = "citations = 0.5 x F1(documents) + 0.5 x F1(exact citations), F1 is an assumption"
NUMBER = re.compile(r"(?<![\w.])[-−–]?\d[\d,]*(?:\.\d+)?")


def normalize(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐‑–—−]", "-", text).lower()
    text = re.sub(r"[^a-z0-9%$.\-]+", " ", text)
    text = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", text)
    return " " + " ".join(text.split()) + " "


def as_list(x):
    if isinstance(x, str):
        return [p.strip() for p in x.split(",") if p.strip() and p.strip() != "NONE"]
    return [p for p in (x or []) if p and p != "NONE"]


def f1(got, want):
    got, want = set(got), set(want)
    if not got and not want:
        return 1.0
    if not got or not want:
        return 0.0
    hit = len(got & want)
    return 0.0 if hit == 0 else 2 * hit / (len(got) + len(want))


def numbers_in(text):
    out = []
    for n in NUMBER.findall(text):
        try:
            out.append(float(n.replace("−", "-").replace("–", "-").replace(",", "")))
        except ValueError:
            pass
    return out


def answer_part(item, answer):
    terms, nums = item.get("required_terms", []), item.get("required_numbers", [])
    if not terms and not nums:
        return 1.0
    text, found = normalize(answer), numbers_in(answer)
    hits = sum(normalize(t) in text for t in terms)
    hits += sum(any(abs(v - n) <= max(abs(n) * 0.01, 1e-9) for v in found) for n in nums)
    return hits / (len(terms) + len(nums))


def score_row(item, row):
    cites, want = as_list(row["citations"]), as_list(item.get("citations", []))
    parts = {"answer": answer_part(item, row["answer"]),
             "status": float(row["status"] == item.get("status", "answered")),
             "figure_F1": f1(as_list(row["figure_references"]), as_list(item.get("figure_references", []))),
             "citations": 0.5 * f1({c.split(":")[0] for c in cites}, {c.split(":")[0] for c in want}) + 0.5 * f1(cites, want)}
    parts["score"] = 0.40 * parts["answer"] + 0.20 * parts["status"] + 0.20 * parts["figure_F1"] + 0.20 * parts["citations"]
    return parts
