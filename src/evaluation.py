"""
evaluation.py  (OPTIONAL module - the main app does not depend on it)
-------------
A small, from-scratch implementation of ROUGE-1, ROUGE-2 and ROUGE-L.

ROUGE compares a system summary with a human-written reference summary.
    ROUGE-1 : overlap of single words (unigrams)
    ROUGE-2 : overlap of two-word sequences (bigrams)
    ROUGE-L : longest common subsequence of words
Each score is reported as Precision, Recall and F1.

Run from the project folder:   python -m src.evaluation
"""

import json
import re
from collections import Counter
from pathlib import Path


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap, candidate_total, reference_total):
    precision = overlap / candidate_total if candidate_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def rouge_n(reference, candidate, n=1):
    ref, cand = _ngrams(_tokens(reference), n), _ngrams(_tokens(candidate), n)
    overlap = sum((ref & cand).values())
    return _prf(overlap, sum(cand.values()), sum(ref.values()))


def _lcs_length(a, b):
    """Length of the longest common subsequence (dynamic programming)."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference, candidate):
    ref, cand = _tokens(reference), _tokens(candidate)
    return _prf(_lcs_length(ref, cand), len(cand), len(ref))


def evaluate(reference, candidate):
    """Return ROUGE-1, ROUGE-2 and ROUGE-L scores."""
    return {
        "ROUGE-1": rouge_n(reference, candidate, 1),
        "ROUGE-2": rouge_n(reference, candidate, 2),
        "ROUGE-L": rouge_l(reference, candidate),
    }


if __name__ == "__main__":
    # Evaluate the summarizer on the sample articles using the
    # human-written reference summaries in data/reference_summaries.json
    from .preprocessing import clean_text
    from .summarizer import summarize
    from .utils import ensure_nltk_resources

    ensure_nltk_resources()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    references = json.loads((data_dir / "reference_summaries.json").read_text(encoding="utf-8"))

    for filename, reference in references.items():
        text = clean_text((data_dir / filename).read_text(encoding="utf-8"))
        for length in ("Short", "Medium", "Long"):
            summary = summarize(text, length)["summary"]
            scores = evaluate(reference, summary)
            line = "  ".join(f"{k}: F1={v['f1']:.3f}" for k, v in scores.items())
            print(f"{filename:32s} {length:6s} {line}")
