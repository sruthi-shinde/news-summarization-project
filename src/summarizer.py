"""
summarizer.py
-------------
Extractive summarization using TF-IDF.

Idea: every sentence gets an importance score. The highest scoring
sentences are selected and printed in their ORIGINAL order.

Final score of a sentence =
      0.45 * (average TF-IDF weight of its words)
    + 0.45 * (cosine similarity between the sentence and the whole article)
    + 0.10 * (position bonus - earlier sentences get a little more weight)
Very short sentences (less than 6 words) are penalised.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_sentence, split_sentences
from .utils import FriendlyError

# Fraction of the article's sentences that go into the summary
SUMMARY_RATIOS = {"Short": 0.20, "Medium": 0.30, "Long": 0.40}

W_TFIDF, W_CENTROID, W_POSITION = 0.45, 0.45, 0.10


def _min_max(values):
    """Scale numbers to the range 0..1."""
    values = np.asarray(values, dtype=float)
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def score_sentences(sentences):
    """Return an array with one importance score per sentence."""
    processed = [" ".join(preprocess_sentence(s)) for s in sentences]

    # Step 1: TF-IDF (every sentence is treated as a small document)
    try:
        vectorizer = TfidfVectorizer(norm=None)
        matrix = vectorizer.fit_transform(processed)
    except ValueError:
        raise FriendlyError(
            "The article does not contain enough meaningful words to summarise."
        )

    # Step 2: average TF-IDF weight of the words in each sentence
    weight_sums = np.asarray(matrix.sum(axis=1)).ravel()
    n_words = np.array([max(len(p.split()), 1) for p in processed])
    avg_tfidf = _min_max(weight_sums / n_words)

    # Step 3: cosine similarity of each sentence with the article "centre"
    centroid = np.asarray(matrix.mean(axis=0))
    similarity = _min_max(cosine_similarity(matrix, centroid).ravel())

    # Step 4: small position bonus (news usually states key facts early)
    n = len(sentences)
    position = 1.0 - (np.arange(n) / n)

    scores = W_TFIDF * avg_tfidf + W_CENTROID * similarity + W_POSITION * position

    # Step 5: penalise very short sentences
    original_lengths = np.array([len(s.split()) for s in sentences])
    scores = np.where(original_lengths < 6, scores * 0.5, scores)
    return scores


def summarize(text, length="Medium"):
    """
    Create an extractive summary.

    Returns a dictionary with:
        summary   - the summary text
        sentences - all sentences of the article
        scores    - importance score of each sentence
        selected  - indexes of the chosen sentences (original order)
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        raise FriendlyError("No sentences could be found in the article.")

    ratio = SUMMARY_RATIOS.get(length, SUMMARY_RATIOS["Medium"])
    n_select = min(max(2, round(ratio * n)), n)

    scores = score_sentences(sentences)

    # Rank sentences (highest score first) and keep the best n_select
    ranked = np.argsort(scores)[::-1]
    selected = sorted(int(i) for i in ranked[:n_select])   # restore original order

    summary = " ".join(sentences[i] for i in selected)
    return {
        "summary": summary,
        "sentences": sentences,
        "scores": [float(s) for s in scores],
        "selected": selected,
    }
