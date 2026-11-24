"""
keyword_extractor.py
--------------------
Keyword extraction with TF-IDF.

1. Every sentence is treated as a small document.
2. TF-IDF weights of each (lemmatized) word are added up over the article.
3. Two-word phrases (e.g. "machine learning") are found from words that
   appear next to each other in the article at least twice.
4. The best scoring words / phrases are returned as keywords.
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import (lemmatize, preprocess_sentence, remove_stopwords,
                            split_sentences, tokenize_words)
from .utils import pretty_keyword, word_count


def get_keyword_count(n_words):
    """Longer articles get more keywords (between 5 and 15)."""
    return int(min(15, max(5, n_words // 35)))


def _unigram_scores(documents):
    """Sum of TF-IDF weights of every word; returns {lemma: score}."""
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(documents)
    except ValueError:               # empty vocabulary
        return {}
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    return dict(zip(vectorizer.get_feature_names_out(), totals))


def _surface_forms(sentences):
    """
    Map each lemma back to the word form used most often in the article
    (e.g. 'learn' -> 'learning', 'striker' -> 'strikers') so keywords look natural.
    """
    forms = {}
    for sentence in sentences:
        for token in remove_stopwords(tokenize_words(sentence)):
            lemma = lemmatize([token])[0]
            forms.setdefault(lemma, Counter())[token] += 1
    return {lemma: counter.most_common(1)[0][0] for lemma, counter in forms.items()}


def _phrase_candidates(sentences, unigram_scores):
    """Find repeated pairs of neighbouring content words, e.g. 'machine learning'."""
    counts = Counter()
    for sentence in sentences:
        previous = None
        for token in tokenize_words(sentence):
            if remove_stopwords([token]):             # a content word
                lemma = lemmatize([token])[0]
                if previous is not None:
                    counts[(previous, lemma)] += 1
                previous = lemma
            else:                                      # stopword breaks the phrase
                previous = None

    phrases = {}
    for (first, second), count in counts.items():
        if count < 2 or first == second:
            continue
        s1, s2 = unigram_scores.get(first, 0), unigram_scores.get(second, 0)
        if min(s1, s2) == 0 or min(s1, s2) / max(s1, s2) < 0.6:
            continue                                   # one word is much weaker
        phrases[f"{first} {second}"] = (s1 + s2) / 2 * 1.2
    return phrases


def extract_keywords(text, top_n=None):
    """Return a list of (keyword, score) pairs, best keyword first."""
    sentences = split_sentences(text)
    documents = [" ".join(preprocess_sentence(s)) for s in sentences]
    documents = [d for d in documents if d.strip()]
    if not documents:
        return []

    if top_n is None:
        top_n = get_keyword_count(word_count(text))

    unigrams = _unigram_scores(documents)
    candidates = dict(unigrams)
    candidates.update(_phrase_candidates(sentences, unigrams))
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)

    selected = []                                      # list of (term, score)
    for term, score in ranked:
        if len(selected) >= top_n:
            break
        if " " in term:
            words = set(term.split())                  # phrase replaces its single words
            selected = [(t, s) for (t, s) in selected if t not in words]
        elif any(" " in t and term in t.split() for t, _ in selected):
            continue                                   # already inside a chosen phrase
        selected.append((term, score))

    selected = sorted(selected, key=lambda item: item[1], reverse=True)[:top_n]

    surface = _surface_forms(sentences)
    keywords = []
    for term, score in selected:
        display = " ".join(surface.get(word, word) for word in term.split())
        keywords.append((pretty_keyword(display, text), round(float(score), 3)))
    return keywords
