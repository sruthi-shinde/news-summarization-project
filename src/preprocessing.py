"""
preprocessing.py
----------------
Text cleaning and the basic NLP steps:
    cleaning -> sentence tokenization -> word tokenization
    -> stopword removal -> lemmatization
"""

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Extra words that are very common in news but carry no topic information
_EXTRA_STOPWORDS = {
    "said", "says", "say", "also", "would", "could", "may", "might",
    "told", "according", "mr", "mrs", "ms", "dr", "per", "like", "get",
    "new", "crore", "lakh",
}

_stop_words = None
_lemmatizer = None


def _get_stopwords():
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english")) | _EXTRA_STOPWORDS
    return _stop_words


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


# ---------------------------------------------------------------------------
# Step 1: Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Clean raw article text:
      * remove URLs and e-mail addresses
      * fix quotes / dashes and strange characters
      * join broken lines (common in PDFs)
      * make sure every paragraph / headline ends with a full stop
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)                  # "inter-\nnational"
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)         # URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)                  # e-mails

    replacements = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)

    # keep only printable characters (and new lines)
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")

    # work paragraph by paragraph so headlines do not merge with sentences
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        para = " ".join(para.split())                          # remove extra spaces/newlines
        if not para:
            continue
        if para[-1] not in ".!?\"'":
            para += "."
        cleaned.append(para)
    return " ".join(cleaned)


def split_headline(raw_text):
    """
    If the first paragraph looks like a headline (short, no full stop at the
    end, followed by more paragraphs) separate it from the article body.
    Returns (headline, body). The headline is "" when none is found.
    """
    text = raw_text.replace("\r", "\n").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        first = " ".join(paragraphs[0].split())
        if len(first.split()) <= 20 and first[-1] not in ".!?\"'":
            return first, "\n\n".join(paragraphs[1:])
    return "", raw_text


# ---------------------------------------------------------------------------
# Step 2: Sentence tokenization
# ---------------------------------------------------------------------------
def split_sentences(text):
    """Split text into a list of sentences using NLTK."""
    sentences = [s.strip() for s in sent_tokenize(text)]
    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Step 3: Word tokenization
# ---------------------------------------------------------------------------
def tokenize_words(sentence):
    """Split a sentence into lower-case alphabetic words."""
    tokens = word_tokenize(sentence.lower())
    return [t for t in tokens if t.isalpha()]


# ---------------------------------------------------------------------------
# Step 4: Stopword removal
# ---------------------------------------------------------------------------
def remove_stopwords(tokens):
    """Remove very common words (the, is, of ...) and very short words."""
    stop_words = _get_stopwords()
    return [t for t in tokens if t not in stop_words and len(t) > 2]


# ---------------------------------------------------------------------------
# Step 5: Lemmatization
# ---------------------------------------------------------------------------
def lemmatize(tokens):
    """Convert words to their base form (players -> player, running -> run)."""
    lemmatizer = _get_lemmatizer()
    result = []
    for t in tokens:
        base = lemmatizer.lemmatize(t, pos="n")      # noun form
        base = lemmatizer.lemmatize(base, pos="v")   # verb form
        result.append(base)
    return result


# ---------------------------------------------------------------------------
# Complete preprocessing of one sentence
# ---------------------------------------------------------------------------
def preprocess_sentence(sentence):
    """Run tokenization, stopword removal and lemmatization on a sentence."""
    return lemmatize(remove_stopwords(tokenize_words(sentence)))


def show_pipeline_steps(sentence):
    """Return the output of every preprocessing step (used for explanation in the UI)."""
    tokens = tokenize_words(sentence)
    no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(no_stop)
    return {
        "Original sentence": sentence,
        "Word tokens": tokens,
        "After stopword removal": no_stop,
        "After lemmatization": lemmas,
    }
