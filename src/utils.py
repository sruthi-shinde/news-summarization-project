"""
utils.py
--------
Small helper functions and shared settings used by the whole project.
"""

import re

import nltk

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MIN_WORDS = 50          # an article shorter than this is "too short"
MIN_SENTENCES = 3       # we need at least 3 sentences to rank them
MAX_CHARS = 200_000     # protects the app from extremely large inputs


class FriendlyError(Exception):
    """An error whose message is safe and helpful to show to the user."""


# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------
# (path used to check if present, name used to download)
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]
_nltk_ready = False


def _is_available(path):
    """True if the NLTK resource exists (as a folder or as a .zip file)."""
    for candidate in (path, path + ".zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources():
    """
    Make sure the NLTK data we need is installed.
    Missing items are downloaded automatically. If that is impossible
    (for example no internet), a FriendlyError is raised.
    """
    global _nltk_ready
    if _nltk_ready:
        return

    missing = set()
    for path, name in _NLTK_RESOURCES:
        if _is_available(path):
            continue
        try:
            nltk.download(name, quiet=True)
        except Exception:
            pass
        if not _is_available(path):
            missing.add(name)

    # The sentence tokenizer needs EITHER 'punkt' OR 'punkt_tab'
    # (depends on the NLTK version), so only complain if both are missing.
    problems = []
    if "punkt" in missing and "punkt_tab" in missing:
        problems.append("punkt")
    for essential in ("stopwords", "wordnet"):
        if essential in missing:
            problems.append(essential)

    if problems:
        raise FriendlyError(
            "Required NLTK resources are missing: "
            + ", ".join(problems)
            + ". Please connect to the internet or run:\n"
            "python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4"
        )
    _nltk_ready = True


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def word_count(text):
    """Number of words in a text (simple whitespace split)."""
    return len(text.split())


def validate_text(text):
    """Raise a FriendlyError if the article cannot be analysed."""
    from .preprocessing import split_sentences  # local import avoids a cycle

    if not text or not text.strip():
        raise FriendlyError("The input is empty. Please paste or upload a news article.")
    if len(text) > MAX_CHARS:
        raise FriendlyError(
            f"The article is too long ({len(text):,} characters). "
            f"Please use an article with fewer than {MAX_CHARS:,} characters."
        )
    if word_count(text) < MIN_WORDS:
        raise FriendlyError(
            f"The article is too short ({word_count(text)} words). "
            f"Please provide at least {MIN_WORDS} words for a meaningful summary."
        )
    if len(split_sentences(text)) < MIN_SENTENCES:
        raise FriendlyError(
            f"The article needs at least {MIN_SENTENCES} sentences to be summarised."
        )


def read_txt_file(uploaded_file):
    """Read an uploaded .txt file (Streamlit UploadedFile) and return text."""
    name = getattr(uploaded_file, "name", "")
    if not name.lower().endswith(".txt"):
        raise FriendlyError("Unsupported file type. Please upload a .txt file.")
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise FriendlyError("Could not read the text file. Please save it with UTF-8 encoding.")


def pretty_keyword(keyword, original_text):
    """
    Make a keyword look nice. Words that appear in capitals in the article
    (like 'AI' or 'ISRO') stay in capitals, others become Title Case.
    """
    pretty_words = []
    for word in keyword.split():
        upper = word.upper()
        mixed = re.search(r"\b" + re.escape(word) + r"\b", original_text, re.IGNORECASE)
        if len(word) <= 5 and re.search(r"\b" + re.escape(upper) + r"\b", original_text):
            pretty_words.append(upper)                      # AI, ISRO
        elif mixed and any(c.isupper() for c in mixed.group(0)[1:]):
            pretty_words.append(mixed.group(0))              # GramAI
        else:
            pretty_words.append(word.capitalize())
    return " ".join(pretty_words)
