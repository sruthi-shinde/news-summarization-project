"""
ner.py
------
Named Entity Recognition (NER) with spaCy's pretrained English model,
plus a few simple rules that clean up the model's output.
"""

import re
from collections import Counter

from .utils import FriendlyError

MODEL_NAME = "en_core_web_sm"

# spaCy label  ->  group name shown in the dashboard
LABEL_TO_GROUP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "DATE": "Date",
    "MONEY": "Money",
    "EVENT": "Event",
}

GROUPS = ["Person", "Organization", "Location", "Date", "Money", "Event"]

# Common job words that the model sometimes wrongly tags as names
_NOT_NAMES = {"player", "captain", "coach", "minister", "president", "doctor",
              "professor", "company", "founder", "leader", "spokesperson"}

_MONTHS = {"january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"}

# Rule 1: a name after a title (Dr., Professor ...) is a person
_TITLE_PATTERN = re.compile(
    r"\b(?:Dr|Prof|Professor|Mr|Mrs|Ms|Shri|Smt)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
# Rule 2: money written like "Rs 5,000 crore", "$2 million", "INR 40 lakh"
_MONEY_PATTERN = re.compile(
    r"(?:Rs\.?|INR|\u20b9|\$|USD|\u20ac)\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s(?:crore|lakh|thousand|million|billion|trillion))?",
    re.IGNORECASE,
)


def load_spacy_model():
    """Load the spaCy model, with a friendly message if it is not installed."""
    try:
        import spacy
    except ImportError:
        raise FriendlyError("spaCy is not installed. Run: pip install spacy")
    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        raise FriendlyError(
            f"The spaCy model '{MODEL_NAME}' is not installed. "
            f"Run this command in the terminal: python -m spacy download {MODEL_NAME}"
        )


def _normalise(entity_text):
    """Tidy an entity string (remove extra spaces and a leading 'the')."""
    value = " ".join(entity_text.split()).strip(" ,.;:'\"")
    if value.lower().startswith("the "):
        value = value[4:]
    return value


def _words(value):
    return {w.strip(".,;:'\"").lower() for w in value.split()}


def _is_real_date(value):
    """Keep dates that contain a number or a month name (drops 'day', 'season' ...)."""
    return any(ch.isdigit() for ch in value) or bool(_words(value) & _MONTHS)


def _merge_short_forms(items):
    """
    Merge short forms into the full name inside one group, e.g.
    'Joshi' -> 'Vikram Joshi', 'Strikers' -> 'Mumbai Strikers'.
    `items` is a dict {lower_text: [display_text, count]}.
    """
    keys = sorted(items, key=lambda k: len(k.split()))
    for short in keys:
        if short not in items:
            continue
        short_words = _words(items[short][0])
        longer = [k for k in items
                  if k != short and len(k.split()) > len(short.split())
                  and short_words <= _words(items[k][0])]
        if longer:
            best = max(longer, key=lambda k: items[k][1])
            items[best][1] += items[short][1]
            del items[short]
    return items


def extract_entities(text, nlp):
    """
    Find named entities.

    Returns a dictionary like:
        {"Person": [("Ananya Deshmukh", 2), ...], "Organization": [...], ...}
    Each entity is a (name, number_of_mentions) pair, most frequent first.
    """
    doc = nlp(text)

    # 1. collect entities found by spaCy
    groups = {g: {} for g in GROUPS}                   # group -> {lower: [text, count]}

    def add(group, value, count=1):
        value = _normalise(value)
        if len(value) < 2:
            return
        entry = groups[group].setdefault(value.lower(), [value, 0])
        entry[1] += count

    for ent in doc.ents:
        group = LABEL_TO_GROUP.get(ent.label_)
        if group is None:
            continue
        if group == "Date" and not _is_real_date(ent.text):
            continue
        if group == "Person" and ent.text.strip().lower() in _NOT_NAMES:
            continue
        add(group, ent.text)

    # 2. add names that follow a title and money written with Rs / $
    for match in _TITLE_PATTERN.finditer(text):
        add("Person", match.group(1))
    for match in _MONEY_PATTERN.finditer(text):
        add("Money", match.group(0))

    # 3. a person's name must not also be listed as an organization / location
    person_words = set()
    for key, (display, _) in groups["Person"].items():
        person_words |= _words(display)
    for group in ("Organization", "Location"):
        for key in list(groups[group]):
            if _words(groups[group][key][0]) <= person_words:
                del groups[group][key]

    # 4. merge short forms and sort by frequency
    result = {}
    for group in GROUPS:
        merged = groups[group] if group == "Date" else _merge_short_forms(groups[group])
        ordered = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        result[group] = [(display, count) for display, count in ordered]
    return result


def count_entities(entities):
    """Total number of unique entities in all groups."""
    return sum(len(items) for items in entities.values())
