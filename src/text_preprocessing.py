"""Query/document text preprocessing aligned with the course checklist."""

from __future__ import annotations

import ast
import string
from typing import Any, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_ENSURED = False


def ensure_nltk() -> None:
    global _ENSURED
    if _ENSURED:
        return
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _ENSURED = True


def preprocess_text(raw: str) -> List[str]:
    """
    Lowercase, strip punctuation, remove stopwords, tokenize, stem.
    Matches typical pipeline used for `cleaned_text` in tokenized_dataset.csv.
    """
    ensure_nltk()
    text = raw.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stops = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    out: List[str] = []
    for t in tokens:
        if not t.isalpha() or t in stops:
            continue
        out.append(stemmer.stem(t))
    return out


def parse_cleaned_tokens(cell: Any) -> List[str]:
    """Parse CSV `cleaned_text` column (stringified Python list)."""
    if isinstance(cell, list):
        return [str(t) for t in cell]
    if cell is None or (isinstance(cell, float) and cell != cell):  # NaN from CSV
        return []
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        val = ast.literal_eval(cell.strip())
    except (SyntaxError, ValueError):
        return []
    if isinstance(val, list):
        return [str(t) for t in val]
    return []
