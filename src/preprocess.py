"""Text cleaning.

"paper"      follows section 3.2 of the paper: drop punctuation, line breaks, numbers and
             stop words, lowercase, and reduce each word to its root (played -> play).
"paper_nltk" is the same but removes NLTK's full English stop-word list (198 words,
             including "not" and "no") instead of the short list.
"minimal"    only lowercases and removes the HTML line breaks, so negations such as "not"
             and ratings such as "10" survive. Used for the improvement experiments.
"""
import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import PROCESSED_DIR, QUICK_N, SEED
from src.data import load_raw

CLEANING_MODES = ("paper", "paper_nltk", "minimal")

# The paper does not name its stop-word list; its examples ('a', 'the', 'of') come from the
# short list used in the tutorial it cites for its method (A. Kub, refs [5, 6]).
SHORT_STOP_WORDS = frozenset(["in", "of", "at", "a", "the"])

_BREAKS = re.compile(r"<br\s*/?>")
_NON_LETTERS = re.compile(r"[^a-z\s]")
_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=None)
def _nltk_stop_words():
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    # Apostrophes are deleted from the text, so "don't" has to match "dont".
    return frozenset(w.replace("'", "") for w in words)


@lru_cache(maxsize=None)
def _root(word):
    try:
        return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos="v"), pos="n")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos="v"), pos="n")


def clean_text(text, mode="paper"):
    text = _BREAKS.sub(" ", text.lower())
    if mode == "minimal":
        return text
    text = _NON_LETTERS.sub(" ", text.replace("'", ""))
    stop = SHORT_STOP_WORDS if mode == "paper" else _nltk_stop_words()
    return " ".join(_root(word) for word in text.split() if word not in stop)


def load_clean(split, mode="paper", quick=False):
    """Return the cleaned reviews of `split` as a DataFrame with columns movie, label, text.

    The cleaning runs once per (split, mode) and is cached in data/processed/.
    """
    if mode not in CLEANING_MODES:
        raise ValueError(f"mode must be one of {CLEANING_MODES}")
    path = PROCESSED_DIR / f"{split}_{mode}.csv"
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False)
    else:
        raw = load_raw(split)
        df = pd.DataFrame({"movie": raw.movie, "label": raw.label,
                           "text": [clean_text(t, mode) for t in raw.text]})
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    if quick:
        df = df.groupby("label").sample(QUICK_N // 2, random_state=SEED)
    return df.reset_index(drop=True)
