"""
Named Wordle word lists.

Each name always means the same words in the same order: remote lists are
pinned to an upstream revision and checked against their expected size. That
matters because actions are indexes into the accepted list, so a checkpoint
only works with the list it was trained on. Remote lists are downloaded once
and cached in data/word_lists/ at the repo root.
"""

import os
import urllib.error
import urllib.request
from typing import List

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "word_lists")

# Small vocabulary for smoke tests and demos; too small for real training.
SAMPLE_WORDS = [
    "apple", "beach", "chair", "dance", "eagle",
    "flame", "grape", "house", "igloo", "jelly",
    "knife", "lemon", "magic", "night", "ocean",
    "piano", "queen", "river", "snake", "tiger"
]

# Named lists: either a pinned "url" with its expected "size", or inline "words".
WORD_LISTS = {
    # Guesses the NYT game accepts, as of 2022-09 (includes every solution)
    "tabatkins-14855": {
        "url": "https://raw.githubusercontent.com/tabatkins/wordle-list/"
               "255b9469c4dad99a3b95cc4ddbe139b3d3747868/words",
        "size": 14855,
    },
    # The original Wordle solutions list
    "cfreshman-2315": {
        "url": "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/"
               "c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt",
        "size": 2315,
    },
    "sample-20": {"words": SAMPLE_WORDS},
}

DEFAULT_ACCEPTED = "tabatkins-14855"
DEFAULT_SOLUTIONS = "cfreshman-2315"


def load_word_list(name: str, refresh: bool = False) -> List[str]:
    """Return the named word list, downloading it if it isn't cached (or refresh is set)."""
    if name not in WORD_LISTS:
        raise ValueError(f"Unknown word list '{name}'. Known lists: {', '.join(WORD_LISTS)}")
    source = WORD_LISTS[name]
    if "words" in source:
        return list(source["words"])

    path = os.path.join(CACHE_DIR, f"{name}.txt")
    if refresh or not os.path.exists(path):
        words = _download(name, source["url"])
        _validate(name, words, source["size"])
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            f.write("\n".join(words) + "\n")
        os.replace(path + ".tmp", path)
        return words

    with open(path) as f:
        words = f.read().split()
    _validate(name, words, source["size"])
    return words


def _download(name: str, url: str) -> List[str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8").split()
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Couldn't download word list '{name}' from {url}: {e}") from e


def _validate(name: str, words: List[str], size: int):
    """Fail unless the list has exactly `size` unique lowercase 5-letter words."""
    bad = [w for w in words if len(w) != 5 or not (w.isascii() and w.isalpha() and w.islower())]
    if bad:
        raise ValueError(f"Word list '{name}' has invalid entries, e.g. {bad[:5]}")
    if len(set(words)) != len(words):
        raise ValueError(f"Word list '{name}' has duplicate words")
    if len(words) != size:
        raise ValueError(f"Word list '{name}' has {len(words)} words, expected {size}")
