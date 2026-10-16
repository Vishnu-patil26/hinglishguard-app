"""Word-level language identification (English vs Hindi-in-Roman-script) and code-mixing index.

SentiMix ships gold word-level language tags (Eng / Hin / O). We train a small linear model on
them and use it everywhere else, where no gold tags exist, to measure how code-mixed a text is.
"""
from __future__ import annotations

import re
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

_WORDCH = r"[\wऀ-ॿ]"              # \w alone drops Devanagari vowel signs and splits the word
_TOKEN = re.compile(rf"@{_WORDCH}+|#{_WORDCH}+|https?://\S+|{_WORDCH}+|[^\w\s]", re.UNICODE)
_DEV = re.compile(r"[ऀ-ॿ]")
_ALPHA = re.compile(r"[^\W\d_]")


def int32_csr(X):
    """numpy 2 + scipy 1.17 on Windows hand back int64 sparse indices, which scikit-learn's linear models reject."""
    X = X.tocsr()
    X.indices = X.indices.astype(np.int32, copy=False)
    X.indptr = X.indptr.astype(np.int32, copy=False)
    return X


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _is_independent(tok: str) -> bool:
    """Tokens that belong to no language: mentions, hashtags, urls, numbers, punctuation, emoji."""
    return tok[:1] in "@#" or tok.startswith("http") or not _ALPHA.search(tok)


def _feats(tokens: list[str], i: int) -> dict:
    t = tokens[i].lower()
    f = {"w=" + t: 1.0, "len": min(len(t), 12) / 12.0, "cap": float(tokens[i][:1].isupper())}
    padded = f"^{t}$"
    for n in (1, 2, 3, 4):
        for j in range(len(padded) - n + 1):
            f[f"c{n}={padded[j:j + n]}"] = 1.0
    f["p=" + (tokens[i - 1].lower() if i else "<s>")] = 1.0
    f["n=" + (tokens[i + 1].lower() if i + 1 < len(tokens) else "</s>")] = 1.0
    return f


class WordLID:
    """English-vs-Hindi word tagger. Devanagari words are Hindi by script, not by model."""

    def __init__(self) -> None:
        self.vec = DictVectorizer()
        self.clf = LinearSVC(C=0.3, max_iter=4000)
        self.known: dict[str, str] = {}          # words whose gold tag is (almost) always the same: trusted over the classifier

    def fit(self, sentences: list[list[str]], tags: list[list[str]]) -> "WordLID":
        feats, y = [], []
        counts: dict[str, list[int]] = {}
        for toks, tgs in zip(sentences, tags):
            for i, tg in enumerate(tgs):
                if tg in ("Eng", "Hin") and not _is_independent(toks[i]):
                    feats.append(_feats(toks, i))
                    y.append(1 if tg == "Hin" else 0)
                    counts.setdefault(toks[i].lower(), [0, 0])[1 if tg == "Hin" else 0] += 1
        self.known = {w: ("hi" if h > e else "en") for w, (e, h) in counts.items() if e + h >= 3 and max(e, h) / (e + h) >= 0.9}
        self.clf.fit(int32_csr(self.vec.fit_transform(feats)), np.array(y))
        return self

    def tag_token_lists(self, token_lists: list[list[str]], chunk: int = 50000) -> list[list[str]]:
        """Tag many tokenised texts with a handful of big matrix operations (one call per text is ~100x slower)."""
        out = [["x"] * len(toks) for toks in token_lists]
        where, feats = [], []
        for d, toks in enumerate(token_lists):
            for i, tok in enumerate(toks):
                if _is_independent(tok):
                    continue
                if _DEV.search(tok):
                    out[d][i] = "hi"
                elif tok.lower() in self.known:
                    out[d][i] = self.known[tok.lower()]
                else:
                    where.append((d, i))
                    feats.append(_feats(toks, i))
        for s0 in range(0, len(feats), chunk):
            pred = self.clf.predict(int32_csr(self.vec.transform(feats[s0:s0 + chunk])))
            for (d, i), p in zip(where[s0:s0 + chunk], pred):
                out[d][i] = "hi" if p == 1 else "en"
        return out

    def tag_tokens(self, tokens: list[str]) -> list[str]:
        """Return one of en | hi | x per token."""
        return self.tag_token_lists([tokens])[0]

    def tag(self, text: str) -> list[tuple[str, str]]:
        toks = tokenize(text)
        return list(zip(toks, self.tag_tokens(toks)))

    def tag_batch(self, texts: list[str]) -> list[list[str]]:
        return self.tag_token_lists([tokenize(t) for t in texts])

    def save(self, path) -> None:
        joblib.dump({"vec": self.vec, "clf": self.clf, "known": self.known}, path, compress=3)

    @classmethod
    def load(cls, path) -> "WordLID":
        obj = cls()
        d = joblib.load(path)
        obj.vec, obj.clf, obj.known = d["vec"], d["clf"], d.get("known", {})
        return obj


def cmi(tags: list[str]) -> float:
    """Code-Mixing Index (Gamback & Das 2016): 0 = one language, ~50 = evenly mixed."""
    n = len(tags)
    u = sum(t == "x" for t in tags)
    if n == 0 or n == u:
        return 0.0
    c = Counter(t for t in tags if t != "x")
    return 100.0 * (1.0 - max(c.values()) / (n - u))


def hindi_share(tags: list[str]) -> float:
    c = Counter(t for t in tags if t != "x")
    total = c["en"] + c["hi"]
    return c["hi"] / total if total else 0.0


def mixing_profile(lid: WordLID, text: str) -> dict:
    tags = [t for _, t in lid.tag(text)]
    return {"cmi": cmi(tags), "hindi_share": hindi_share(tags), "n_words": sum(t != "x" for t in tags)}
