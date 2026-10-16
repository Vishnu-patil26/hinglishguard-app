"""Hinglish profanity lexicon (Mathur et al., ALW 2018): used as a baseline and for highlighting.

Hinglish has no standard spelling, so matching is done on a canonical form:
lower-case, elongations squeezed ("chuuutiya" -> "chutiya"), obfuscation symbols
("ch*tiya", "m@darchod") treated as wildcards. An optional 'fuzzy' mode also matches on the
consonant skeleton ("chutya" ~ "chutiya"), which catches more spellings and a few false hits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data_sources import profanity_list

_WORD = re.compile(r"[\w*@$#!]+", re.UNICODE)
_ELONG = re.compile(r"(.)\1+")
_VOWELS = set("aeiou")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _canon(w: str) -> str:
    return _ELONG.sub(r"\1", w.lower().translate(_LEET))


def _skeleton(w: str) -> str:
    c = _canon(w)
    sk = "".join(ch for ch in c if ch not in _VOWELS and ch.isalpha())
    return _ELONG.sub(r"\1", sk)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Hit:
    word: str
    start: int
    end: int
    entry: str
    gloss: str
    severity: int
    fuzzy: bool


class ProfanityLexicon:
    def __init__(self, fuzzy: bool = False) -> None:
        self.fuzzy = fuzzy
        self.entries = profanity_list()
        self.by_canon: dict[str, tuple[str, str, int]] = {}
        self.by_skel: dict[str, tuple[str, str, int]] = {}
        for word, gloss, sev in self.entries:
            if not word.isascii():
                continue
            self.by_canon.setdefault(_canon(word), (word, gloss, sev))
            sk = _skeleton(word)
            if len(sk) >= 4:                                   # 3-consonant skeletons collide with everyday words
                self.by_skel.setdefault(sk, (word, gloss, sev))

    def find(self, text: str) -> list[Hit]:
        hits: list[Hit] = []
        for m in _WORD.finditer(text):
            tok = m.group(0)
            if "*" in tok:                                   # ch*tiya -> wildcard regex over the canon keys
                pat = re.compile("^" + re.escape(_canon(tok.replace("*", "\x00"))).replace("\x00", ".") + "$")
                cand = [v for k, v in self.by_canon.items() if pat.match(k)]
                if cand:
                    w, g, s = cand[0]
                    hits.append(Hit(tok, m.start(), m.end(), w, g, s, False))
                continue
            key = _canon(tok)
            if key in self.by_canon:
                w, g, s = self.by_canon[key]
                hits.append(Hit(tok, m.start(), m.end(), w, g, s, False))
            elif self.fuzzy:
                sk = _skeleton(tok)
                if len(sk) >= 4 and sk in self.by_skel:
                    w, g, s = self.by_skel[sk]
                    if _edit_distance(key, _canon(w)) <= 2:     # same consonants AND a close spelling
                        hits.append(Hit(tok, m.start(), m.end(), w, g, s, True))
        return hits

    def score(self, text: str) -> int:
        """0 = no hit, otherwise the highest severity (1-5) found."""
        return max((h.severity for h in self.find(text)), default=0)
