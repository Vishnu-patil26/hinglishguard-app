"""Text clean-up for noisy Hinglish social-media text.

Two levels:
  light  - what the transformers see: fix encoding junk, drop URLs / RT prefixes,
           collapse @mentions, tidy whitespace. Spelling, case and emoji are left alone
           because the pretrained tokenizers already know them.
  full   - light + lower-case + squeeze elongated letters ("haaaan" -> "haan")
           + emoji turned into words. Used by the classical (TF-IDF) models and by the
           normalisation ablation.
"""
from __future__ import annotations

import html
import re
import unicodedata

try:                                     # optional; only needed for emoji -> words
    import emoji as _emoji
except Exception:                        # pragma: no cover
    _emoji = None

_URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.I)
_RT = re.compile(r"^\s*RT\s+@\w+\s*:?\s*", re.I)
_MENTION_RUN = re.compile(r"(?:@\w+[\s,:]*)+")
_HEXESC = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_ZERO_WIDTH = re.compile("[​-‏⁠﻿]")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(\w)\1{2,}")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF"
    "\U0001F900-\U0001F9FF❤♥]"
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def _fix_hex(match: re.Match) -> str:
    """The HOT file contains literal '\\xe2\\x80\\xa6' text where a '...' was cut."""
    try:
        return bytes.fromhex(match.group(0).replace("\\x", "")).decode("utf-8", "ignore")
    except ValueError:
        return " "


def normalise(text: object, level: str = "light", emoji_words: bool = True) -> str:
    s = "" if text is None else str(text)
    s = _HEXESC.sub(_fix_hex, s)
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = _ZERO_WIDTH.sub("", s)
    s = _RT.sub("", s)
    s = _URL.sub(" ", s)
    s = _MENTION_RUN.sub(" @user ", s)
    s = s.replace("…", " ")                       # a bare ellipsis carries nothing
    s = _SPACES.sub(" ", s).strip()
    if level == "full":
        s = s.lower()
        s = s.replace("#", " ")
        if emoji_words and _emoji is not None and any(ord(ch) >= 0x2000 for ch in s):   # demojize is slow: skip emoji-free text
            s = _emoji.demojize(s, delimiters=(" ", " "), language="en").replace("_", " ")
        s = _REPEAT.sub(r"\1\1", s)
        s = _SPACES.sub(" ", s).strip()
    return s


def dedupe_key(text: object) -> str:
    """Two tweets that differ only in URL / mention / punctuation / case share a key."""
    s = normalise(text, "full", emoji_words=False)
    s = s.replace("@user", " ")
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE)


def script_of(text: object) -> str:
    """roman | devanagari | mixed | none (counted over letters only)."""
    s = "" if text is None else str(text)
    dev, lat = len(_DEVANAGARI.findall(s)), len(_LATIN.findall(s))
    if dev + lat == 0:
        return "none"
    share = dev / (dev + lat)
    if share < 0.10:
        return "roman"
    if share > 0.90:
        return "devanagari"
    return "mixed"


def n_emoji(text: object) -> int:
    return len(_EMOJI_RE.findall("" if text is None else str(text)))


def split_words(text: str) -> list[str]:
    """Whitespace words - the unit the explainer switches on and off."""
    return text.split()
