"""Small HTML/CSS helpers for the dashboard. Everything uses translucent colours + inherited text colour,
so it reads correctly in Streamlit's light and dark themes."""
from __future__ import annotations

import html
import re

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = {250: "#86b6ef", 400: "#3987e5", 550: "#1c5cab", 700: "#0d366b"}
NEUTRAL = "#b4b2aa"
CLASS_COLOURS = {
    "negative": CAT[1], "neutral": NEUTRAL, "positive": CAT[0],
    "NAG": BLUE[250], "CAG": BLUE[400], "OAG": BLUE[700],
    "not_offensive": BLUE[250], "abusive": BLUE[400], "hate_inducing": BLUE[700],
    "no_hate": BLUE[250], "hate": BLUE[700],
}
NICE = {"not_offensive": "not offensive", "hate_inducing": "hate-inducing", "no_hate": "no hate", "NAG": "non-aggressive (NAG)",
        "CAG": "covertly aggressive (CAG)", "OAG": "overtly aggressive (OAG)"}
HEAD_TITLE = {"sentiment": "Sentiment", "aggression": "Aggression level", "offensive": "Offensiveness", "hate": "Hate speech"}
HEAD_HELP = {
    "sentiment": "How the writer feels (SentiMix tweets).",
    "aggression": "How aggressive the tone is, from calm to openly hostile (TRAC-1 Facebook comments).",
    "offensive": "Is it clean, abusive, or hate-inducing? (HOT tweets).",
    "hate": "Does it attack a group of people? (Bohra et al. tweets).",
}
ACTION = {"allow": ("✓", "Allow", "#0ca30c"), "review": ("⚠", "Review", "#fab219"), "block": ("⛔", "Block", "#d03b3b")}

CSS = """
<style>
.hg-chip{display:inline-flex;align-items:center;gap:.4rem;padding:.18rem .7rem;border-radius:999px;font-weight:600;font-size:.92rem;border:1.5px solid}
.hg-big{font-size:1.9rem;font-weight:700;line-height:1.15;margin:.1rem 0 .15rem 0}
.hg-sub{opacity:.72;font-size:.9rem}
.hg-k{opacity:.7;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;font-weight:600}
.hg-row{display:flex;align-items:center;gap:.6rem;margin:.22rem 0;font-size:.9rem}
.hg-row .l{width:11.5rem;flex:none;opacity:.85}
.hg-row .b{flex:1;height:.62rem;border-radius:.4rem;background:rgba(128,128,128,.18);overflow:hidden}
.hg-row .b>i{display:block;height:100%;border-radius:.4rem}
.hg-row .p{width:3rem;text-align:right;font-variant-numeric:tabular-nums;flex:none}
.hg-tok{display:inline-block;padding:.05rem .32rem;margin:.1rem .08rem;border-radius:.35rem;font-size:1rem;line-height:1.55}
.hg-hi{background:rgba(74,58,167,.20);border-bottom:2px solid #4a3aa7}
.hg-en{background:rgba(27,175,122,.20);border-bottom:2px solid #1baf7a}
.hg-x{opacity:.65}
.hg-prof{outline:2px solid #d03b3b;outline-offset:1px;font-weight:700}
.hg-legend span{display:inline-block;margin-right:1rem;font-size:.85rem}
.hg-sw{display:inline-block;width:.8rem;height:.8rem;border-radius:.2rem;margin-right:.3rem;vertical-align:-.1rem}
.hg-warn{border-left:4px solid #fab219;padding:.5rem .8rem;border-radius:.3rem;background:rgba(250,178,25,.12);font-size:.9rem}
.hg-find{border-left:4px solid #2a78d6;padding:.55rem .9rem;border-radius:.3rem;background:rgba(42,120,214,.09);margin:.4rem 0 .9rem 0}
</style>
"""


def chip(kind: str) -> str:
    icon, label, col = ACTION[kind]
    return f'<span class="hg-chip" style="border-color:{col};background:{col}22">{icon} {label}</span>'


def bars(probs: dict[str, float], order: list[str] | None = None) -> str:
    out = []
    for k in order or list(probs):
        v = probs[k]
        out.append(f'<div class="hg-row"><span class="l">{html.escape(NICE.get(k, k))}</span>'
                   f'<span class="b"><i style="width:{max(v * 100, 0.6):.1f}%;background:{CLASS_COLOURS.get(k, CAT[0])}"></i></span>'
                   f'<span class="p">{v * 100:.0f}%</span></div>')
    return "".join(out)


def legend() -> str:
    return ('<div class="hg-legend"><span><i class="hg-sw" style="background:#4a3aa7"></i>Hindi word (Roman script)</span>'
            '<span><i class="hg-sw" style="background:#1baf7a"></i>English word</span>'
            '<span><i class="hg-sw" style="background:transparent;outline:2px solid #d03b3b"></i>in the profanity list</span></div>')


def language_html(tags: list[tuple[str, str]], lexicon: list[dict]) -> str:
    bad = {re.sub(r"\W", "", h["word"]).lower() for h in lexicon}
    parts = []
    for tok, tag in tags:
        cls = {"hi": "hg-hi", "en": "hg-en"}.get(tag, "hg-x")
        if re.sub(r"\W", "", tok).lower() in bad and tag != "x":
            cls += " hg-prof"
        parts.append(f'<span class="hg-tok {cls}">{html.escape(tok)}</span>')
    return " ".join(parts)


def occlusion_html(pairs: list[tuple[str, float]]) -> str:
    """Blue = the word pushes toward the chosen label, orange = pushes away. Strength = how far the probability moves."""
    if not pairs:
        return ""
    m = max(max(abs(d) for _, d in pairs), 0.05)
    out = []
    for w, d in pairs:
        a = min(abs(d) / m, 1.0) * 0.75
        col = "42,120,214" if d > 0 else "235,104,52"
        out.append(f'<span class="hg-tok" style="background:rgba({col},{a:.2f})" title="{d * 100:+.1f} points">{html.escape(w)}</span>')
    return " ".join(out)
