"""One look for every static figure (README, report). Colours follow a validated categorical order."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import FIGS  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical slots 1-8 (fixed order, never cycled)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = {250: "#86b6ef", 400: "#3987e5", 550: "#1c5cab", 700: "#0d366b"}    # ordinal ramp for ordered levels
NEUTRAL = "#b4b2aa"

# colour per class, so a class keeps its colour in every chart
CLASS_COLOURS = {
    "negative": CAT[1], "neutral": NEUTRAL, "positive": CAT[0],
    "NAG": BLUE[250], "CAG": BLUE[400], "OAG": BLUE[700],
    "not_offensive": BLUE[250], "abusive": BLUE[400], "hate_inducing": BLUE[700],
    "no_hate": BLUE[250], "hate": BLUE[700],
}
TASK_TITLES = {"sentiment": "Sentiment", "aggression": "Aggression", "offensive": "Offensiveness", "hate": "Hate speech"}
DATASET_TITLES = {"sentiment": "SentiMix tweets", "aggression": "TRAC-1 Facebook", "offensive": "HOT tweets",
                  "hate": "Bohra hate tweets", "youtube": "YouTube comments"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    })


def save(fig, name: str) -> str:
    path = FIGS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return str(path)
