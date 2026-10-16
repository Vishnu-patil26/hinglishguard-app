"""Where everything lives. Every other module imports its paths from here.

HG_DATA / HG_MODELS / HG_OUT can redirect the processed data, models and outputs (the smoke test uses this to run
the whole pipeline on a tiny dataset without touching the real results)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ["HG_DATA"]) if os.environ.get("HG_DATA") else ROOT / "data"
RAW = ROOT / "data" / "raw"
PROC = DATA / "processed"
MODELS = Path(os.environ["HG_MODELS"]) if os.environ.get("HG_MODELS") else ROOT / "models"
BUNDLE = MODELS / "hinglish_guard"          # the trained model the dashboard loads
OUT = Path(os.environ["HG_OUT"]) if os.environ.get("HG_OUT") else ROOT / "outputs"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
DOCS = ROOT / "docs"
RESOURCES = Path(__file__).resolve().parent / "resources"

for _p in (RAW, PROC, MODELS, BUNDLE, TABLES, FIGS, DOCS, OUT / "runs", OUT / "preds"):
    _p.mkdir(parents=True, exist_ok=True)
