"""Download, parse and split the five real Hinglish datasets used in this project.

Nothing here is generated or simulated: every row comes from a published research corpus.
See DATA.md for provenance and licences.

  sentiment   SentiMix (SemEval-2020 Task 9), Hinglish tweets      neg / neu / pos     official train/dev/test
  aggression  TRAC-1 Hindi shared task, Facebook comments           NAG / CAG / OAG     official train/dev/test(+Twitter test)
  offensive   HOT (Mathur et al., ALW 2018), Hinglish tweets        clean/abusive/hate  random 70/10/20
  hate        Bohra et al. 2018 Hindi-English hate tweets           no / yes            random 70/10/20
  youtube     Hinglish YouTube comments (CC-BY-4.0)                 neg / neu / pos     evaluation only (out-of-domain)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .paths import PROC, RAW, RESOURCES
from .textnorm import dedupe_key, n_emoji, normalise, script_of

SENTIMIX = "https://huggingface.co/datasets/RTT1/SentiMix/resolve/main/"
TRAC = "https://raw.githubusercontent.com/victor7246/Hinglish_Hate_Detection/HEAD/data/raw/trac1-dataset/"
VICTOR = "https://raw.githubusercontent.com/victor7246/Hinglish_Hate_Detection/HEAD/data/raw/"
HOT = "https://raw.githubusercontent.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/HEAD/Data/HOT_dataset.csv"
PROFANITY = "https://raw.githubusercontent.com/pmathur5k10/Hinglish-Offensive-Text-Classification/HEAD/Hinglish_Profanity_List.csv"
YOUTUBE = "https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset/resolve/main/yt_hinglish_comments_dataset.csv"

FILES = {
    "sentimix/train.txt": SENTIMIX + "train_14k_split_conll.txt",
    "sentimix/dev.txt": SENTIMIX + "dev_3k_split_conll.txt",
    "sentimix/test_tokens.txt": SENTIMIX + "Hindi_test_unalbelled_conll_updated.txt",
    "sentimix/test_labels.txt": SENTIMIX + "test_labels_hinglish.txt",
    "trac/train.csv": TRAC + "hindi/agr_hi_train.csv",
    "trac/dev.csv": TRAC + "hindi/agr_hi_dev.csv",
    "trac/test_fb.csv": TRAC + "trac-gold-set/agr_hi_fb_gold.csv",
    "trac/test_tw.csv": TRAC + "trac-gold-set/agr_hi_tw_gold.csv",
    "hot/HOT_dataset.csv": HOT,
    "hot/Hinglish_Profanity_List.csv": PROFANITY,
    "bohra/hate_speech.tsv": VICTOR + "Hate-speech-dataset/hate_speech.tsv",
    "youtube/comments.csv": YOUTUBE,
}

TASKS: dict[str, list[str]] = {
    "sentiment": ["negative", "neutral", "positive"],
    "aggression": ["NAG", "CAG", "OAG"],
    "offensive": ["not_offensive", "abusive", "hate_inducing"],
    "hate": ["no_hate", "hate"],
}
TASK_SOURCE = {"sentiment": "sentimix", "aggression": "trac", "offensive": "hot", "hate": "bohra"}
SEED = 42


# --------------------------------------------------------------------------- download
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str, url: str, tries: int = 4) -> Path:
    dest = RAW / rel
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hinglish-guard/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
                while True:
                    block = r.read(1 << 16)
                    if not block:
                        break
                    f.write(block)
            dest.with_suffix(dest.suffix + ".part").replace(dest)
            return dest
        except Exception as exc:                       # network hiccup -> retry
            last = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def download_all(verbose: bool = True) -> dict[str, Path]:
    paths, manifest = {}, {}
    for rel, url in FILES.items():
        if verbose:
            print(f"  fetching {rel}", flush=True)
        p = fetch(rel, url)
        paths[rel] = p
        manifest[rel] = {"url": url, "bytes": p.stat().st_size, "sha256": _sha256(p)}
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- parsers
_NO_SPACE_BEFORE = set(".,!?;:)]}%_")
_NO_SPACE_AFTER = set("([{@#$_")


def detokenise(tokens: list[str]) -> str:
    """Re-glue SentiMix's one-token-per-line text into something a tokenizer expects."""
    out: list[str] = []
    glue_next = False
    for i, tok in enumerate(tokens):
        if tok in ("'", "’") and out and i + 1 < len(tokens) and tokens[i + 1].isalpha() and out[-1][-1:].isalpha():
            out[-1] += tok                              # don ' t  ->  don't
            glue_next = True
            continue
        if glue_next or (out and (tok in _NO_SPACE_BEFORE)):
            if out:
                out[-1] += tok
            else:
                out.append(tok)
            glue_next = False
        else:
            out.append(tok)
        if tok in _NO_SPACE_AFTER:
            glue_next = True
    return " ".join(out)


def strip_url_runs(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    """SentiMix splits every t.co link into 'https // t . co / ID' tokens (tagged Eng/Hin, which is nonsense).
    Remove those runs before building the text or training the language tagger."""
    kt, kg, i, n = [], [], 0, len(tokens)
    while i < n:
        if tokens[i].lower() in ("http", "https") and i + 1 < n and tokens[i + 1] in ("//", ":", "://"):
            k = i + 1
            while k < n and k - i <= 8 and tokens[k] != "/":
                k += 1
            i = min(k + 2, n) if (k < n and tokens[k] == "/") else i + 2
            continue
        kt.append(tokens[i])
        kg.append(tags[i])
        i += 1
    return kt, kg


def parse_conll(path: Path) -> list[dict]:
    """SentiMix CoNLL: 'meta<TAB>id[<TAB>label]' then 'token<TAB>LID' lines, blank line between tweets."""
    rows, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip("\r\n")
        if not line.strip():
            if cur:
                rows.append(cur)
            cur = None
            continue
        parts = line.split("\t")
        if parts[0] == "meta":
            if cur:
                rows.append(cur)
            cur = {"uid": parts[1] if len(parts) > 1 else "", "label_name": parts[2].strip() if len(parts) > 2 else None,
                   "tokens": [], "tags": []}
        elif cur is not None and len(parts) >= 2:
            cur["tokens"].append(parts[0])
            cur["tags"].append(parts[-1].strip())
    if cur:
        rows.append(cur)
    for r in rows:
        r["tokens"], r["tags"] = strip_url_runs(r["tokens"], r["tags"])
    return rows


def _sentimix() -> dict[str, pd.DataFrame]:
    def frame(rows: list[dict], split: str) -> pd.DataFrame:
        recs = []
        for r in rows:
            if not r["tokens"]:
                continue
            recs.append({"uid": f"sm_{r['uid']}", "text_raw": detokenise(r["tokens"]), "label_name": r["label_name"],
                         "lid_tokens": "␟".join(r["tokens"]), "lid_tags": " ".join(r["tags"]), "split": split})
        return pd.DataFrame(recs)

    train = frame(parse_conll(RAW / "sentimix/train.txt"), "train")
    dev = frame(parse_conll(RAW / "sentimix/dev.txt"), "dev")
    test = frame(parse_conll(RAW / "sentimix/test_tokens.txt"), "test")
    lab = pd.read_csv(RAW / "sentimix/test_labels.txt")
    lab["uid"] = "sm_" + lab["Uid"].astype(str)
    test = test.drop(columns=["label_name"]).merge(lab[["uid", "Sentiment"]], on="uid", how="left").rename(columns={"Sentiment": "label_name"})
    return {"sentiment": pd.concat([train, dev, test], ignore_index=True)}


def _read_trac(rel: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / rel, header=None, names=["uid", "text_raw", "label_name"], encoding="utf-8",
                     keep_default_na=False, dtype=str)
    df["split"] = split
    return df


def _trac() -> dict[str, pd.DataFrame]:
    parts = [_read_trac("trac/train.csv", "train"), _read_trac("trac/dev.csv", "dev"),
             _read_trac("trac/test_fb.csv", "test"), _read_trac("trac/test_tw.csv", "test_tw")]
    return {"aggression": pd.concat(parts, ignore_index=True)}


def _hot() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RAW / "hot/HOT_dataset.csv", encoding="utf-8", encoding_errors="replace", keep_default_na=False, dtype=str)
    df = df.rename(columns={"text": "text_raw"})
    df["label_name"] = df["score"].astype(float).round().astype(int).map({0: "not_offensive", 1: "abusive", 2: "hate_inducing"})
    df["uid"] = [f"hot_{i}" for i in range(len(df))]
    return {"offensive": df[["uid", "text_raw", "label_name"]]}


def _bohra() -> dict[str, pd.DataFrame]:
    recs = []
    with open(RAW / "bohra/hate_speech.tsv", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if len(row) >= 2 and row[-1].strip() in ("yes", "no"):    # 4 rows carry a stray label ('n', 'on')
                recs.append({"uid": f"bh_{i}", "text_raw": "\t".join(row[:-1]), "label_name": "hate" if row[-1].strip() == "yes" else "no_hate"})
    return {"hate": pd.DataFrame(recs)}


def _youtube() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RAW / "youtube/comments.csv", keep_default_na=False, dtype=str)
    df = df.rename(columns={"comment": "text_raw"})
    df["label_name"] = df["sentiment"].str.strip().str.lower()
    df["uid"] = [f"yt_{i}" for i in range(len(df))]
    df["split"] = "test"
    return {"youtube": df[["uid", "text_raw", "label_name", "split"]]}


# --------------------------------------------------------------------------- splitting
def random_split(df: pd.DataFrame, label_col: str = "label_name") -> pd.DataFrame:
    """Stratified 70/10/20 after removing duplicates (a duplicate must not sit in two splits)."""
    df = df.copy()
    df["_key"] = df["text_raw"].map(dedupe_key)
    conflicts = df.groupby("_key")[label_col].nunique()
    df = df[~df["_key"].isin(conflicts[conflicts > 1].index)]       # same text, two labels -> unusable
    df = df.drop_duplicates("_key").drop(columns="_key")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby(label_col).groups.items():
        idx = list(idx)
        n = len(idx)
        n_dev, n_test = int(round(n * 0.10)), int(round(n * 0.20))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_dev]] = "dev"
    df["split"] = split
    return df


def finish(task: str, df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    """Normalise text, drop empties, remove train rows that leak into dev/test, add features."""
    labels = TASKS[task] if task in TASKS else ["negative", "neutral", "positive"]
    df = df.copy()
    df = df[df["label_name"].isin(labels)]
    df["text"] = df["text_raw"].map(lambda s: normalise(s, "light"))
    n0 = len(df)
    df = df[df["text"].str.len() > 0]
    df["key"] = df["text"].map(dedupe_key)
    df = df[df["key"].str.len() > 0]
    n_empty = n0 - len(df)

    held = set(df.loc[df["split"] != "train", "key"])
    leak = df["split"].eq("train") & df["key"].isin(held)
    n_leak = int(leak.sum())
    df = df[~leak]
    tr = df[df["split"] == "train"]
    conflict = tr.groupby("key")["label_name"].nunique()
    bad_keys = set(conflict[conflict > 1].index)
    dup_in_train = tr.duplicated("key", keep="first") | tr["key"].isin(bad_keys)
    n_dup = int(dup_in_train.sum())
    df = df.drop(index=tr.index[dup_in_train])

    df["label"] = df["label_name"].map({n: i for i, n in enumerate(labels)}).astype(int)
    df["script"] = df["text_raw"].map(script_of)
    df["n_words"] = df["text"].str.split().str.len()
    df["n_emoji"] = df["text_raw"].map(n_emoji)
    counts = df["split"].value_counts().to_dict()
    audit.append({"dataset": task, "rows_after_cleaning": len(df), "empty_removed": n_empty,
                  "train_rows_leaking_into_dev_or_test_removed": n_leak, "train_duplicates_removed": n_dup,
                  **{f"n_{k}": v for k, v in sorted(counts.items())}})
    return df.drop(columns=["key"]).reset_index(drop=True)


def build_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Parse every source, split, clean, and write data/processed/<name>.csv."""
    raw = {}
    for builder in (_sentimix, _trac, _hot, _bohra, _youtube):
        raw.update(builder())
    raw["offensive"] = random_split(raw["offensive"])
    raw["hate"] = random_split(raw["hate"])
    audit, out = [], {}
    for name, df in raw.items():
        out[name] = finish(name, df, audit)
        out[name].to_csv(PROC / f"{name}.csv", index=False, encoding="utf-8")
        if verbose:
            print(f"  {name:11s} {len(out[name]):6d} rows  {out[name]['split'].value_counts().to_dict()}", flush=True)
    pd.DataFrame(audit).to_csv(PROC / "data_audit.csv", index=False)
    return out


def load(name: str) -> pd.DataFrame:
    path = PROC / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the data stage first (run.bat pipeline)")
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8")


def profanity_list() -> list[tuple[str, str, int]]:
    """(romanised word, english gloss, severity 1-5) rows from Mathur et al. 2018."""
    path = RESOURCES / "Hinglish_Profanity_List.csv"
    if not path.exists():
        path = RAW / "hot/Hinglish_Profanity_List.csv"
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                try:
                    sev = int(row[2]) if len(row) > 2 else 3
                except ValueError:
                    sev = 3
                rows.append((row[0].strip().lower(), row[1].strip().lower(), sev))
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("downloading ...")
    download_all()
    print("building ...")
    build_all()
