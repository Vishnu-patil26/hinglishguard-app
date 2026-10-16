"""Load the exported model bundle and analyse Hinglish text (used by the dashboard and the export check).

Two interchangeable back-ends with the same interface:
  TransformerDetector  the multi-task encoder (best accuracy)
  LiteDetector         TF-IDF + logistic regression (tiny, instant, needs no GPU or torch)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import joblib
import numpy as np

from .data_sources import TASKS
from .lexicon import ProfanityLexicon
from .lid import WordLID, cmi, hindi_share, int32_csr, tokenize
from .paths import BUNDLE
from .textnorm import normalise

HEADS = list(TASKS)                       # sentiment, aggression, offensive, hate


@dataclass
class Policy:
    """How the combined toxicity score becomes a moderation action. Thresholds are user-adjustable."""
    block: float = 0.85
    review: float = 0.50

    def action(self, score: float) -> str:
        return "block" if score >= self.block else "review" if score >= self.review else "allow"


RULES = ("max", "median", "mean")


def toxicity_components(probs: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 3) 'how bad' probabilities of the three toxicity heads:
    aggression -> P(overtly aggressive), offensive -> P(abusive) + P(hate-inducing), hate -> P(hate)."""
    p_agg = probs["aggression"][:, TASKS["aggression"].index("OAG")]
    p_off = probs["offensive"][:, 1:].sum(1)
    p_hate = probs["hate"][:, TASKS["hate"].index("hate")]
    return np.stack([p_agg, p_off, p_hate], axis=1)


def combine(components: np.ndarray, rule: str = "max") -> np.ndarray:
    if rule == "max":
        return components.max(axis=1)
    if rule == "median":
        return np.median(components, axis=1)
    if rule == "mean":
        return components.mean(axis=1)
    raise ValueError(f"unknown rule {rule!r}")


def toxicity_score(probs: dict[str, np.ndarray], rule: str = "max") -> np.ndarray:
    """Combined risk in [0, 1]. 'max' is the conservative default (any one signal can raise the alarm); stage 6 compares the
    rules on the dev sets and the exported model records the winner."""
    return combine(toxicity_components(probs), rule)


def _softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class Result:
    text: str
    clean: str
    probs: dict[str, dict[str, float]]
    toxicity: float
    action: str
    cmi: float
    hindi_share: float
    tags: list[tuple[str, str]]
    lexicon: list[dict] = field(default_factory=list)

    def label(self, head: str) -> str:
        p = self.probs[head]
        return max(p, key=p.get)


class _Base:
    backend = "base"
    labels = TASKS

    def __init__(self, bundle=BUNDLE) -> None:
        self.bundle = bundle
        self.lid = WordLID.load(bundle / "lid.joblib")
        self.lex = ProfanityLexicon(fuzzy=True)
        self.policy = Policy()
        self.rule = "max"

    # subclasses implement: raw class probabilities for every head
    def predict_probs(self, clean_texts: list[str]) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def analyse(self, texts: list[str], policy: Policy | None = None) -> list[Result]:
        policy = policy or self.policy
        clean = [normalise(t, "light") or " " for t in texts]
        probs = self.predict_probs(clean)
        tox = toxicity_score(probs, self.rule)
        token_lists = [tokenize(c) for c in clean]
        tag_lists = self.lid.tag_token_lists(token_lists)
        out = []
        for i, (raw, c) in enumerate(zip(texts, clean)):
            tags = tag_lists[i]
            hits = [{"word": h.word, "matches": h.entry, "meaning": h.gloss, "severity": h.severity, "fuzzy": h.fuzzy}
                    for h in self.lex.find(c)]
            out.append(Result(
                text=raw, clean=c,
                probs={h: {lab: float(probs[h][i, j]) for j, lab in enumerate(self.labels[h])} for h in HEADS},
                toxicity=float(tox[i]), action=policy.action(float(tox[i])),
                cmi=cmi(tags), hindi_share=hindi_share(tags), tags=list(zip(token_lists[i], tags)), lexicon=hits))
        return out

    def explain(self, text: str, head: str, label: str | None = None, max_words: int = 60) -> list[tuple[str, float]]:
        """Occlusion: how much does P(label) drop when each word is removed? Positive = the word pushes toward the label."""
        clean = normalise(text, "light")
        words = clean.split()[:max_words]
        if not words:
            return []
        variants = [clean] + [" ".join(words[:i] + words[i + 1:]) or " " for i in range(len(words))]
        p = self.predict_probs(variants)[head]
        j = self.labels[head].index(label) if label else int(p[0].argmax())
        return [(w, float(p[0, j] - p[i + 1, j])) for i, w in enumerate(words)]


class LiteDetector(_Base):
    backend = "lite"

    def __init__(self, bundle=BUNDLE) -> None:
        super().__init__(bundle)
        self.lite = joblib.load(bundle / "lite.joblib")
        cfg_p = bundle / "config.json"
        if cfg_p.exists():
            self.rule = json.loads(cfg_p.read_text(encoding="utf-8")).get("rule_lite", "max")

    def predict_probs(self, clean_texts):
        feats = [normalise(t, "full") for t in clean_texts]
        out = {}
        for h in HEADS:
            X = int32_csr(self.lite[h]["vectorizer"].transform(feats))
            out[h] = self.lite[h]["model"].predict_proba(X)
        return out


class TransformerDetector(_Base):
    backend = "transformer"

    def __init__(self, bundle=BUNDLE, threads: int | None = None) -> None:
        super().__init__(bundle)
        import torch
        from transformers import AutoTokenizer

        from .modeling import MultiHeadModel

        self.torch = torch
        torch.set_num_threads(threads or min(8, os.cpu_count() or 4))
        self.cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(str(bundle / "tokenizer"))
        self.max_len = int(self.cfg.get("max_len", 96))
        self.rule = self.cfg.get("rule", "max")
        self.temperature = self.cfg.get("temperature", {})
        model = MultiHeadModel(str(bundle / "encoder_config"), {h: len(TASKS[h]) for h in HEADS}, from_config_only=True,
                               freeze_word_embeddings=False)
        state = torch.load(bundle / self.cfg["weights"], map_location="cpu", weights_only=True)
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}, strict=True)
        model.eval()
        if self.cfg.get("quantize", False):
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model

    def predict_probs(self, clean_texts, batch_size: int = 32):
        from .modeling import collate

        torch = self.torch
        ids = self.tok(list(clean_texts), truncation=True, max_length=self.max_len)["input_ids"]
        order = np.argsort([len(x) for x in ids])
        out = {h: np.zeros((len(ids), len(TASKS[h])), dtype=np.float32) for h in HEADS}
        with torch.inference_mode():
            for s in range(0, len(ids), batch_size):
                sel = order[s:s + batch_size]
                inp, att = collate([ids[i] for i in sel], self.tok.pad_token_id)
                pooled = self.model.pooled(inp, att)
                for h in HEADS:
                    logits = self.model.heads[h](pooled).numpy()
                    out[h][sel] = _softmax(logits, self.temperature.get(h, 1.0))
        return out


def load(bundle=BUNDLE, prefer: str = "auto"):
    """Best available detector. 'auto' uses the transformer if its files and torch are present, else the lite models."""
    if prefer in ("auto", "transformer") and (bundle / "config.json").exists():
        try:
            return TransformerDetector(bundle)
        except Exception as exc:                       # missing torch / weights -> fall back, but say so
            if prefer == "transformer":
                raise
            print(f"[hinglishguard] transformer model unavailable ({exc}); using the lite model")
    return LiteDetector(bundle)


def benchmark(det, texts: list[str], repeat: int = 3) -> dict:
    """Median milliseconds per text at batch size 1 and as one batch."""
    det.predict_probs(texts[:2])                      # warm-up
    one = []
    for t in texts[:20]:
        s = time.perf_counter()
        det.predict_probs([t])
        one.append((time.perf_counter() - s) * 1000)
    bat = []
    for _ in range(repeat):
        s = time.perf_counter()
        det.predict_probs(texts)
        bat.append((time.perf_counter() - s) * 1000 / len(texts))
    return {"ms_per_text_single": float(np.median(one)), "ms_per_text_batched": float(np.median(bat))}
