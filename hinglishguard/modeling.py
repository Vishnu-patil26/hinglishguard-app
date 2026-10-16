"""One shared encoder with one classification head per task, plus training and inference helpers.

Single-task fine-tuning is the special case of a model with one head. Everything runs on a GPU
when there is one and on the CPU otherwise.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MAX_LEN = 96


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiHeadModel(nn.Module):
    """encoder -> mean-pool -> dropout -> one Linear per task."""

    def __init__(self, encoder: str, heads: dict[str, int], dropout: float = 0.1, from_config_only: bool = False,
                 freeze_word_embeddings: bool = True):
        super().__init__()
        if from_config_only:                       # used when loading a saved bundle (weights come from the state dict)
            from transformers import AutoConfig
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder))
        else:
            self.encoder = AutoModel.from_pretrained(encoder)
        if freeze_word_embeddings:
            # XLM-R-sized vocabularies are ~190M of 280M parameters; training them needs gradients + Adam state
            # (~2.3 GB) for words these tasks barely move. Freezing them fits a 6 GB GPU and costs almost no accuracy.
            self.encoder.get_input_embeddings().weight.requires_grad_(False)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, n) for t, n in heads.items()})

    def pooled(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, task: str):
        return self.heads[task](self.drop(self.pooled(input_ids, attention_mask)))


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def encode(tok, texts: list[str], max_len: int = MAX_LEN) -> list[list[int]]:
    return tok(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]


def collate(ids: list[list[int]], pad_id: int):
    width = max(len(x) for x in ids)
    inp = torch.full((len(ids), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return inp, att


@dataclass
class TaskData:
    name: str
    n_classes: int
    train_ids: list[list[int]]
    train_y: np.ndarray
    dev_ids: list[list[int]]
    dev_y: np.ndarray
    class_weights: torch.Tensor | None = None
    extra_eval: dict = field(default_factory=dict)          # name -> (ids, y)  e.g. out-of-domain test sets


def class_weights(y: np.ndarray, n: int) -> torch.Tensor:
    """Inverse-sqrt-frequency weights, mean-normalised: nudges the loss toward rare classes without swamping it."""
    counts = np.bincount(y, minlength=n).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.no_grad()
def predict_logits(model: MultiHeadModel, ids: list[list[int]], task: str, pad_id: int, bs: int = 128,
                   dev: torch.device | None = None, amp: bool = True) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval()
    order = np.argsort([len(x) for x in ids])                # length-sorted batches -> little padding
    out = np.zeros((len(ids), model.heads[task].out_features), dtype=np.float32)
    for s in range(0, len(ids), bs):
        sel = order[s:s + bs]
        inp, att = collate([ids[i] for i in sel], pad_id)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            lg = model(inp.to(dev), att.to(dev), task)
        out[sel] = lg.float().cpu().numpy()
    return out


def macro_f1(y: np.ndarray, logits: np.ndarray) -> float:
    return float(f1_score(y, logits.argmax(1), average="macro", zero_division=0))


def train(model: MultiHeadModel, tasks: dict[str, TaskData], pad_id: int, *, epochs: float = 4.0, lr: float = 3e-5,
          bs: int = 32, seed: int = 42, warmup: float = 0.1, weight_decay: float = 0.01, eval_every: int | None = None,
          sample_power: float = 0.5, log=print) -> dict:
    """Fine-tune. One task = ordinary fine-tuning. Several tasks = multi-task with temperature sampling
    (task chosen per step with probability proportional to size**sample_power). The checkpoint with the best
    mean dev macro-F1 is restored at the end."""
    seed_everything(seed)
    dev = device()
    model.to(dev)
    names = list(tasks)
    sizes = np.array([len(tasks[t].train_y) for t in names], dtype=float)
    steps_per_epoch = int(math.ceil(sizes.sum() / bs))
    total = int(steps_per_epoch * epochs)
    probs = sizes ** sample_power
    probs /= probs.sum()
    eval_every = eval_every or max(steps_per_epoch // 2, 50)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [
        {"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    # per-task epoch-style shuffled index streams, so every example is seen before any repeats
    rng = np.random.default_rng(seed)
    streams = {t: [] for t in names}

    def next_batch(t: str):
        d = tasks[t]
        if len(streams[t]) < bs:
            streams[t] = list(rng.permutation(len(d.train_y))) + streams[t]
        take, streams[t] = streams[t][:bs], streams[t][bs:]
        inp, att = collate([d.train_ids[i] for i in take], pad_id)
        return inp.to(dev), att.to(dev), torch.tensor(d.train_y[take], dtype=torch.long, device=dev)

    best, best_state, history = -1.0, None, []
    t0, run_loss = time.time(), 0.0
    for step in range(1, total + 1):
        model.train()
        t = names[int(rng.choice(len(names), p=probs))]
        inp, att, y = next_batch(t)
        w = tasks[t].class_weights.to(dev) if tasks[t].class_weights is not None else None
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(inp, att, t)
        loss = F.cross_entropy(logits.float(), y, weight=w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()

        if step % eval_every == 0 or step == total:
            scores = {n: macro_f1(tasks[n].dev_y, predict_logits(model, tasks[n].dev_ids, n, pad_id, dev=dev)) for n in names}
            mean = float(np.mean(list(scores.values())))
            history.append({"step": step, "loss": run_loss / eval_every, **{f"dev_f1_{k}": v for k, v in scores.items()}, "dev_mean_f1": mean})
            log(f"    step {step:5d}/{total}  loss {run_loss / eval_every:.3f}  dev macro-F1 "
                + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()) + f"  ({time.time() - t0:.0f}s)")
            run_loss = 0.0
            if mean > best:
                best = mean
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_dev_mean_f1": best, "history": history, "seconds": time.time() - t0, "steps": total}


def temperature_fit(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature T minimising dev NLL (post-hoc calibration, Guo et al. 2017).
    A log-spaced grid search: slower than L-BFGS in theory, but it cannot diverge on degenerate logits."""
    lg = torch.tensor(np.nan_to_num(logits), dtype=torch.float)
    yy = torch.tensor(y, dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.2), np.log(10.0), 160))
    nll = [F.cross_entropy(lg / float(t), yy).item() for t in grid]
    return float(grid[int(np.nanargmin(nll))])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
