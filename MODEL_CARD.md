# Model card — HinglishGuard

## Model details

- **Architecture:** `l3cube-pune/hing-mbert` (a BERT-style encoder pretrained on Roman-script Hinglish tweets) → mean pooling → dropout → **four linear heads**: sentiment (3 classes), aggression (3), offensiveness (3), hate speech (2).
- **Training:** multi-task fine-tuning on SentiMix, TRAC-1 Hindi, HOT and the Bohra et al. corpus; word embeddings frozen; AdamW, 4 epochs over the combined data, class-weighted loss, best-dev-macro-F1 checkpoint; seed 43 of 2. Probabilities are temperature-calibrated per head on dev data.
- **Input:** one comment, up to 96 tokens (longer text is truncated). Roman Hinglish works best; Devanagari and mixed script are supported but thinner in the training data.
- **Output:** class probabilities for each head, plus a combined toxicity score and an allow / review / block action from adjustable thresholds.
- **A TF-IDF + logistic-regression "lite" model** with the same four heads ships alongside for instant, PyTorch-free use.

## Intended use

Triage of Hindi-English comments for a **human moderator**, aggregate sentiment measurement, teaching and research on code-mixed NLP.

## Out of scope

Automatic punishment or account action without human review; judging an individual from a single comment; languages or scripts beyond Hindi-English; audio, images or conversation context; any commercial use (the TRAC-1 data is CC BY-NC-SA 4.0).

## Performance (test macro-F1, mean of 2 seeds)

| Task | Test set | Macro-F1 | 95% CI |
|---|---|---:|---:|
| Sentiment | SentiMix tweets | 0.726 | 0.71–0.74 |
| Sentiment | YouTube comments (new platform) | 0.478 | 0.46–0.49 |
| Aggression | TRAC-1 Facebook comments | 0.661 | 0.63–0.69 |
| Aggression | TRAC-1 Twitter (new platform) | 0.406 | 0.38–0.44 |
| Offensiveness | HOT tweets | 0.802 | 0.76–0.84 |
| Hate speech | Bohra et al. tweets | 0.674 | 0.64–0.71 |

**Moderation:** combined score = `mean` of the three toxicity signals; review at 0.40, block at 0.70 (chosen on dev). On the pooled test sets blocking gives precision 81% / recall 33%; reviewing gives precision 67% / recall 78%.

**Calibration:** mean expected calibration error 12.9% before temperature scaling, 4.3% after.

## Known biases and failure modes

- **Identity terms.** On eight harmless hand-written templates the mean toxicity score ranges from 0.04 (Yadav) to 0.31 (Muslim); 12% of these harmless sentences would be sent to review at the default threshold.
- **Sarcasm, banter, reclaimed words** are misread; covert aggression (TRAC's CAG class) is the hardest label.
- **Dataset notions of toxicity differ** (aggression vs offensiveness vs hate); a head transfers poorly to another dataset's definition.
- **New platforms and spelling variation** reduce accuracy; see the *Where it breaks* page.

## Training and evaluation data

See [DATA.md](DATA.md): sources, licences, cleaning and leakage control. All data are real, human-annotated, published corpora; nothing is synthetic. The identity-term probe sentences are hand-written *probes*, not training or evaluation data.

## Use it

```python
from hinglishguard import inference
det = inference.load()                       # transformer if present, else the lite model
r = det.analyse(["Yaar ye video bohot accha tha 🙏"])[0]
print(r.label('sentiment'), r.toxicity, r.action)   # e.g. positive 0.03 allow
print(det.explain(r.text, 'sentiment', 'positive'))  # word-by-word occlusion scores
```

CPU latency ≈ 30 ms per comment (one at a time), 15.1 ms in batches.
