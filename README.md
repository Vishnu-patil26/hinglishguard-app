# HinglishGuard

Sentiment and toxicity detector for Hindi-English (Hinglish) text, in Roman or Devanagari script.
One multi-task transformer (HingMBERT) reads a message and returns sentiment, aggression, offensiveness and
hate-speech scores, then a moderation policy turns them into *allow / review / block*.

This repository is the **deployable app** (Streamlit). The transformer weights (356 MB) are hosted at
[huggingface.co/Vishnu-patil26/hinglishguard](https://huggingface.co/Vishnu-patil26/hinglishguard) and are
downloaded and SHA-256 checked on first start; if that download fails the app runs a lightweight TF-IDF model instead.

- Entry point: `dashboard/app.py` (six pages: Analyse, Moderate a feed, How good, Where it breaks, Data & method, About).
- Test macro-F1 (2 seeds): sentiment 0.726, aggression 0.661, offensive 0.802, hate 0.674.
- All numbers come from real, published, human-annotated datasets (SentiMix, TRAC-1, HOT, Bohra et al., and a YouTube
  comments set used only as an out-of-domain test). No dataset text is redistributed here except short error examples
  from datasets whose licences allow it. See `DATA.md` and `MODEL_CARD.md`.

Run locally:

```
pip install -r requirements.txt
streamlit run dashboard/app.py
```

**Non-commercial use only** (TRAC-1 is CC BY-NC-SA 4.0). A student research project, not a production moderation
system; do not use it to make decisions about people.
