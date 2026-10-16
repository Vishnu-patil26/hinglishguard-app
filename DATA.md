# Data sources, licences and what this project does with them

Every row used here is **real, human-annotated data from a published research corpus**. Nothing is generated,
simulated or augmented. The raw files are **not included** in this repository or its zip: `pipeline/s01_data.py`
downloads them from the public locations below on your machine, caches them in `data/raw/` and records each file's
SHA-256 in `data/raw/MANIFEST.json`.

| Task | Dataset | What it is | Labels | Splits used | Licence / terms |
|---|---|---|---|---|---|
| Sentiment | **SentiMix**, SemEval-2020 Task 9 (Hinglish) | Hinglish tweets with word-level language tags | negative / neutral / positive | official train, dev, test (labels released after the task) | OpenRAIL on the hosted copy; research use of the task data |
| Aggression | **TRAC-1 Hindi** (Kumar, Reganti, Bhatia, Maheshwari, LREC 2018) | Facebook comments (Roman and Devanagari) + a Twitter "surprise" test | NAG / CAG / OAG | official train, dev, Facebook gold test, Twitter gold test | **CC BY-NC-SA 4.0** (non-commercial, share-alike) |
| Offensiveness | **HOT** (Mathur, Sawhney, Ayyar, Shah, ALW 2018) | Hinglish tweets | not offensive / abusive / hate-inducing | random stratified 70/10/20 after de-duplication | research use; the authors share the tweet text on request |
| Hate speech | **Bohra et al. 2018** | Hindi-English code-mixed tweets | no hate / hate | random stratified 70/10/20 after de-duplication | research use; the authors share the tweet text on request |
| Out-of-domain test | **Hinglish YouTube comments** (`shae2977/hinglish-youtube-sentiments-dataset`) | 3,190 comments, hand-labelled in Label Studio | negative / neutral / positive | test only, never trained on | **CC BY 4.0** |
| Lexicon | **Hinglish profanity list** (Mathur et al. 2018) | 208 romanised Hindi profanities with English gloss and a 1-5 severity | - | used for the lexicon baseline and word highlighting | published by the authors with the HOT paper |

## Where each file comes from

`hinglishguard/data_sources.py` lists the exact URLs:

* SentiMix - `huggingface.co/datasets/RTT1/SentiMix` (the four CoNLL files)
* TRAC-1 Hindi - the `data/raw/trac1-dataset/` folder of `github.com/victor7246/Hinglish_Hate_Detection` (official TRAC files, unmodified)
* HOT - `github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/Data/HOT_dataset.csv` (a public copy of the authors' file; 3,189 tweets, matching the paper's class counts 1,765 / 303 / 1,121)
* Bohra - `data/raw/Hate-speech-dataset/hate_speech.tsv` in the same `victor7246` repository (4,579 tweets, matching the paper's 1,661 hate / 2,914 normal)
* YouTube - `huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset`
* Profanity list - `github.com/pmathur5k10/Hinglish-Offensive-Text-Classification`

**A note on HOT and Bohra.** Their authors state that, because the corpora are tweets, they share the text only for research
and on request. The copies above are public mirrors. This project uses them for research and teaching only, does not
redistribute them, and the trained model is a derivative that stores no tweet text. If you build on this, please cite the
original papers and respect their terms; if you can, request the data from the authors directly.
TRAC-1 is CC BY-NC-SA 4.0, so **this project's use is non-commercial**.

## Cleaning that was applied (and is reported in the app)

* `normalise()` in `hinglishguard/textnorm.py`: repairs literal `\xNN` byte escapes left in HOT, removes URLs and the `RT @user:` prefix,
  collapses runs of `@mentions` to one `@user`, unescapes HTML, normalises Unicode (NFC), removes zero-width characters.
  Spelling, case and emoji are left alone for the transformers.
* SentiMix stores every link as separate tokens (`https // t . co / ID`); those runs are removed and the tweet is re-assembled from its tokens.
* Rows with no real text after cleaning (for example a tweet that was only mentions) are dropped.
* **Leakage control:** any training row whose normalised text also appears in the dev or test set is removed from training.
  Duplicate training texts are collapsed; a text that carries two different labels is dropped entirely.
  For the two random-split datasets duplicates are removed *before* splitting, so a retweet cannot sit on both sides.
* The exact counts are in `outputs/tables/data_audit.csv` and shown on the *Data & method* page.
  Pairwise overlap between datasets is in `outputs/tables/dataset_overlap.csv` (at most 3 identical texts between any two datasets).

## What the labels mean (and why they do not agree)

* **Sentiment** - the writer's overall feeling, three-way.
* **Aggression (TRAC-1)** - *overtly* aggressive (OAG: direct abuse, threats), *covertly* aggressive (CAG: sarcasm, insinuation, indirect hostility), or not aggressive (NAG).
* **Offensiveness (HOT)** - *hate-inducing* (targets a group, incites), *abusive* (profanity or insults, no group), or not offensive.
* **Hate (Bohra)** - does the tweet express hate speech, yes/no.

These are four different annotation schemes by four teams. The *Where it breaks* page measures how far a model trained on one
transfers to another: they overlap, but they are not the same concept.

## Citations

* Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., Das, A. (2020). *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.* SemEval.
* Kumar, R., Reganti, A. N., Bhatia, A., Maheshwari, T. (2018). *Aggression-annotated Corpus of Hindi-English Code-mixed Data.* LREC. Also: Kumar et al. (2018) *Benchmarking Aggression Identification in Social Media* (TRAC-1).
* Mathur, P., Sawhney, R., Ayyar, M., Shah, R. (2018). *Did you offend me? Classification of Offensive Tweets in Hinglish Language.* ALW2.
* Bohra, A., Vijay, D., Singh, V., Akhtar, S. S., Shrivastava, M. (2018). *A Dataset of Hindi-English Code-Mixed Social Media Text for Hate Speech Detection.* PEOPLES workshop.
* Sengupta, A., Bhattacharjee, S. K., Akhtar, M. S., Chakraborty, T. (2022). *Does aggression lead to hate? Detecting and reasoning offensive traits in Hinglish code-mixed texts.* Neurocomputing - the multi-trait framing this project borrows.
