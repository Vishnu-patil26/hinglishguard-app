"""HinglishGuard dashboard.  Run with:  run.bat   (Windows)   or   bash run.sh   (macOS / Linux)."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402
from hinglishguard import inference as I  # noqa: E402
from hinglishguard.data_sources import TASKS  # noqa: E402
from hinglishguard.paths import BUNDLE, TABLES  # noqa: E402

st.set_page_config(page_title="HinglishGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

EXAMPLES = {
    "😂 Happy review": "Yaar yeh movie toh ekdum paisa vasool thi, maza aa gaya 😂🔥",
    "😠 Angry customer": "Kya bakwaas service hai, 2 ghante se wait kar raha hoon aur koi reply nahi aaya",
    "🚫 Hostile": "Tum jaise logon ki wajah se hi yeh desh barbaad hua hai, nikal jao yahan se",
    "💬 Friendly banter": "Bhai tu pagal hai kya, itna bhi kya gussa karna, chill kar 😄",
    "🇮🇳 Devanagari": "आज का मौसम बहुत अच्छा है, चलो घूमने चलते हैं",
    "🙏 Thank-you": "Thanks bro, video bohot helpful tha, subscribe kar diya 🙏",
    "🏛️ Political": "Sab politicians chor hain, in sab ko jail me daal dena chahiye",
}
SAMPLE_FEED = Path(__file__).resolve().parent / "sample_feed.csv"


# --------------------------------------------------------------------------- shared state
@st.cache_resource(show_spinner="Loading the model (first time only) ...")
def get_detector(prefer: str):
    if prefer == "transformer" and not (BUNDLE / "weights.pt").exists():
        from hinglishguard.fetch_model import main as fetch_weights   # hosted copy; checksum-verified
        fetch_weights()
    return I.load(BUNDLE, prefer=prefer)


@st.cache_data
def read_table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_json(name: str) -> dict | None:
    p = TABLES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def bundle_config() -> dict:
    p = BUNDLE / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def sidebar() -> tuple[I.Policy, str]:
    cfg = bundle_config()
    has_tf = (BUNDLE / "config.json").exists() and (BUNDLE / cfg.get("weights", "weights.pt")).exists()
    with st.sidebar:
        st.markdown("### 🛡️ HinglishGuard")
        st.caption("Sentiment and toxicity for Hindi-English text.")
        options = ["Best accuracy (transformer)", "Instant (lite model)"] if has_tf else ["Instant (lite model)"]
        choice = st.radio("Model", options, help="The transformer is more accurate; the lite TF-IDF model answers instantly and needs no PyTorch.")
        prefer = "transformer" if choice.startswith("Best") else "lite"
        st.markdown("**Moderation policy**")
        pol = cfg.get("policy_lite" if prefer == "lite" else "policy", {"block": 0.85, "review": 0.5})
        review = st.slider("Send to review at toxicity ≥", 0.05, 0.95, float(pol["review"]), 0.05, key=f"review_{prefer}")
        block = st.slider("Block at toxicity ≥", 0.05, 0.99, max(float(pol["block"]), review), 0.05, key=f"block_{prefer}")
        if block < review:
            block = review
        rule = cfg.get("rule_lite" if prefer == "lite" else "rule", "max")
        how = {"max": "the highest of three signals", "median": "the middle of three signals", "mean": "the average of three signals"}[rule]
        st.caption(f"Toxicity = {how}: overt aggression, offensiveness and hate. See *Moderate a feed → how thresholds were chosen*.")
    return I.Policy(block=block, review=review), prefer


def detector(prefer: str):
    if prefer == "transformer" and not (BUNDLE / "config.json").exists():
        prefer = "lite"
    try:
        return get_detector("transformer" if prefer == "transformer" else "lite")
    except Exception as exc:                                    # model files missing or torch not installed
        st.error(f"Could not load the model: {exc}")
        st.stop()


# --------------------------------------------------------------------------- pages
def page_analyse():
    policy, prefer = sidebar()
    det = detector(prefer)
    st.title("Hinglish sentiment & toxicity detector")
    st.caption("Type or paste any Hindi-English comment — Roman script, Devanagari, or a mix — and see what the model reads in it.")
    if det.backend == "lite":
        st.info("Running the lite model (TF-IDF). Choose *Best accuracy (transformer)* in the sidebar for the full model.")

    if "text" not in st.session_state:
        st.session_state["text"] = next(iter(EXAMPLES.values()))

    def _use_example():
        pick = st.session_state.get("example_pick")
        if pick:
            st.session_state["text"] = EXAMPLES[pick]
            st.session_state["example_pick"] = None          # so the same example can be chosen again

    st.pills("Try an example", list(EXAMPLES), key="example_pick", on_change=_use_example, label_visibility="collapsed")
    text = st.text_area("Comment", key="text", height=110, label_visibility="collapsed",
                        placeholder="e.g.  Yaar ye video bohot accha tha, thanks bro 🙏")
    if not text.strip():
        st.stop()

    r = det.analyse([text], policy)[0]
    sent = r.label("sentiment")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown('<div class="hg-k">Sentiment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hg-big">{sent.capitalize()}</div><div class="hg-sub">{r.probs["sentiment"][sent] * 100:.0f}% sure</div>', unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown('<div class="hg-k">Toxicity</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hg-big">{r.toxicity * 100:.0f}%</div>{ui.chip(r.action)}', unsafe_allow_html=True)
    with c3.container(border=True):
        st.markdown('<div class="hg-k">Language mix</div>', unsafe_allow_html=True)
        label = "single language" if r.cmi < 1 else "lightly mixed" if r.cmi < 12 else "mixed" if r.cmi < 30 else "heavily mixed"
        st.markdown(f'<div class="hg-big">{label.capitalize()}</div><div class="hg-sub">{r.hindi_share * 100:.0f}% Hindi words · code-mixing index {r.cmi:.0f}</div>', unsafe_allow_html=True)

    st.markdown("#### The four readings")
    g = st.columns(2)
    for i, head in enumerate(I.HEADS):
        with g[i % 2].container(border=True):
            st.markdown(f"**{ui.HEAD_TITLE[head]}**  <span class='hg-sub'>— {ui.HEAD_HELP[head]}</span>", unsafe_allow_html=True)
            st.markdown(ui.bars(r.probs[head], TASKS[head]), unsafe_allow_html=True)

    st.markdown("#### How the model sees the words")
    st.markdown(ui.legend(), unsafe_allow_html=True)
    st.caption("Language tags come from a small word tagger (about 87% word accuracy against noisy gold tags) — treat them as a guide.")
    st.markdown(ui.language_html(r.tags, r.lexicon), unsafe_allow_html=True)
    if r.lexicon:
        st.caption("Profanity-list matches: " + ", ".join(f"**{h['word']}** (≈ {h['meaning']}{', fuzzy match' if h['fuzzy'] else ''})" for h in r.lexicon))

    with st.expander("Why did it say that? (word-by-word explanation)"):
        c = st.columns(2)
        head = c[0].selectbox("Which reading", I.HEADS, format_func=lambda h: ui.HEAD_TITLE[h])
        label = c[1].selectbox("Which outcome", TASKS[head], index=len(TASKS[head]) - 1 if head != "sentiment" else 0, format_func=lambda k: ui.NICE.get(k, k))
        pairs = det.explain(text, head, label)
        st.markdown(f"Each word is removed in turn and we measure how much P(**{ui.NICE.get(label, label)}**) changes. "
                    "<span style='background:rgba(42,120,214,.6);padding:0 .3rem;border-radius:.3rem'>blue</span> = pushes toward it, "
                    "<span style='background:rgba(235,104,52,.6);padding:0 .3rem;border-radius:.3rem'>orange</span> = pushes away. Hover a word for the exact change.",
                    unsafe_allow_html=True)
        st.markdown(ui.occlusion_html(pairs), unsafe_allow_html=True)
        st.caption("Occlusion shows what the model reacts to, not what is 'true' — a model can be right for a wrong reason.")
    st.caption("An automated reading, not a verdict. Context, sarcasm and in-group banter regularly fool every Hinglish classifier — see *Where it breaks*.")


@st.cache_data(show_spinner=False)
def _sample_feed() -> pd.DataFrame | None:
    return pd.read_csv(SAMPLE_FEED) if SAMPLE_FEED.exists() else None


def _analyse_table(_det, texts: list[str], policy: I.Policy) -> pd.DataFrame:
    rows = []
    bar = st.progress(0.0, text="Reading comments ...")
    chunk = 64
    for s in range(0, len(texts), chunk):
        for r in _det.analyse(texts[s:s + chunk], policy):
            rows.append({"comment": r.text, "sentiment": r.label("sentiment"), "toxicity": round(r.toxicity, 3), "action": r.action,
                         "aggression": r.label("aggression"), "offensive": r.label("offensive"), "hate": r.label("hate"),
                         "hindi_share": round(r.hindi_share, 2), "code_mixing": round(r.cmi, 1)})
        bar.progress(min((s + chunk) / len(texts), 1.0), text=f"Reading comments ... {min(s + chunk, len(texts))}/{len(texts)}")
    bar.empty()
    return pd.DataFrame(rows)


def page_feed():
    policy, prefer = sidebar()
    det = detector(prefer)
    st.title("Moderate a feed")
    st.caption("Run the detector over many comments at once — the use-case Hinglish models are really built for.")
    src = st.radio("Comments from", ["Sample feed (real YouTube comments)", "Paste comments", "Upload a CSV"], horizontal=True)
    texts, human = [], None
    if src.startswith("Sample"):
        feed = _sample_feed()
        if feed is None:
            st.warning("sample_feed.csv is missing.")
            st.stop()
        n = st.slider("How many comments", 50, len(feed), min(300, len(feed)), 50)
        sub = feed.head(n)
        texts, human = sub["comment"].tolist(), sub["human_sentiment"].str.lower().tolist()
        st.caption("600 YouTube comments on Indian entertainment videos, hand-labelled for sentiment by the dataset's author "
                   "([shae2977/hinglish-youtube-sentiments-dataset](https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset), CC-BY-4.0). "
                   "Real comments — some contain strong language.")
    elif src.startswith("Paste"):
        raw = st.text_area("One comment per line", height=180, placeholder="Bhai kya mast video hai 🔥\nYeh sab bakwaas hai yaar")
        texts = [t.strip() for t in raw.splitlines() if t.strip()]
    else:
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            col = st.selectbox("Column with the comments", df.columns)
            texts = df[col].astype(str).tolist()[:5000]
            st.caption(f"{len(texts):,} comments (first 5,000 are used).")
    if not texts:
        st.stop()

    key = (src, len(texts), hash(tuple(texts[:50])), prefer)
    if st.session_state.get("feed_key") != key:
        st.session_state["feed_key"] = key
        st.session_state["feed_probs"] = _analyse_table(det, texts, I.Policy(0.85, 0.5))
    table = st.session_state["feed_probs"].copy()
    table["action"] = table["toxicity"].map(policy.action)

    counts = table["action"].value_counts()
    m = st.columns(4)
    m[0].metric("Comments", f"{len(table):,}")
    for col, a in zip(m[1:], ["allow", "review", "block"]):
        col.metric(f"{ui.ACTION[a][0]} {ui.ACTION[a][1]}  ·  {counts.get(a, 0) / len(table) * 100:.0f}% of comments", f"{counts.get(a, 0):,}")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Sentiment mix**")
        s = table["sentiment"].value_counts().reindex(TASKS["sentiment"]).fillna(0).reset_index()
        s.columns = ["sentiment", "comments"]
        st.altair_chart(alt.Chart(s).mark_bar(cornerRadiusEnd=4, size=26).encode(
            y=alt.Y("sentiment:N", sort=TASKS["sentiment"], title=None), x=alt.X("comments:Q", title="comments"),
            color=alt.Color("sentiment:N", scale=alt.Scale(domain=TASKS["sentiment"], range=[ui.CLASS_COLOURS[k] for k in TASKS["sentiment"]]), legend=None),
            tooltip=["sentiment", "comments"]).properties(height=150), use_container_width=True)
    with c2.container(border=True):
        st.markdown("**Toxicity scores**")
        h = alt.Chart(table).mark_bar(color=ui.CAT[0], cornerRadiusEnd=3).encode(
            x=alt.X("toxicity:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="toxicity score"), y=alt.Y("count()", title="comments"), tooltip=["count()"])
        rules = alt.Chart(pd.DataFrame({"t": [policy.review, policy.block], "what": ["review", "block"]})).mark_rule(strokeDash=[4, 3], size=2, color="#d03b3b").encode(x="t:Q", tooltip=["what", "t"])
        st.altair_chart((h + rules).properties(height=150), use_container_width=True)

    if human is not None:
        agree = (table["sentiment"].str.lower().values == np.array(human)).mean()
        st.markdown(f'<div class="hg-find"><b>Against the human labels:</b> the model agrees with the annotator on <b>{agree * 100:.0f}%</b> of these comments '
                    f'(three classes; picking at random would be ~33%). This is out-of-domain for the model — it never saw YouTube comments in training.</div>', unsafe_allow_html=True)
        ct = pd.crosstab(pd.Series(human, name="human label"), table["sentiment"].rename("model says")).reindex(index=TASKS["sentiment"], columns=TASKS["sentiment"], fill_value=0)
        with st.expander("Confusion table (human vs model)"):
            st.dataframe(ct, use_container_width=True)

    st.markdown("#### Comments")
    f1, f2 = st.columns(2)
    act = f1.multiselect("Show actions", ["allow", "review", "block"], default=["review", "block"] if counts.get("review", 0) + counts.get("block", 0) else ["allow"])
    sent = f2.multiselect("Show sentiment", TASKS["sentiment"], default=TASKS["sentiment"])
    view = table[table["action"].isin(act) & table["sentiment"].isin(sent)].sort_values("toxicity", ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True, height=360, column_config={
        "toxicity": st.column_config.ProgressColumn("toxicity", min_value=0.0, max_value=1.0, format="%.2f"),
        "comment": st.column_config.TextColumn("comment", width="large")})
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    st.download_button("⬇ Download all results (CSV)", buf.getvalue().encode("utf-8-sig"), "hinglishguard_results.csv", "text/csv")

    with st.expander("How the default thresholds were chosen"):
        sfx = "_lite" if prefer == "lite" else ""
        curve = read_table(f"moderation_curve{sfx}.csv")
        summ = read_json(f"moderation_summary{sfx}.json")
        if curve is None or summ is None:
            st.write("Run the analysis stage to see this.")
        else:
            st.markdown(f"The review and block thresholds were picked on the **dev** sets (never the test sets): *block* = the lowest score at which dev precision reaches 80% "
                        f"(**{summ['block']:.2f}**), *review* = the score that maximises dev F1 (**{summ['review']:.2f}**). "
                        f"On the pooled test sets that gives: block → precision {summ['test_block']['precision'] * 100:.0f}%, recall {summ['test_block']['recall'] * 100:.0f}%; "
                        f"review → precision {summ['test_review']['precision'] * 100:.0f}%, recall {summ['test_review']['recall'] * 100:.0f}%. "
                        f"Ranking quality (AUROC) on the pooled test sets: {summ['auroc_test']:.2f}.")
            t = curve[curve["set"] == "test"].melt(id_vars="threshold", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
            ch = alt.Chart(t).mark_line(point=True, size=2.5).encode(
                x=alt.X("threshold:Q", title="toxicity threshold"), y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall", "f1"], range=[ui.CAT[0], ui.CAT[1], ui.CAT[2]]), title=None), tooltip=["metric", "threshold", alt.Tooltip("value:Q", format=".2f")])
            vr = alt.Chart(pd.DataFrame({"t": [policy.review, policy.block]})).mark_rule(strokeDash=[4, 3], color="#d03b3b").encode(x="t:Q")
            st.altair_chart((ch + vr).properties(height=240), use_container_width=True)


# --------------------------------------------------------------------------- navigation
import pages_report as R  # noqa: E402

nav = st.navigation([
    st.Page(page_analyse, title="Analyse a comment", icon="🔎", default=True, url_path="analyse"),
    st.Page(page_feed, title="Moderate a feed", icon="🧹", url_path="feed"),
    st.Page(R.page_performance, title="How good is it?", icon="📊", url_path="performance"),
    st.Page(R.page_breaks, title="Where it breaks", icon="🔬", url_path="breaks"),
    st.Page(R.page_data, title="Data & method", icon="🗂️", url_path="data"),
    st.Page(R.page_about, title="About", icon="ℹ️", url_path="about"),
], position="top")
nav.run()
