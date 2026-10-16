"""The four read-only pages: performance, failure analysis, data & method, about. All numbers come from outputs/tables."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from hinglishguard.data_sources import TASKS
from hinglishguard.paths import BUNDLE, FIGS, OUT, TABLES

TASK_LABEL = {"sentiment": "Sentiment", "aggression": "Aggression", "offensive": "Offensiveness", "hate": "Hate speech"}
DATASET_NAME = {"sentiment": "SentiMix tweets", "aggression": "TRAC-1 Facebook comments", "offensive": "HOT tweets", "hate": "Bohra et al. tweets"}
FAMILY_COL = {"deployed": ui.CAT[0], "single": ui.BLUE[250], "baseline": ui.NEUTRAL}


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def jsonf(name: str, base: str = "tables") -> dict | None:
    p = (TABLES if base == "tables" else OUT) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def need(*names: str) -> bool:
    missing = [n for n in names if not (TABLES / n).exists()]
    if missing:
        st.info("This page fills in once the pipeline has been run (`run.bat pipeline`). Missing: " + ", ".join(missing))
        return False
    return True


def finding(html_text: str) -> None:
    st.markdown(f'<div class="hg-find">{html_text}</div>', unsafe_allow_html=True)


def fig(name: str, caption: str | None = None) -> None:
    p = FIGS / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def pretty_model(m: str, family: str) -> str:
    names = {"majority": "Majority class", "lexicon_exact": "Profanity list (exact)", "lexicon_fuzzy": "Profanity list (fuzzy)",
             "tfidf_logreg": "TF-IDF + logistic regression", "tfidf_linsvm": "TF-IDF + linear SVM"}
    if family == "multi":
        return "HinglishGuard (multi-task transformer)"
    if family == "single":
        return "Same encoder, one model per task"
    return names.get(m, m)


# --------------------------------------------------------------------------------------------- performance
def page_performance():
    st.title("How good is it?")
    st.caption("Macro-F1 (the average of each class's F1, so rare classes count as much as common ones) on held-out test sets the model never trained on.")
    if not need("headline_results.csv"):
        return
    h = table("headline_results.csv").copy()
    h["family"] = h["family"].replace({"multi": "deployed"})
    h["label"] = [pretty_model(m, "multi" if f == "deployed" else f) for m, f in zip(h["model"], h["family"])]

    task = st.radio("Task", list(TASK_LABEL), format_func=lambda t: f"{TASK_LABEL[t]}  ·  {DATASET_NAME[t]}", horizontal=True)
    d = h[(h["task"] == task) & (h["eval_set"] == "test")].sort_values("macro_f1", ascending=False)
    best = d.iloc[0]
    dep = d[d["family"] == "deployed"].iloc[0]
    tf = d[d["model"] == "tfidf_logreg"].iloc[0]
    maj = d[d["model"] == "majority"].iloc[0]
    finding(f"<b>{TASK_LABEL[task]}:</b> the deployed model scores <b>{dep['macro_f1']:.3f}</b> macro-F1 (95% interval {dep['ci_lo']:.2f}–{dep['ci_hi']:.2f}). "
            f"A TF-IDF classifier gets {tf['macro_f1']:.3f} and always guessing the commonest class gets {maj['macro_f1']:.3f}. "
            + (f"The best model here is <b>{best['label']}</b> at {best['macro_f1']:.3f}." if best["family"] != "deployed" else "That is the best model on this task."))
    d = d.assign(color=d["family"].map(FAMILY_COL))
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=list(d["label"]), title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadiusEnd=4, size=20).encode(
        x=alt.X("macro_f1:Q", title="macro-F1", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("family:N", scale=alt.Scale(domain=list(FAMILY_COL), range=list(FAMILY_COL.values())), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("label", title="model"), alt.Tooltip("macro_f1:Q", format=".3f"), alt.Tooltip("ci_lo:Q", format=".3f"), alt.Tooltip("ci_hi:Q", format=".3f"), alt.Tooltip("accuracy:Q", format=".3f")])
    err = base.mark_rule(size=2, color="#52514e").encode(x="ci_lo:Q", x2="ci_hi:Q")
    txt = base.mark_text(align="left", dx=6, fontSize=12).encode(x=alt.X("ci_hi:Q"), text=alt.Text("macro_f1:Q", format=".3f"))
    st.altair_chart((bars + err + txt).properties(height=alt.Step(36)), use_container_width=True)
    st.caption("Whiskers: 95% bootstrap interval over test examples (transformers: mean over 2 training seeds). Intervals that overlap are not distinguishable.")

    st.markdown("##### Every task at a glance")
    sc = h[h["eval_set"] == "test"].pivot_table(index="label", columns="task", values="macro_f1")[list(TASK_LABEL)]
    sc.columns = [TASK_LABEL[c] for c in sc.columns]
    st.dataframe(sc.sort_values(sc.columns[0], ascending=False, na_position="last").style.format("{:.3f}", na_rep="–")
                 .background_gradient(cmap="Blues", vmin=0.2, vmax=0.9).highlight_null(props="background-color:#f0efec;color:#898781"), use_container_width=True)

    enc = table("results_encoders.csv")
    if enc is not None:
        st.markdown("### Which pretrained model understands Hinglish best?")
        e = enc[(enc["eval_set"] == "test")].copy()
        names = {"hingbert": "HingBERT", "hingroberta": "HingRoBERTa", "hingmbert": "HingMBERT", "muril": "MuRIL", "xlmr": "XLM-R base", "mbert": "mBERT", "distilmbert": "DistilmBERT"}
        e["encoder"] = e["encoder"].map(names).fillna(e["encoder"])
        order = e.groupby("encoder")["macro_f1"].mean().sort_values(ascending=False)
        e["task"] = e["task"].map(TASK_LABEL)
        finding(f"Each encoder was fine-tuned identically on each task (same data, epochs and tricks). Averaged over the four tasks, <b>{order.index[0]}</b> comes first ({order.iloc[0]:.3f}) "
                f"and <b>{order.index[-1]}</b> last ({order.iloc[-1]:.3f}). The Hinglish-specific models (Hing*) were pretrained on Roman-script Hinglish tweets; the others on multilingual text.")
        dots = alt.Chart(e).mark_circle(size=130, color=ui.CAT[0]).encode(
            y=alt.Y("encoder:N", sort=list(order.index), title=None), x=alt.X("macro_f1:Q", title="test macro-F1", scale=alt.Scale(zero=False)),
            tooltip=["encoder", "task", alt.Tooltip("macro_f1:Q", format=".3f")]).properties(width=170, height=190)
        st.altair_chart(dots.facet(column=alt.Column("task:N", sort=[TASK_LABEL[t] for t in TASK_LABEL], title=None)), use_container_width=True)

    fin = table("results_final_summary.csv")
    if fin is not None:
        st.markdown("### One shared model vs four separate ones")
        f = fin[fin["eval_set"] == "test"].pivot(index="task", columns="mode", values="mean")
        sd = fin[fin["eval_set"] == "test"].pivot(index="task", columns="mode", values="std")
        def pm(m, d):
            return f"{m:.3f}" if pd.isna(d) else f"{m:.3f} ± {d:.3f}"

        out = pd.DataFrame({TASK_LABEL[t]: {"One model per task": pm(f.loc[t, 'single'], sd.loc[t, 'single']),
                                           "One multi-task model": pm(f.loc[t, 'multi'], sd.loc[t, 'multi']),
                                           "Difference": f"{f.loc[t, 'multi'] - f.loc[t, 'single']:+.3f}"} for t in TASK_LABEL if t in f.index})
        st.dataframe(out, use_container_width=True)
        diff = {t: f.loc[t, "multi"] - f.loc[t, "single"] for t in f.index}
        noise = {t: float(np.nanmax([sd.loc[t, "multi"], sd.loc[t, "single"], 0.0])) for t in f.index}
        better = [t for t in f.index if diff[t] > noise[t]]
        worse = [t for t in f.index if diff[t] < -noise[t]]
        same = [t for t in f.index if t not in better and t not in worse]
        finding(f"Against four separate models, the shared model is <b>better on {len(better)}</b>, <b>within seed-to-seed noise on {len(same)}</b> and <b>worse on {len(worse)}</b> of {len(f)} tasks "
                f"(mean ± std over 2 seeds; a gap smaller than the std is noise). One model serves all four readings with a single forward pass, instead of four.")

    conf = jsonf("confusion.json")
    if conf:
        st.markdown("### Where the deployed model confuses classes")
        t2 = st.radio("Task ", list(TASK_LABEL), format_func=lambda t: TASK_LABEL[t], horizontal=True, key="conf_task")
        cm = np.array(conf[t2]["test"])
        labels = [ui.NICE.get(k, k) for k in TASKS[t2]]
        df = pd.DataFrame([{"true": labels[i], "predicted": labels[j], "share": cm[i, j] / max(cm[i].sum(), 1), "n": int(cm[i, j])} for i in range(len(labels)) for j in range(len(labels))])
        base = alt.Chart(df).encode(x=alt.X("predicted:N", sort=labels, title="model says"), y=alt.Y("true:N", sort=labels, title="human label"))
        rect = base.mark_rect(cornerRadius=3).encode(color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=["#eef5fd", "#0d366b"]), legend=None), tooltip=["true", "predicted", "n", alt.Tooltip("share:Q", format=".0%")])
        text = base.mark_text(fontSize=13).encode(text=alt.Text("share:Q", format=".0%"), color=alt.condition("datum.share > 0.5", alt.value("white"), alt.value("#0b0b0b")))
        st.altair_chart((rect + text).properties(height=60 * len(labels) + 40), use_container_width=True)
        st.caption("Each row sums to 100%: of all texts humans gave that label, how many did the model send where.")


# --------------------------------------------------------------------------------------------- failure analysis
def page_breaks():
    st.title("Where it breaks")
    st.caption("A single accuracy number hides how a model fails. These are the stress tests.")
    st.markdown('<div class="hg-warn">Some examples on this page come from real social-media datasets and may contain offensive language.</div>', unsafe_allow_html=True)

    # 1. transfer
    tr = table("transfer_auroc.csv")
    if tr is not None:
        st.markdown("### 1 · Does 'toxic' mean the same thing in every dataset?")
        mode = st.radio("Model family", ["multi", "single", "tfidf"], horizontal=True,
                        format_func=lambda m: {"multi": "multi-task transformer", "single": "single-task transformers", "tfidf": "TF-IDF baseline"}[m])
        st.caption("Targets: TRAC = 'overtly aggressive' (OAG) vs the rest · HOT = abusive or hate-inducing vs clean · Bohra = hate vs no hate. Each head's score is its own 'bad' probability.")
        d = tr[tr["model"] == mode].copy()
        d["head"] = d["source_head"].map(lambda h: f"{TASK_LABEL[h]} head")
        order_t = ["TRAC Facebook", "TRAC Twitter", "HOT tweets", "Bohra tweets"]
        base = alt.Chart(d).encode(x=alt.X("target:N", sort=order_t, title="tested on …", axis=alt.Axis(labelAngle=0)), y=alt.Y("head:N", sort=[f"{TASK_LABEL[h]} head" for h in TASKS], title="… a head trained for"))
        rect = base.mark_rect(cornerRadius=3).encode(color=alt.Color("auroc:Q", scale=alt.Scale(domain=[0.5, 1], range=["#eef5fd", "#0d366b"]), legend=alt.Legend(title="AUROC")), tooltip=["head", "target", alt.Tooltip("auroc:Q", format=".3f"), "n", "positives"])
        text = base.mark_text(fontSize=13).encode(text=alt.Text("auroc:Q", format=".2f"), color=alt.condition("datum.auroc > 0.78", alt.value("white"), alt.value("#0b0b0b")))
        st.altair_chart((rect + text).properties(height=210), use_container_width=True)
        m = tr[(tr["model"] == "multi")]
        indom = m[m["in_domain"]]["auroc"].mean()
        off = m[(~m["in_domain"]) & (m["source_head"] != "sentiment")]["auroc"].mean()
        sent = m[m["source_head"] == "sentiment"]["auroc"].mean()
        if sent >= off - 0.02:
            sent_txt = (f"The sentiment head's &ldquo;is it negative?&rdquo; score reaches <b>{sent:.2f}</b> — about as well as a toxicity head trained on a <i>different</i> dataset ({off:.2f}) — "
                        "a reminder that much of what these corpora call toxic is simply negativity.")
        else:
            sent_txt = f"The sentiment head — &ldquo;is this negative?&rdquo; — manages only <b>{sent:.2f}</b>: angry is not the same as toxic."
        finding(f"AUROC = how often a truly toxic text scores higher than a clean one (0.5 = coin flip, 1 = perfect). A head scored on the dataset it was trained for reaches <b>{indom:.2f}</b> on average; "
                f"applied to the <i>other</i> toxicity datasets it drops to <b>{off:.2f}</b>. {sent_txt} "
                "Each dataset defines toxicity differently (Facebook aggression, Twitter hate, abusive language), which is why a model cannot simply be 'trained on toxicity' once.")
        cag = table("cag_position.csv")
        if cag is not None:
            p = cag[cag["head"] == "offensive"].set_index("aggression_label")["mean_score"]
            st.caption(f"Where does 'covertly aggressive' sit? Average 'offensive' score of the single-task offensiveness model: non-aggressive {p['NAG']:.2f} · covertly aggressive {p['CAG']:.2f} · overtly aggressive {p['OAG']:.2f}.")

    # 2. OOD
    hl = table("headline_results.csv")
    if hl is not None:
        st.markdown("### 2 · New place, new people")
        d = hl[(hl["family"].isin(["multi", "single"])) & hl["eval_set"].isin(["test", "youtube", "test_tw"])]
        d = d.groupby(["task", "eval_set"])[["macro_f1", "ci_lo", "ci_hi"]].mean().reset_index()
        rows = []
        for task, ev, name in (("sentiment", "test", "Sentiment · SentiMix tweets (in-domain)"), ("sentiment", "youtube", "Sentiment · YouTube comments (new platform)"),
                               ("aggression", "test", "Aggression · Facebook comments (in-domain)"), ("aggression", "test_tw", "Aggression · Twitter (new platform)")):
            r = d[(d["task"] == task) & (d["eval_set"] == ev)]
            if len(r):
                rows.append({"setting": name, "macro_f1": float(r["macro_f1"].iloc[0]), "ci_lo": float(r["ci_lo"].iloc[0]), "ci_hi": float(r["ci_hi"].iloc[0]), "kind": "new platform" if "new" in name else "in-domain"})
        if rows:
            df = pd.DataFrame(rows)
            base = alt.Chart(df).encode(y=alt.Y("setting:N", sort=list(df["setting"]), title=None, axis=alt.Axis(labelLimit=330)))
            st.altair_chart((base.mark_bar(cornerRadiusEnd=4, size=22).encode(x=alt.X("macro_f1:Q", scale=alt.Scale(domain=[0, 1]), title="macro-F1"),
                                                                              color=alt.Color("kind:N", scale=alt.Scale(domain=["in-domain", "new platform"], range=[ui.CAT[0], ui.CAT[1]]), legend=alt.Legend(title=None, orient="bottom")))
                             + base.mark_rule(size=2, color="#52514e").encode(x="ci_lo:Q", x2="ci_hi:Q")).properties(height=alt.Step(44)), use_container_width=True)
            a, b = df.iloc[0]["macro_f1"], df.iloc[1]["macro_f1"] if len(df) > 1 else np.nan
            finding(f"Moving the sentiment model from the tweets it learnt on to YouTube comments changes macro-F1 from <b>{a:.2f}</b> to <b>{b:.2f}</b> — the model never saw a YouTube comment, "
                    "and the comments differ in length, emoji use and vocabulary. Expect real-world performance on a new platform to land below the headline numbers.")

    # 3. slices
    sl = table("slice_results.csv")
    if sl is not None:
        st.markdown("### 3 · Who does it fail for?")
        c = st.columns(2)
        task = c[0].selectbox("Task", list(TASK_LABEL), format_func=lambda t: TASK_LABEL[t], key="sl_task")
        kind = c[1].selectbox("Slice by", sorted(sl["slice_type"].unique()), key="sl_kind")
        d = sl[(sl["task"] == task) & (sl["slice_type"] == kind) & (sl["eval_set"] == "test")]
        SLICE_ORDER = {"code-mixing": ["monolingual (CMI 0)", "light (0-10)", "moderate (10-25)", "heavy (25+)"], "length": ["1-8 words", "9-15", "16-30", "31+"],
                       "script": ["roman", "mixed", "devanagari"], "emoji": ["no emoji", "has emoji"]}
        order_s = [v for v in SLICE_ORDER.get(kind, []) if v in set(d["slice"])]
        if len(d):
            d = d.assign(model=d["model"].map({"multi": "transformer", "tfidf": "TF-IDF"}))
            ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("slice:N", title=None, sort=order_s or None, axis=alt.Axis(labelLimit=220)), x=alt.X("macro_f1:Q", scale=alt.Scale(domain=[0, 1]), title="macro-F1"),
                yOffset="model:N", color=alt.Color("model:N", scale=alt.Scale(domain=["transformer", "TF-IDF"], range=[ui.CAT[0], ui.NEUTRAL]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["slice", "model", "n", alt.Tooltip("macro_f1:Q", format=".3f")]).properties(height=78 * d["slice"].nunique() + 40)
            st.altair_chart(ch, use_container_width=True)
            t = d[d["model"] == "transformer"].sort_values("macro_f1")
            finding(f"For {TASK_LABEL[task].lower()}, the weakest slice by {kind} is <b>{t.iloc[0]['slice']}</b> ({t.iloc[0]['macro_f1']:.2f} macro-F1, {int(t.iloc[0]['n'])} texts); the strongest is "
                    f"<b>{t.iloc[-1]['slice']}</b> ({t.iloc[-1]['macro_f1']:.2f}, {int(t.iloc[-1]['n'])} texts). Slices under 30 texts are hidden; small ones are noisy.")

    # 4. robustness
    rb = table("robustness.csv")
    if rb is not None:
        st.markdown("### 4 · Does spelling break it?")
        names = {"lowercase": "all lower-case", "uppercase": "ALL UPPER-CASE", "no_emoji": "emoji removed", "elongate": "elongated words (bahuuut)",
                 "vowel_drop": "SMS-style vowel drop (nhi, kya→ky)", "typo": "random typos", "obfuscate": "masked profanity (ch*tiya)"}
        d = rb.groupby(["perturbation", "model"])[["drop", "flip_rate_on_changed"]].mean().reset_index()
        d["what"] = d["perturbation"].map(names)
        d["model"] = d["model"].map({"transformer": "transformer", "tfidf": "TF-IDF"})
        st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("what:N", sort=alt.EncodingSortField("drop", order="descending"), title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("drop:Q", title="macro-F1 lost (average over tasks)"), yOffset="model:N",
            color=alt.Color("model:N", scale=alt.Scale(domain=["transformer", "TF-IDF"], range=[ui.CAT[0], ui.NEUTRAL]), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["what", "model", alt.Tooltip("drop:Q", format=".3f"), alt.Tooltip("flip_rate_on_changed:Q", format=".1%", title="predictions flipped")]).properties(height=330), use_container_width=True)
        tt = d[d["model"] == "transformer"].sort_values("drop", ascending=False)
        finding(f"Hinglish has no standard spelling, so people write the same word many ways. Each perturbation rewrites the real test texts and re-scores them. "
                f"The biggest hit to the transformer is <b>{tt.iloc[0]['what']}</b> (−{tt.iloc[0]['drop'] * 100:.1f} macro-F1 points, {tt.iloc[0]['flip_rate_on_changed'] * 100:.0f}% of changed texts flip class); "
                f"the mildest is <b>{tt.iloc[-1]['what']}</b> (−{tt.iloc[-1]['drop'] * 100:.1f}).")

    # 5. bias
    bp = table("bias_probe.csv")
    if bp is not None:
        st.markdown("### 5 · Does it punish identity words?")
        d = bp.sort_values("toxicity_mean")
        ref = d[d["group"] == "reference (no identity)"]["toxicity_mean"].mean()
        ch = alt.Chart(d).mark_circle(size=120).encode(
            y=alt.Y("term:N", sort=list(d["term"]), title=None), x=alt.X("toxicity_mean:Q", title="average toxicity score of identical, harmless sentences", scale=alt.Scale(domain=[0, max(0.5, float(d["toxicity_mean"].max()) * 1.1)])),
            color=alt.Color("group:N", legend=alt.Legend(title=None, orient="bottom", columns=3), scale=alt.Scale(range=ui.CAT[:6])),
            tooltip=["term", "group", alt.Tooltip("toxicity_mean:Q", format=".3f"), alt.Tooltip("flagged_share:Q", format=".0%", title="flagged for review")])
        rule = alt.Chart(pd.DataFrame({"x": [ref]})).mark_rule(strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
        st.altair_chart((ch + rule).properties(height=22 * len(d) + 40), use_container_width=True)
        worst, best = d.iloc[-1], d.iloc[0]
        flagged = bp[bp["group"] != "reference (no identity)"]["flagged_share"].mean()
        finding(f"Eight hand-written, harmless sentence templates (&ldquo;Mere <i>X</i> dost bahut acche insaan hain&rdquo;, …) were filled with {len(bp)} identity and reference words. "
                f"A fair detector scores them all alike. Here the spread runs from <b>{best['term']}</b> ({best['toxicity_mean']:.2f}) to <b>{worst['term']}</b> ({worst['toxicity_mean']:.2f}); "
                f"the dashed line is the no-identity reference ({ref:.2f}). On average {flagged * 100:.0f}% of these harmless sentences would be sent to review. "
                "Terms that appear mostly in hateful training examples inherit that toxicity — a known failure of every toxicity classifier.")

    # 6. calibration
    cal = table("calibration.csv")
    if cal is not None:
        st.markdown("### 6 · Can you trust the percentages?")
        fig("calibration")
        finding(f"A model that says '80% sure' should be right about 80% of the time. After fitting one temperature per head on the dev data, the average calibration error drops from "
                f"<b>{cal['ece_raw'].mean() * 100:.1f}%</b> to <b>{cal['ece_calibrated'].mean() * 100:.1f}%</b>. The confidence numbers in the Analyse page use the calibrated version.")

    # 7. mistakes
    er = table("confident_errors.csv")
    if er is not None:
        st.markdown("### 7 · The most confident mistakes")
        st.caption("Texts the model got wrong while being most sure. From the SentiMix, TRAC-1 and YouTube datasets (the ones whose licences allow showing text). Often the 'mistake' is a debatable label.")
        if st.toggle("Show examples (may contain offensive language)", value=False):
            d = er.copy()
            d["confidence"] = (d["confidence"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(d[["dataset", "text", "true", "predicted", "confidence"]], hide_index=True, use_container_width=True, height=380)


# --------------------------------------------------------------------------------------------- data & method
def page_data():
    st.title("Data & method")
    st.markdown("Everything here is **real, published, human-annotated data** — nothing generated or simulated. The model learns four related skills from four datasets.")
    cards = pd.DataFrame([
        {"Skill": "Sentiment", "Dataset": "SentiMix (SemEval-2020 Task 9)", "Text": "Hinglish tweets", "Labels": "negative · neutral · positive", "Split": "official train / dev / test", "Licence": "OpenRAIL (research)"},
        {"Skill": "Aggression", "Dataset": "TRAC-1 Hindi (Kumar et al., 2018)", "Text": "Facebook comments (Roman + Devanagari)", "Labels": "non-aggressive · covertly · overtly aggressive", "Split": "official train / dev / Facebook test / Twitter test", "Licence": "CC BY-NC-SA 4.0"},
        {"Skill": "Offensiveness", "Dataset": "HOT (Mathur et al., ALW 2018)", "Text": "Hinglish tweets", "Labels": "not offensive · abusive · hate-inducing", "Split": "random 70 / 10 / 20, de-duplicated", "Licence": "research use (authors share on request)"},
        {"Skill": "Hate speech", "Dataset": "Bohra et al., 2018", "Text": "Hindi-English tweets", "Labels": "no hate · hate", "Split": "random 70 / 10 / 20, de-duplicated", "Licence": "research use (authors share on request)"},
        {"Skill": "(test only) sentiment", "Dataset": "Hinglish YouTube comments", "Text": "YouTube comments, hand-labelled", "Labels": "negative · neutral · positive", "Split": "never trained on — a new-platform test", "Licence": "CC BY 4.0"},
    ])
    st.table(cards.set_index("Skill"))
    st.caption("Raw datasets are downloaded from their public sources by the pipeline; they are not redistributed with this project. See DATA.md.")

    summ = table("dataset_summary.csv")
    if summ is not None:
        st.markdown("#### How the labels are spread")
        fig("data_class_balance")
        st.markdown("#### What the text looks like")
        a, b = st.columns(2)
        with a:
            fig("data_code_mixing")
        with b:
            fig("data_scripts")
        lid = table("lid_eval.csv")
        if lid is not None:
            acc = float(lid.set_index(lid.columns[0]).loc["macro avg", "f1-score"])
            st.caption(f"Code-mixing index (CMI) = how evenly two languages share a sentence. Words are tagged English or Hindi by a small classifier trained on SentiMix's gold word tags (macro-F1 {acc:.2f} on held-out tokens).")
    aud = table("data_audit.csv")
    if aud is not None:
        with st.expander("Data audit: what was removed, and why"):
            st.dataframe(aud.rename(columns={"rows_after_cleaning": "rows kept", "empty_removed": "empty texts removed", "train_rows_leaking_into_dev_or_test_removed": "train rows that also appear in dev/test (removed)",
                                             "train_duplicates_removed": "duplicate train rows removed"}), hide_index=True, use_container_width=True)
            ov = table("dataset_overlap.csv")
            if ov is not None:
                st.caption("Identical texts shared between datasets (so one dataset's test set doesn't hide inside another's training data):")
                st.dataframe(ov.set_index(ov.columns[0]), use_container_width=True)
    words = table("distinguishing_words.csv")
    if words is not None:
        st.markdown("#### Words that give each class away")
        c = st.columns(2)
        t = c[0].selectbox("Task", list(TASK_LABEL), format_func=lambda x: TASK_LABEL[x], key="w_task")
        cls = c[1].selectbox("Class", TASKS[t], format_func=lambda k: ui.NICE.get(k, k), key="w_cls", index=len(TASKS[t]) - 1)
        w = words[(words["task"] == t) & (words["class"] == cls)].head(15)
        st.caption("Weighted log-odds (Monroe et al. 2008) of words in this class vs the rest of the training set. Words only — may include slurs.")
        if st.toggle("Show the words", value=False):
            st.dataframe(w[["word", "z", "count_in_class", "count_elsewhere"]].rename(columns={"z": "z-score"}), hide_index=True, use_container_width=True)

    st.markdown("### How it works")
    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.6rem">
<div class="hg-find"><b>1 · Clean</b><br>fix encoding junk, drop links and RT, merge @mentions. Spelling and emoji are left alone.</div>
<div class="hg-find"><b>2 · Encode</b><br>a Hinglish-aware pretrained transformer reads the whole comment and produces one vector.</div>
<div class="hg-find"><b>3 · Four heads</b><br>four small classifiers read that vector: sentiment, aggression, offensiveness, hate.</div>
<div class="hg-find"><b>4 · Calibrate</b><br>one temperature per head turns raw scores into honest probabilities.</div>
<div class="hg-find"><b>5 · Decide</b><br>the three toxicity signals are combined into one score and compared with your review / block thresholds.</div>
</div>""", unsafe_allow_html=True)
    st.caption("Training detail: word embeddings frozen, other weights fine-tuned for 4 epochs with AdamW, a linear warm-up/decay schedule, class-weighted loss, and the checkpoint with the best dev macro-F1 kept. "
               "The multi-task model draws each training batch from one dataset, with smaller datasets sampled more often than their size alone would suggest.")


# --------------------------------------------------------------------------------------------- about
def page_about():
    st.title("About")
    cfg = {}
    p = BUNDLE / "config.json"
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    exp = jsonf("export_report.json")
    st.markdown("""
**HinglishGuard** reads Hindi-English ("Hinglish") social-media text and reports its sentiment and how toxic it is.
It is a complete, reproducible NLP project: real data, honest evaluation, a trained model, and this app.
""")
    if cfg:
        st.markdown(f"**Model:** `{cfg.get('encoder')}` with four heads · trained with seed {cfg.get('seed')} · max {cfg.get('max_len')} tokens.")
    if exp:
        st.markdown(f"**Speed on a laptop CPU:** about {exp['fp32_latency_ms']['ms_per_text_single']:.0f} ms per comment one at a time, "
                    f"{exp['fp32_latency_ms']['ms_per_text_batched']:.1f} ms per comment in batches"
                    + (f"; the 8-bit version is {exp['int8_speedup_single']:.1f}× faster" if exp.get("int8_kept") else "") + ".")
    st.markdown("### Use it for")
    st.markdown("- Triaging comments for a human moderator (sorted by toxicity, thresholds you control)\n- Measuring audience sentiment on Indian-language content\n- Teaching and research on code-mixed NLP")
    st.markdown("### Do not use it for")
    st.markdown("- Automatic punishment of people without human review — it is wrong often enough to hurt, and *more* wrong on some groups and spellings (see *Where it breaks*)\n"
                "- Languages and scripts beyond Hindi-English; or fully Devanagari / Urdu-script text, which is thin in the training data\n- Judging a single person's character from one comment")
    st.markdown("### Known limits")
    st.markdown("- The datasets are small, from 2017–2020 social media, and each defines toxicity its own way; annotators disagree and labels are noisy.\n"
                "- Sarcasm, in-group banter and reclaimed slurs are routinely misread.\n- Identity words can raise toxicity scores (measured on the *Where it breaks* page).\n"
                "- Performance on a new platform is lower than the headline numbers.")
    st.markdown("### References")
    st.markdown("""
- Patwa et al. (2020). *SemEval-2020 Task 9: Overview of sentiment analysis of code-mixed tweets.* — SentiMix
- Kumar, Reganti, Bhatia, Maheshwari (2018). *Aggression-annotated corpus of Hindi-English code-mixed data.* LREC — TRAC-1
- Mathur, Sawhney, Ayyar, Shah (2018). *Did you offend me? Classification of offensive tweets in Hinglish language.* ALW2 — HOT
- Bohra, Vijay, Singh, Akhtar, Shrivastava (2018). *A dataset of Hindi-English code-mixed social media text for hate speech detection.*
- Nayak & Joshi (2022). *L3Cube-HingCorpus and HingBERT.* · Khanuja et al. (2021) *MuRIL* · Conneau et al. (2020) *XLM-R* · Devlin et al. (2019) *BERT*
- Gambäck & Das (2016). *Comparing the level of code-switching in corpora.* — Code-Mixing Index
- Guo et al. (2017). *On calibration of modern neural networks.* — temperature scaling
- Monroe, Colaresi, Quinn (2008). *Fightin' words.* — weighted log-odds
""")
