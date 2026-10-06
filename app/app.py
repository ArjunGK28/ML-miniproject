"""Streamlit demo: type a movie review and see what each trained model predicts.

    streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import PAPER_BEST_ACCURACY
from src.evaluate import compare_with_paper, load_all_results
from src.predict import Predictor

EXAMPLES = {
    "Clearly positive": "One of the best films I have seen in years. The acting is superb, the "
                        "story is moving and the ending is perfect. I loved every minute of it.",
    "Clearly negative": "A complete waste of time. The plot is dull, the acting is terrible and "
                        "the dialogue is laughable. I want my two hours back.",
    "Negation": "This movie is not good. I did not enjoy it and I would not recommend it to anyone.",
    "Mixed": "The first half is slow and the script is clumsy, but the performances are wonderful "
             "and the final act is genuinely thrilling. Flawed, but worth watching.",
}

st.set_page_config(page_title="IMDb Sentiment Analysis", layout="wide")
st.title("IMDb Sentiment Analysis")
st.caption("Reproducing and improving on Wu & Shin, \"Machine Learning based classification "
           "for Sentimental analysis of IMDb reviews\" (Stanford CS229).")


@st.cache_resource(show_spinner="Loading trained models ...")
def get_predictor():
    return Predictor()


def show_predictions(predictor, text):
    results = {name: predictor.predict(name, text) for name in predictor.names}
    positive = sum(r["label"] for r in results.values())
    st.subheader(f"{positive} of {len(results)} models say this review is positive")

    rows = []
    for name, result in results.items():
        bundle = predictor.bundles[name]
        if result["probability"] is not None:
            confidence = f"{max(result['probability'], 1 - result['probability']):.1%}"
        else:
            confidence = f"margin {result['score']:+.2f}"
        rows.append({"Model": name, "Group": bundle.get("group", ""),
                     "Prediction": "Positive" if result["label"] else "Negative",
                     "Confidence": confidence, "Features": bundle["kind"],
                     "Test accuracy": bundle.get("test_accuracy")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Test accuracy": st.column_config.NumberColumn(format="%.3f")})

    st.subheader("Why? The n-grams that mattered most")
    st.caption("Bars to the right push towards positive, bars to the left towards negative.")
    explainable = [name for name in predictor.names if predictor.can_explain(name)]
    for column, name in zip(st.columns(max(len(explainable), 1)), explainable):
        terms = pd.DataFrame(predictor.explain(name, text, top=10), columns=["ngram", "weight"])
        terms["direction"] = terms.weight.map(lambda w: "positive" if w > 0 else "negative")
        with column:
            st.markdown(f"**{name}**")
            st.bar_chart(terms, x="ngram", y="weight", color="direction", horizontal=True,
                         height=320)


def show_results():
    comparison = compare_with_paper()
    done = comparison.dropna(subset=["accuracy"])
    st.subheader("Reproduction of the paper's Table 2")
    if done.empty:
        st.info("No reproduction results yet. Run the scripts in `src/experiments/`.")
    else:
        st.caption("`paper_pos` and `paper_neg` are the columns the paper labels positive and "
                   "negative precision. They line up with our per-class recall, not our precision.")
        table = comparison[["model", "vectorization", "paper_accuracy", "accuracy", "accuracy_diff",
                            "paper_pos", "pos_recall", "pos_precision",
                            "paper_neg", "neg_recall", "neg_precision"]]
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in table.columns[2:]})
        short = done.model.replace({"Logistic Regression": "LR", "Naive Bayes": "NB",
                                    "Random Forest": "RF"})
        chart = done.assign(setting=short + " / " + done.vectorization)
        chart = chart.rename(columns={"paper_accuracy": "Paper", "accuracy": "Ours"})
        st.bar_chart(chart, x="setting", y=["Paper", "Ours"], stack=False, horizontal=True,
                     height=420)

    st.subheader("Improvements")
    improved = load_all_results("improve")
    if improved.empty:
        st.info("No improvement results yet.")
        return
    improved = improved.sort_values("accuracy", ascending=False)
    improved["vs_paper_best"] = improved.accuracy - PAPER_BEST_ACCURACY
    st.caption(f"The paper's best accuracy is {PAPER_BEST_ACCURACY:.1%} (DNN, binary 3-grams).")
    table = improved[["model", "vectorization", "cleaning", "C", "accuracy", "vs_paper_best",
                      "pos_precision", "neg_precision", "pos_recall", "neg_recall", "f1"]]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in table.columns[4:]})


predictor = get_predictor()
# A radio rather than st.tabs, so only the selected view is drawn: a table first drawn
# inside a hidden tab came out collapsed.
view = st.radio("View", ["Try a review", "Results vs paper"], horizontal=True,
                label_visibility="collapsed")

if view == "Results vs paper":
    show_results()
elif not predictor.names:
    st.warning("No trained models found in `models/`. Run "
               "`python -m src.experiments.linear_models` first.")
else:
    choice = st.selectbox("Start from an example", list(EXAMPLES))
    review = st.text_area("Movie review", EXAMPLES[choice], height=140)
    if review.strip():
        show_predictions(predictor, review)
