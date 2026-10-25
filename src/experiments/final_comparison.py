"""Final comparison: ensembles with the transformers, confidence intervals, significance tests.

1. Members: NB-LR (our best n-gram model) is refitted here; the fine-tuned transformers
   are read from models/scores/ (run src/experiments/transformer_models.py first).
2. Ensembles: every combination of two or more members, averaging their positive-class
   probabilities. The combination is chosen on the grouped validation split; the test set
   is used once, at the end.
3. Statistics on the 25,000 test reviews, for every model in results/:
   - a 95% confidence interval for accuracy (Wilson score interval);
   - a one-sided two-proportion z-test of "accuracy > 0.906", the paper's best;
   and for the models trained here, an exact McNemar test against our reproduction of the
   paper's Logistic Regression (binary features, 0.900 in both the paper and ours). McNemar
   compares two models on the same reviews, so it is stronger than comparing two numbers.
   We cannot run it against the paper's DNN because its predictions are not published.

Writes results/final_comparison.csv and results/final_comparison.png.

    python -m src.experiments.final_comparison
"""
import itertools
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.exceptions import ConvergenceWarning

from src.config import PAPER_BEST_ACCURACY, RESULTS_DIR
from src.evaluate import compute_metrics, load_all_results
from src.experiments.improve_ensemble import fit_member
from src.experiments.improve_features import NB_MODELS, RICH
from src.experiments.linear_models import MODELS as PAPER_MODELS
from src.experiments.runner import prepare
from src.experiments.transformer_models import MODELS as TRANSFORMERS, SCORES_DIR, slug

Z95 = 1.959964


def wilson_interval(accuracy, n):
    """95% Wilson score interval for a proportion."""
    centre = (accuracy + Z95 ** 2 / (2 * n)) / (1 + Z95 ** 2 / n)
    half = Z95 * np.sqrt(accuracy * (1 - accuracy) / n + Z95 ** 2 / (4 * n ** 2)) / (1 + Z95 ** 2 / n)
    return centre - half, centre + half


def p_above_paper(accuracy, n):
    """One-sided two-proportion z-test: is this accuracy above the paper's best?

    The paper's 0.906 was also measured on the 25,000 test reviews, so it has sampling
    noise too; both are counted. (A paired test would be more powerful, but needs the
    paper's per-review predictions.)
    """
    p = PAPER_BEST_ACCURACY
    se = np.sqrt(accuracy * (1 - accuracy) / n + p * (1 - p) / n)
    return float(norm.sf((accuracy - p) / se))


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on the reviews where exactly one of the two models is right."""
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    if a_only + b_only == 0:
        return 1.0
    return binomtest(a_only, a_only + b_only, 0.5).pvalue


def collect_members():
    members = {}
    print("A. NB-LR on binary n-grams (minimal cleaning, rich tokens)")
    data = prepare("minimal", vocab="full", tokens="rich")
    data.cleaning = RICH
    fit_member("NB-LR", *NB_MODELS["NB-LR"], data, "binary", members)
    del data

    print("B. Fine-tuned transformers (scores saved by transformer_models.py)")
    for name in TRANSFORMERS:
        path = SCORES_DIR / f"{slug(name)}.npz"
        if not path.exists():
            print(f"  {name}: no saved scores, skipped")
            continue
        saved = np.load(path)
        assert (saved["y_test"] == members["NB-LR"]["y_test"]).all(), "test order differs"
        assert (saved["y_val"] == members["NB-LR"]["y_val"]).all(), "validation split differs"
        members[f"{name} (fine-tuned)"] = {"val": saved["val"], "test": saved["test"],
                         "y_val": saved["y_val"], "y_test": saved["y_test"]}
        print(f"  {name}: test accuracy {((saved['test'] > 0.5) == saved['y_test']).mean():.4f}")
    return members


def best_ensemble(members):
    names = list(members)
    y_val = members[names[0]]["y_val"]
    best = None
    print("C. Ensembles, chosen on validation data")
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(names, size):
            val_acc = float(((np.mean([members[n]["val"] for n in combo], axis=0) > 0.5)
                             == y_val).mean())
            print(f"  {val_acc:.4f}  {' + '.join(combo)}")
            if best is None or val_acc > best[0]:
                best = (val_acc, combo)
    return best


def paper_baseline():
    """Test predictions of our reproduction of the paper's Logistic Regression (binary)."""
    print("D. Paper's Logistic Regression on binary n-grams (baseline for McNemar)")
    data = prepare("paper", vocab="full")
    store = {}
    fit_member("Logistic Regression", *PAPER_MODELS["Logistic Regression"], data, "binary", store)
    return store["Logistic Regression"]["test"] > 0.5


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    members = collect_members()
    y_test = members["NB-LR"]["y_test"]
    n = len(y_test)

    predictions = {name: m["test"] > 0.5 for name, m in members.items()}
    val_accuracy = {name: float(((m["val"] > 0.5) == m["y_val"]).mean())
                    for name, m in members.items()}
    if len(members) > 1:
        val_acc, combo = best_ensemble(members)
        name = "Ensemble (" + " + ".join(combo) + ")"
        predictions[name] = np.mean([members[c]["test"] for c in combo], axis=0) > 0.5
        val_accuracy[name] = val_acc
        print(f"  chosen: {name}")

    baseline_correct = paper_baseline() == y_test
    nblr_correct = predictions["NB-LR"] == y_test

    rows = []
    for name, predicted in predictions.items():
        correct = predicted == y_test
        accuracy = float(correct.mean())
        low, high = wilson_interval(accuracy, n)
        rows.append({"model": name, "group": "this script", "val_accuracy": val_accuracy[name],
                     **compute_metrics(y_test, predicted.astype(int)),
                     "ci_low": low, "ci_high": high,
                     "p_above_paper_best": p_above_paper(accuracy, n),
                     "mcnemar_p_vs_paper_lr": mcnemar(correct, baseline_correct),
                     "mcnemar_p_vs_nblr": mcnemar(correct, nblr_correct) if name != "NB-LR" else None})

    # Every earlier result, with the same interval and test (no per-review predictions needed).
    earlier = load_all_results().dropna(subset=["accuracy"])
    for r in earlier.itertuples():
        if r.experiment == "validation_split_check" or r.model in predictions:
            continue
        low, high = wilson_interval(r.accuracy, n)
        rows.append({"model": r.model, "vectorization": r.vectorization, "cleaning": r.cleaning,
                     "group": r.experiment, "val_accuracy": r.val_accuracy,
                     "accuracy": r.accuracy, "ci_low": low, "ci_high": high,
                     "p_above_paper_best": p_above_paper(r.accuracy, n)})

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    table.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)
    shown = ["model", "group", "accuracy", "ci_low", "ci_high", "p_above_paper_best",
             "mcnemar_p_vs_paper_lr", "mcnemar_p_vs_nblr"]
    print(table[shown].head(15).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    plot(table)
    print("wrote results/final_comparison.csv and results/final_comparison.png")


def plot(table, path=RESULTS_DIR / "final_comparison.png"):
    """One bar per model family: the paper's best, our reproduction, our improvements."""
    paper = load_all_results("reproduction")
    picks = []
    for model in dict.fromkeys(paper.model):   # best feature type of each reproduced model
        r = paper[paper.model == model].sort_values("accuracy").iloc[-1]
        picks.append((f"{model} ({r.vectorization}), paper's method", r.accuracy, "Paper's models, our reproduction"))
    earlier = {"NB-LR": "NB-LR",
               "Ensemble (NB-LR + SVM (tf-idf, wide C grid))": "NB-LR + SVM ensemble (earlier)"}
    for model, label in earlier.items():
        picks.append((label, table[table.model == model].accuracy.max(), "Our n-gram improvements"))
    for r in table[table.group == "this script"].itertuples():
        if r.model not in earlier:
            label = r.model.replace(" (fine-tuned)", "")
            if label.startswith("Ensemble ("):
                label = label[len("Ensemble ("):-1] + " ensemble"
            picks.append((label + (" (fine-tuned)" if "ensemble" not in label else ""),
                          r.accuracy, "New: pretrained transformers"))

    colours = {"Paper's models, our reproduction": "#9aa5b8",
               "Our n-gram improvements": "#5b8fd1",
               "New: pretrained transformers": "#1f4e8c"}
    picks.sort(key=lambda p: p[1])
    labels, accuracy, group = zip(*picks)
    accuracy = np.array(accuracy)
    low, high = wilson_interval(accuracy, 25000)

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(picks) + 1.4))
    y = np.arange(len(picks))
    ax.barh(y, accuracy, color=[colours[g] for g in group],
            xerr=[accuracy - low, high - accuracy], capsize=2, ecolor="#333333")
    for yi, a, h in zip(y, accuracy, high):
        ax.text(h + 0.002, yi, f"{a:.1%}", va="center", fontsize=8)
    ax.axvline(PAPER_BEST_ACCURACY, color="black", linestyle="--", linewidth=1)
    ax.text(PAPER_BEST_ACCURACY, len(picks) - 0.2, f" paper's best ({PAPER_BEST_ACCURACY:.1%})",
            fontsize=8, va="bottom")
    ax.set_ylim(-0.6, len(picks) + 0.4)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0.80, max(0.97, accuracy.max() + 0.02))
    ax.set_xlabel("Accuracy on the 25,000 test reviews (bars: 95% confidence interval)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, colours.keys(), fontsize=8, loc="lower right")
    ax.set_title("Every model against the paper's best", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
