"""Improvements on the paper that use a better neural network and an ensemble.

1. Better MLP. The paper's DNN has five narrow sigmoid layers (30, 30, 20, 10, 10). We use
   one wide ReLU layer instead, on binary n-grams with the lighter cleaning that worked
   best in improve_features.py. The L2 strength is chosen on the validation split from
   MLP_ALPHAS (to keep the run short we use a single value, so nothing is really compared).
2. Ensemble. We average the positive-class scores of four models (NB-LR, NB-SVM, SVM on
   tf-idf with the wide C grid, and the MLP). Which of the 11 possible combinations to use
   is chosen on the validation split; the test set is only used once, at the end.
   Probabilities are used where a model has them; for a linear SVM we pass its decision
   value through a sigmoid, which is only an approximate probability.
3. Extra metrics: ROC-AUC for every model (results/improve_ensemble_auc.csv) and ROC curves
   (results/roc_curves.png).

    python -m src.experiments.improve_ensemble            # full run (slow: expect tens of minutes)
    python -m src.experiments.improve_ensemble --quick    # smoke test on a 2k-review subset
"""
import argparse
import itertools
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

from src.config import RESULTS_DIR, SEED
from src.evaluate import compute_metrics
from src.experiments.improve_features import NB_MODELS, RICH, WIDE_GRID_MODELS
from src.experiments.runner import prepare, report, run

MLP_MIN_DF = 50          # keep n-grams found in at least this many reviews (smaller = faster)
MLP_ALPHAS = [0.001]     # L2 strengths compared on the validation split; add more to compare


def make_mlp(alpha):
    return MLPClassifier(hidden_layer_sizes=(128,), activation="relu", alpha=alpha,
                         early_stopping=True, n_iter_no_change=3, max_iter=30,
                         random_state=SEED)


def positive_score(model, X):
    """Score for the positive class in [0, 1]: predict_proba if there is one, else a sigmoid."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


def fit_member(name, make_model, grid, data, kind, store):
    """Train one model with `run`, then keep its validation and test scores for the ensemble."""
    row, model = run(name, make_model, grid, data, kind)
    # Validation scores come from a copy fitted on the 80% split only, with the chosen setting.
    on_fit = make_model(row["C"]).fit(data.X_fit[kind], data.y_fit)
    store[name] = {"row": row, "val": positive_score(on_fit, data.X_val[kind]),
                   "test": positive_score(model, data.X_test[kind]),
                   "y_val": data.y_val, "y_test": data.y_test}


def main(quick=False):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    members = {}

    print("A. Linear members on binary / tf-idf n-grams (full vocabulary)")
    data = prepare("minimal", quick=quick, vocab="full", tokens="rich")
    data.cleaning = RICH
    fit_member("NB-LR", *NB_MODELS["NB-LR"], data, "binary", members)
    fit_member("NB-SVM", *NB_MODELS["NB-SVM"], data, "binary", members)
    fit_member("SVM (tf-idf, wide C grid)", *WIDE_GRID_MODELS["SVM (wide C grid)"], data, "tfidf", members)
    del data   # free the multi-million-column matrices before building the MLP's

    print("B. Better MLP (one wide ReLU layer, reduced vocabulary)")
    data = prepare("minimal", quick=quick, vocab="reduced", min_df=MLP_MIN_DF, tokens="rich")
    data.cleaning = RICH
    fit_member("MLP (ReLU, 128 units)", make_mlp, MLP_ALPHAS, data, "binary", members)
    del data

    names = list(members)
    y_val, y_test = members[names[0]]["y_val"], members[names[0]]["y_test"]
    assert all((members[n]["y_test"] == y_test).all() for n in names), "test sets differ"

    print("C. Ensemble: choose the combination on validation data")
    best = None
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(names, size):
            val_score = np.mean([members[n]["val"] for n in combo], axis=0)
            val_acc = float(((val_score > 0.5) == y_val).mean())
            print(f"  {val_acc:.4f}  {' + '.join(combo)}")
            if best is None or val_acc > best[0]:
                best = (val_acc, combo)
    val_acc, combo = best
    ensemble_test = np.mean([members[n]["test"] for n in combo], axis=0)
    ensemble_name = "Ensemble (" + " + ".join(combo) + ")"
    ensemble_row = {"model": ensemble_name, "vectorization": "mixed", "cleaning": RICH, "C": None,
                    "n_features": None, "val_accuracy": val_acc,
                    **compute_metrics(y_test, (ensemble_test > 0.5).astype(int)),
                    "train_seconds": round(sum(members[n]["row"]["train_seconds"] for n in combo), 1)}
    print(f"  chosen on validation: {ensemble_name}  test accuracy {ensemble_row['accuracy']:.4f}")

    rows = [members[n]["row"] for n in names] + [ensemble_row]
    report(rows, "improve_ensemble", quick)

    scores = {n: members[n]["test"] for n in names}
    scores[ensemble_name] = ensemble_test
    auc = pd.DataFrame([{"model": n, "roc_auc": roc_auc_score(y_test, s),
                         "accuracy": ((s > 0.5) == y_test).mean()} for n, s in scores.items()])
    print("\nROC-AUC on the test set\n" + auc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not quick:
        auc.to_csv(RESULTS_DIR / "improve_ensemble_auc.csv", index=False)
        plt.figure(figsize=(6, 5))
        for n, s in scores.items():
            fpr, tpr, _ = roc_curve(y_test, s)
            plt.plot(fpr, tpr, label=f"{n[:40]} (AUC {roc_auc_score(y_test, s):.3f})")
        plt.plot([0, 1], [0, 1], "k:", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curves on the 25,000 test reviews")
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
        print("wrote results/improve_ensemble_auc.csv and results/roc_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    main(parser.parse_args().quick)
