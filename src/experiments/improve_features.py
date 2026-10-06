"""Improvements on the paper that only change the features and the linear models.

1. Cleaning ablation: the paper's models and C grid on binary n-grams with
   - NLTK's full stop-word list ("paper_nltk"),
   - minimal cleaning (stop words, negations and numbers are kept), and
   - minimal cleaning with rich tokens (one-letter words, "!" and "?" are kept too).
2. NB-SVM / NB-LR: linear models on Naive Bayes log-count ratios (Wang & Manning, 2012).
3. A wider C grid: the paper stops at C = 1, which over-regularises tf-idf features.

Steps 2 and 3 use minimal cleaning with rich tokens.

    python -m src.experiments.improve_features            # full run
    python -m src.experiments.improve_features --quick    # smoke test on a 2k-review subset
"""
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.config import C_GRID, SEED
from src.experiments.linear_models import MODELS
from src.experiments.runner import prepare, report, run
from src.nbsvm import NBSVM
from src.predict import save_featurizer, save_model

WIDE_C_GRID = C_GRID + [4, 16, 64]
RICH = "minimal + rich tokens"

NB_MODELS = {
    "NB-SVM": (lambda C: NBSVM(C=C, base="svm", random_state=SEED), C_GRID),
    "NB-LR": (lambda C: NBSVM(C=C, base="lr", random_state=SEED), WIDE_C_GRID),
    "NB-SVM (interpolated, beta=0.25)":
        (lambda C: NBSVM(C=C, beta=0.25, base="svm", random_state=SEED), C_GRID),
}
WIDE_GRID_MODELS = {
    "Logistic Regression (wide C grid)":
        (lambda C: LogisticRegression(C=C, tol=1e-4, solver="liblinear", random_state=SEED),
         WIDE_C_GRID),
    "SVM (wide C grid)": (lambda C: LinearSVC(C=C, tol=1e-4, random_state=SEED), WIDE_C_GRID),
}


def main(quick=False):
    rows, best = [], None

    def record(models, kind):
        nonlocal best
        for name, (make_model, grid) in models.items():
            row, model = run(name, make_model, grid, data, kind)
            rows.append(row)
            if data.cleaning == RICH and (best is None
                                          or row["val_accuracy"] > best[0]["val_accuracy"]):
                best = (row, model)

    print("1. Paper's models on binary n-grams with other cleaning")
    for cleaning in ("paper_nltk", "minimal"):
        data = prepare(cleaning, quick=quick, vocab="full")
        record(MODELS, "binary")
        del data    # release these feature matrices before building the next ones
    data = prepare("minimal", quick=quick, vocab="full", tokens="rich")
    data.cleaning = RICH
    record(MODELS, "binary")

    print("2. NB-SVM / NB-LR on binary n-grams")
    record(NB_MODELS, "binary")

    print("3. Wider C grid on tf-idf")
    record(WIDE_GRID_MODELS, "tfidf")

    report(rows, "improve_features", quick)

    if not quick:
        row, model = best
        save_featurizer("minimal_rich", data.featurizer)
        save_model(f"{row['model']} (improved)", model, "minimal_rich", row["vectorization"],
                   "minimal", group="Improvement", test_accuracy=row["accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    main(parser.parse_args().quick)
