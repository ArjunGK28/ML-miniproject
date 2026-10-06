"""Reproduce the Logistic Regression, SVM and Naive Bayes rows of the paper's Table 2.

    python -m src.experiments.linear_models            # full run, writes results/ and models/
    python -m src.experiments.linear_models --quick    # smoke test on a 2k-review subset
"""
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import C_GRID, SEED, VECTORIZATIONS
from src.experiments.runner import prepare, report, run
from src.predict import save_featurizer, save_model

# Settings from section 3.3 of the paper. L2 is the default penalty of both linear models.
MODELS = {
    "Logistic Regression": (lambda C: LogisticRegression(C=C, tol=1e-4, solver="liblinear",
                                                         random_state=SEED), C_GRID),
    "SVM": (lambda C: LinearSVC(C=C, tol=1e-4, random_state=SEED), C_GRID),
    "Naive Bayes": (lambda _: MultinomialNB(alpha=1.0), [None]),
}


def main(quick=False):
    data = prepare("paper", quick=quick, vocab="full")
    rows, best = [], {}
    for kind in VECTORIZATIONS:
        for name, (make_model, grid) in MODELS.items():
            row, model = run(name, make_model, grid, data, kind)
            rows.append(row)
            # Keep the best vectorization per model for the demo, judged on validation data.
            if name not in best or row["val_accuracy"] > best[name][0]["val_accuracy"]:
                best[name] = (row, model)
    report(rows, "reproduction_linear", quick)

    if not quick:
        save_featurizer("paper_full", data.featurizer)
        for name, (row, model) in best.items():
            save_model(name, model, "paper_full", row["vectorization"], "paper",
                       group="Paper reproduction", test_accuracy=row["accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    main(parser.parse_args().quick)
