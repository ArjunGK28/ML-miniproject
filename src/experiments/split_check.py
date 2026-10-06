"""Why the validation split is grouped by movie.

Trains the same models with a random validation split and with a movie-grouped one, and
prints each validation accuracy next to the accuracy the model reaches on the test set.

    python -m src.experiments.split_check            # writes results/validation_split_check.csv
    python -m src.experiments.split_check --quick    # smoke test (too few reviews per movie
                                                     # in the subset to show the effect)
"""
import argparse

import pandas as pd

from src.config import RESULTS_DIR
from src.experiments.linear_models import MODELS
from src.experiments.runner import prepare, run


def main(quick=False):
    rows = []
    for grouped in (False, True):
        print("validation split:", "grouped by movie" if grouped else "random")
        data = prepare("paper", quick=quick, grouped=grouped, vocab="full")
        for name in ("Logistic Regression", "Naive Bayes"):
            make_model, grid = MODELS[name]
            row, _ = run(name, make_model, grid, data, "binary")
            rows.append({"split": "grouped by movie" if grouped else "random", "model": name,
                         "C": row["C"], "val_accuracy": row["val_accuracy"],
                         "test_accuracy": row["accuracy"],
                         "val_minus_test": row["val_accuracy"] - row["accuracy"]})
        del data
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not quick:
        table.to_csv(RESULTS_DIR / "validation_split_check.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    main(parser.parse_args().quick)
