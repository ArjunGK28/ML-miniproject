"""Reproduce the Random Forest, Boosting and DNN rows of the paper's Table 2.

    python -m src.experiments.nonlinear_models                      # all 9 rows (3 models x 3 vectorizations)
    python -m src.experiments.nonlinear_models --quick              # smoke test on a 2k-review subset
    python -m src.experiments.nonlinear_models --models "Random Forest" --kinds binary
    python -m src.experiments.nonlinear_models --min-df 50          # smaller vocabulary = faster

These three models cannot handle millions of features, so they use the "reduced" vocabulary:
only the n-grams that occur in at least --min-df reviews (default 5, see src/config.py).
The settings are the ones listed in section 3.3 of the paper. The paper gives no
hyperparameter grid for these models, so nothing is tuned and the validation step is skipped.
Results are saved after every fit to results/reproduction_nonlinear.csv, so a long run that
is interrupted keeps what it has finished, and re-running a pair replaces only that row.
"""
import argparse
import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from src.config import REDUCED_MIN_DF, RESULTS_DIR, SEED, VECTORIZATIONS
from src.evaluate import save_results
from src.experiments.runner import prepare, report, run

NAME = "reproduction_nonlinear"
DNN_ALPHA = 0.0001   # the paper's "regularization" column for the DNN

# Section 3.3 of the paper. Boosting is listed last because it is by far the slowest.
MODELS = {
    "Random Forest": lambda _: RandomForestClassifier(
        n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
        n_jobs=-1, random_state=SEED),
    "DNN": lambda _: MLPClassifier(
        hidden_layer_sizes=(30, 30, 20, 10, 10), activation="logistic", alpha=DNN_ALPHA,
        early_stopping=True, solver="adam", random_state=SEED),
    "Boosting": lambda _: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, criterion="squared_error", min_samples_split=2,
        random_state=SEED),
}


def save_merged(rows):
    """Save `rows`, keeping earlier results for (model, vectorization) pairs not re-run now."""
    path = RESULTS_DIR / f"{NAME}.csv"
    if path.exists():
        rerun = {(r["model"], r["vectorization"]) for r in rows}
        old = pd.read_csv(path).to_dict("records")
        rows = [r for r in old if (r["model"], r["vectorization"]) not in rerun] + rows
    save_results(rows, NAME)


def main(quick=False, models=None, kinds=None, min_df=REDUCED_MIN_DF):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    models = models or list(MODELS)
    kinds = kinds or list(VECTORIZATIONS)
    data = prepare("paper", quick=quick, vocab="reduced", min_df=min_df)
    rows = []
    for kind in kinds:
        for name in models:
            row, _ = run(name, MODELS[name], None, data, kind)   # grid=None: no validation step
            if name == "DNN":
                row["C"] = DNN_ALPHA
            rows.append(row)
            if not quick:
                save_merged([row])
    report(rows, NAME, quick=True)   # prints the table; the file is already saved above


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="run only these models")
    parser.add_argument("--kinds", nargs="+", choices=VECTORIZATIONS, help="run only these vectorizations")
    parser.add_argument("--min-df", type=int, default=REDUCED_MIN_DF,
                        help="keep n-grams found in at least this many reviews (default %(default)s)")
    args = parser.parse_args()
    main(args.quick, args.models, args.kinds, args.min_df)
