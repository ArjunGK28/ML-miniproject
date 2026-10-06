"""Shared experiment loop: build features, tune on a validation split, test once."""
import time
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import SEED, VAL_SIZE
from src.evaluate import compute_metrics, save_results
from src.features import Featurizer
from src.preprocess import load_clean


def prepare(cleaning="paper", quick=False, **featurizer_options):
    """Load the reviews and build two sets of feature matrices.

    fit/val  : 80/20 split of the training set, vocabulary fitted on the 80% only.
               Used to choose hyperparameters without touching the test set.
    train/test: vocabulary fitted on the whole training set. Used for the reported numbers.

    The validation split is grouped by movie. The dataset's train and test sets share no
    movies, so a random split (same movie on both sides) would overestimate accuracy.

    Each X_* is a dict {"binary": ..., "count": ..., "tfidf": ...}.
    """
    train, test = load_clean("train", cleaning, quick), load_clean("test", cleaning, quick)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(train.text, train.label, groups=train.movie))
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]

    selection = Featurizer(**featurizer_options)
    X_fit, X_val = selection.fit_transform(fit.text), selection.transform(val.text)
    featurizer = Featurizer(**featurizer_options)
    X_train, X_test = featurizer.fit_transform(train.text), featurizer.transform(test.text)
    print(f"[{cleaning} cleaning] {len(train)} train / {len(test)} test reviews, "
          f"{featurizer.n_features:,} features")
    return SimpleNamespace(cleaning=cleaning, featurizer=featurizer,
                           X_fit=X_fit, y_fit=fit.label.to_numpy(),
                           X_val=X_val, y_val=val.label.to_numpy(),
                           X_train=X_train, y_train=train.label.to_numpy(),
                           X_test=X_test, y_test=test.label.to_numpy())


def run(name, make_model, grid, data, kind):
    """Train one (model, vectorization) pair and return (results row, fitted model).

    make_model(value) builds an unfitted estimator. `grid` lists the hyperparameter values
    to compare on the validation split; pass None to skip validation for slow models.
    """
    best_value, best_val = None, float("nan")
    if grid is not None:
        scores = [make_model(value).fit(data.X_fit[kind], data.y_fit)
                  .score(data.X_val[kind], data.y_val) for value in grid]
        best_val = max(scores)
        best_value = grid[scores.index(best_val)]

    start = time.perf_counter()
    model = make_model(best_value).fit(data.X_train[kind], data.y_train)
    seconds = time.perf_counter() - start
    row = {"model": name, "vectorization": kind, "cleaning": data.cleaning, "C": best_value,
           "n_features": data.featurizer.n_features, "val_accuracy": best_val,
           **compute_metrics(data.y_test, model.predict(data.X_test[kind])),
           "train_seconds": round(seconds, 1)}
    print(f"  {name:<22} {kind:<7} C={best_value!s:<5} val={best_val:.4f} "
          f"test={row['accuracy']:.4f} ({seconds:.0f}s)")
    return row, model


def report(rows, name, quick):
    """Print the results table; save it to results/<name>.csv unless this is a --quick run."""
    df = pd.DataFrame(rows) if quick else save_results(rows, name)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if quick:
        print("\n--quick run on a small subset: nothing was saved.")
    return df
