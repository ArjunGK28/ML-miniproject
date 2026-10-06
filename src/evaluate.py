"""Metrics and results tables shared by every experiment."""
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import PAPER_TABLE2, RESULT_COLUMNS, RESULTS_DIR


def compute_metrics(y_true, y_pred):
    """The paper's three metrics (section 4) plus per-class recall and F1.

    Precision follows the formulas printed in the paper: TP / (TP + FP) and TN / (TN + FN).
    Recall is included because the numbers in the paper's Table 2 behave like recall:
    in 15 of its 18 rows accuracy equals the mean of the two "precision" columns to within
    rounding, which holds for recall on a balanced test set but not for precision.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pos_precision": tp / (tp + fp) if tp + fp else 0.0,
        "neg_precision": tn / (tn + fn) if tn + fn else 0.0,
        "pos_recall": tp / (tp + fn),
        "neg_recall": tn / (tn + fp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def save_results(rows, name):
    """Write one experiment's rows to results/<name>.csv with the shared column order."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def load_all_results(prefix=""):
    """Concatenate every results CSV whose file name starts with `prefix`."""
    frames = [pd.read_csv(path).assign(experiment=path.stem)
              for path in sorted(RESULTS_DIR.glob(f"{prefix}*.csv"))]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["experiment"])
    return pd.concat(frames, ignore_index=True)


def paper_table():
    return pd.DataFrame(PAPER_TABLE2, columns=["model", "vectorization", "paper_C",
                                               "paper_pos", "paper_neg", "paper_accuracy"])


def compare_with_paper():
    """Table 2 side by side with our reproduction (rows we have not run yet stay empty)."""
    ours = load_all_results("reproduction")
    merged = paper_table().merge(ours, on=["model", "vectorization"], how="left")
    merged["accuracy_diff"] = merged["accuracy"] - merged["paper_accuracy"]
    return merged
