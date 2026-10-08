"""Figures made from the results CSVs (nothing is trained here).

    python -m src.plots

Writes results/reproduction_vs_paper.png (the paper's Table 2 next to ours) and
results/improvements.png (our best models against the paper's best accuracy).
Rows that have not been run yet are simply left out.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PAPER_BEST_ACCURACY, RESULTS_DIR, VECTORIZATIONS
from src.evaluate import compare_with_paper, load_all_results


def plot_reproduction(path=RESULTS_DIR / "reproduction_vs_paper.png"):
    table = compare_with_paper()
    models = list(dict.fromkeys(table.model))
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 3.6), sharey=True)
    x = np.arange(len(VECTORIZATIONS))
    for ax, model in zip(np.atleast_1d(axes), models):
        part = table[table.model == model].set_index("vectorization").reindex(VECTORIZATIONS)
        ax.bar(x - 0.2, part.paper_accuracy, 0.4, label="Paper", color="#9aa5b8")
        ax.bar(x + 0.2, part.accuracy.fillna(0), 0.4, label="Ours", color="#2f6db5")
        ax.set_title(model, fontsize=9)
        ax.set_xticks(x, VECTORIZATIONS, fontsize=8)
        ax.set_ylim(0.78, 0.93)
    np.atleast_1d(axes)[0].set_ylabel("Test accuracy")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Reproduction of Table 2 (missing bars = not run yet)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_improvements(path=RESULTS_DIR / "improvements.png", top=10):
    rows = load_all_results("improve").dropna(subset=["accuracy"])
    rows = rows.sort_values("accuracy", ascending=False).head(top).iloc[::-1]
    labels = [f"{r.model} | {r.vectorization}" for r in rows.itertuples()]
    colours = ["#2f6db5" if a > PAPER_BEST_ACCURACY else "#9aa5b8" for a in rows.accuracy]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    ax.barh(labels, rows.accuracy, color=colours)
    ax.axvline(PAPER_BEST_ACCURACY, color="black", linestyle="--", linewidth=1,
               label=f"Paper's best ({PAPER_BEST_ACCURACY:.3f})")
    ax.set_xlim(0.88, max(0.93, rows.accuracy.max() + 0.005))
    ax.set_xlabel("Test accuracy")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Best models from the improvement experiments")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for path in (plot_reproduction(), plot_improvements()):
        print("wrote", path.relative_to(path.parent.parent))
