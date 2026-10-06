"""Shared paths, constants and the paper's reported numbers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

SEED = 42
VAL_SIZE = 0.2          # share of the training set held out to choose hyperparameters
QUICK_N = 2000          # reviews per split when a script is run with --quick

NGRAM_RANGE = (1, 3)    # the paper's "3 grams"
REDUCED_MIN_DF = 5      # vocabulary cut used for the tree models and the DNN
VECTORIZATIONS = ["binary", "count", "tfidf"]
C_GRID = [0.01, 0.05, 0.25, 0.5, 1]   # section 3.3 of the paper

RESULT_COLUMNS = [
    "model", "vectorization", "cleaning", "C", "n_features", "val_accuracy",
    "pos_precision", "neg_precision", "accuracy", "f1", "train_seconds",
]

# Table 2 of the paper: (model, vectorization, regularization, pos precision, neg precision, accuracy)
PAPER_TABLE2 = [
    ("Logistic Regression", "binary", 1, 0.908, 0.893, 0.900),
    ("Logistic Regression", "count", 1, 0.899, 0.894, 0.897),
    ("Logistic Regression", "tfidf", 1, 0.881, 0.872, 0.877),
    ("SVM", "binary", 100, 0.908, 0.894, 0.901),
    ("SVM", "count", 20, 0.900, 0.895, 0.898),
    ("SVM", "tfidf", 1, 0.904, 0.896, 0.900),
    ("Naive Bayes", "binary", None, 0.839, 0.923, 0.881),
    ("Naive Bayes", "count", None, 0.836, 0.912, 0.874),
    ("Naive Bayes", "tfidf", None, 0.819, 0.868, 0.879),
    ("Random Forest", "binary", None, 0.859, 0.845, 0.852),
    ("Random Forest", "count", None, 0.860, 0.839, 0.849),
    ("Random Forest", "tfidf", None, 0.864, 0.786, 0.844),
    ("Boosting", "binary", None, 0.863, 0.798, 0.831),
    ("Boosting", "count", None, 0.869, 0.800, 0.834),
    ("Boosting", "tfidf", None, 0.865, 0.789, 0.825),
    ("DNN", "binary", 0.0001, 0.911, 0.901, 0.906),
    ("DNN", "count", 0.0001, 0.896, 0.900, 0.898),
    ("DNN", "tfidf", 0.0001, 0.881, 0.921, 0.901),
]
PAPER_BEST_ACCURACY = 0.906
