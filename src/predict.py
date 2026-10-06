"""Saved models for the demos: raw review text in, sentiment out."""
import re

import joblib
import numpy as np

from src.config import MODELS_DIR
from src.preprocess import clean_text


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_featurizer(key, featurizer):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(featurizer, MODELS_DIR / f"featurizer_{key}.joblib")


def save_model(name, model, featurizer_key, kind, cleaning, **info):
    """Store a fitted model with everything needed to rebuild its input from raw text."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"name": name, "model": model, "featurizer": featurizer_key,
              "kind": kind, "cleaning": cleaning, **info}
    joblib.dump(bundle, MODELS_DIR / f"model_{_slug(name)}.joblib")


def _weights(model):
    """Per-feature evidence for the positive class, for models that have one."""
    if hasattr(model, "coef_"):
        return np.ravel(model.coef_)
    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    return None


class Predictor:
    """Loads every model in models/ and predicts the sentiment of raw reviews."""

    def __init__(self):
        self.bundles = {}
        for path in sorted(MODELS_DIR.glob("model_*.joblib")):
            bundle = joblib.load(path)
            self.bundles[bundle["name"]] = bundle
        self._featurizers = {}

    @property
    def names(self):
        return list(self.bundles)

    def _features(self, bundle, text):
        key = bundle["featurizer"]
        if key not in self._featurizers:
            self._featurizers[key] = joblib.load(MODELS_DIR / f"featurizer_{key}.joblib")
        cleaned = clean_text(text, bundle["cleaning"])
        return self._featurizers[key], self._featurizers[key].transform([cleaned])[bundle["kind"]]

    def predict(self, name, text):
        """Return {"label": 0/1, "probability": P(positive) or None, "score": margin or None}."""
        bundle = self.bundles[name]
        model = bundle["model"]
        _, X = self._features(bundle, text)
        result = {"label": int(model.predict(X)[0]), "probability": None, "score": None}
        if hasattr(model, "predict_proba"):
            result["probability"] = float(model.predict_proba(X)[0, 1])
        elif hasattr(model, "decision_function"):
            result["score"] = float(model.decision_function(X)[0])
        return result

    def explain(self, name, text, top=8):
        """The n-grams of `text` that pushed the model hardest, as (ngram, contribution) pairs.

        Positive contributions argue for a positive review. Returns [] for models
        without per-feature weights (trees, neural networks).
        """
        bundle = self.bundles[name]
        weights = _weights(bundle["model"])
        if weights is None:
            return []
        featurizer, X = self._features(bundle, text)
        contributions = X.data * weights[X.indices]
        order = np.argsort(-np.abs(contributions))[:top]
        names = featurizer.feature_names
        return [(names[X.indices[i]], float(contributions[i])) for i in order]
