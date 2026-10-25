"""Saved models for the demos: raw review text in, sentiment out."""
import re

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR
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


def _transformer_bundles():
    """The fine-tuned transformers saved by src/experiments/transformer_models.py."""
    path = RESULTS_DIR / "transformer_models.csv"
    if not path.exists():
        return []
    bundles = []
    for row in pd.read_csv(path).itertuples():
        folder = MODELS_DIR / _slug(row.model.replace(" (fine-tuned)", ""))
        if (folder / "config.json").exists():
            bundles.append({"name": row.model, "kind": "subword tokens", "cleaning": "raw text",
                            "group": "Transformer", "test_accuracy": row.accuracy,
                            "folder": folder})
    return bundles


class _Transformer:
    """A fine-tuned transformer, loaded on first use (importing torch is slow)."""

    def __init__(self, folder):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.experiments.transformer_models import MAX_TOKENS, encode
        self.torch, self.encode, self.max_tokens = torch, encode, MAX_TOKENS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model = AutoModelForSequenceClassification.from_pretrained(folder).to(self.device)
        self.model.eval()

    def positive_probability(self, texts):
        """P(positive) for each text, using the head-and-tail cut from training."""
        torch = self.torch
        rows = self.encode(self.tokenizer, texts, self.max_tokens)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = torch.tensor(row)
            mask[r, :len(row)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device),
                                attention_mask=mask.to(self.device)).logits
        return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()


class Predictor:
    """Loads every model in models/ and predicts the sentiment of raw reviews."""

    def __init__(self):
        bundles = [joblib.load(path) for path in MODELS_DIR.glob("model_*.joblib")]
        # Reproduced models first, then improvements, transformers last.
        order = {"Improvement": 1, "Transformer": 2}
        bundles.sort(key=lambda b: (order.get(b.get("group"), 0), b["name"]))
        bundles += _transformer_bundles()
        self.bundles = {bundle["name"]: bundle for bundle in bundles}
        self._featurizers = {}
        self._transformers = {}

    @property
    def names(self):
        return list(self.bundles)

    def is_transformer(self, name):
        return "folder" in self.bundles[name]

    def _transformer(self, name):
        if name not in self._transformers:
            self._transformers[name] = _Transformer(self.bundles[name]["folder"])
        return self._transformers[name]

    def _features(self, bundle, text):
        key = bundle["featurizer"]
        if key not in self._featurizers:
            self._featurizers[key] = joblib.load(MODELS_DIR / f"featurizer_{key}.joblib")
        cleaned = clean_text(text, bundle["cleaning"])
        X = self._featurizers[key].transform([cleaned])[bundle["kind"]]
        return cleaned, self._featurizers[key].vectorizer, X

    def can_explain(self, name):
        return self.is_transformer(name) or _weights(self.bundles[name]["model"]) is not None

    def predict(self, name, text):
        """Return {"label": 0/1, "probability": P(positive) or None, "score": margin or None}."""
        if self.is_transformer(name):
            probability = float(self._transformer(name).positive_probability([text])[0])
            return {"label": int(probability > 0.5), "probability": probability, "score": None}
        bundle = self.bundles[name]
        model = bundle["model"]
        _, _, X = self._features(bundle, text)
        result = {"label": int(model.predict(X)[0]), "probability": None, "score": None}
        if hasattr(model, "predict_proba"):
            result["probability"] = float(model.predict_proba(X)[0, 1])
        elif hasattr(model, "decision_function"):
            result["score"] = float(model.decision_function(X)[0])
        return result

    def explain(self, name, text, top=8):
        """The n-grams of `text` that pushed the model hardest, as (ngram, contribution) pairs.

        Positive contributions argue for a positive review. Returns [] for models
        without per-feature weights (trees, neural networks). Transformers have no
        per-n-gram weights either, so for them see explain_transformer.
        """
        if self.is_transformer(name):
            return self.explain_transformer(name, text, top)
        bundle = self.bundles[name]
        weights = _weights(bundle["model"])
        if weights is None:
            return []
        cleaned, vectorizer, X = self._features(bundle, text)
        contributions = X.data * weights[X.indices]
        order = np.argsort(-np.abs(contributions))[:top]
        # Name only the n-grams of this review instead of inverting the whole vocabulary.
        vocabulary = vectorizer.vocabulary_
        names = {vocabulary[t]: t for t in vectorizer.build_analyzer()(cleaned) if t in vocabulary}
        return [(names[X.indices[i]], float(contributions[i])) for i in order]

    def explain_transformer(self, name, text, top=8):
        """Leave-one-word-out: how far the log-odds of positive fall when a word is deleted.

        Every copy of the word is deleted at once. A word whose removal lowers the log-odds
        argued for a positive review, so it gets a positive contribution. Log-odds rather
        than P(positive), because a confident model sits near 0 or 1, where probabilities
        barely move. All the shortened reviews go through the model in one batch.
        """
        tokens = text.split()
        words = sorted({w for w in (re.sub(r"\W", "", t.lower()) for t in tokens) if w})
        if not words:
            return []
        variants = [" ".join(t for t in tokens if re.sub(r"\W", "", t.lower()) != w)
                    for w in words]
        p = self._transformer(name).positive_probability([text] + variants).astype(np.float64)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        log_odds = np.log(p / (1 - p))
        contributions = log_odds[0] - log_odds[1:]
        order = np.argsort(-np.abs(contributions))[:top]
        return [(words[i], float(contributions[i])) for i in order]
