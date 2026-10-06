"""Turn cleaned text into the paper's three feature matrices: binary, word count and tf-idf."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.config import NGRAM_RANGE, REDUCED_MIN_DF, VECTORIZATIONS


class Featurizer:
    """One n-gram vocabulary, three representations.

    The n-grams are counted once; the binary and tf-idf matrices are derived from the
    counts, so all three representations share exactly the same columns.

    vocab="full" keeps every n-gram; vocab="reduced" keeps those seen in at least
    REDUCED_MIN_DF reviews (for models that cannot handle millions of features).
    """

    def __init__(self, vocab="full", ngram_range=NGRAM_RANGE, min_df=None, sublinear_tf=False):
        if min_df is None:
            min_df = 1 if vocab == "full" else REDUCED_MIN_DF
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.tfidf = TfidfTransformer(sublinear_tf=sublinear_tf)

    def fit_transform(self, texts):
        counts = self.vectorizer.fit_transform(texts)
        self.tfidf.fit(counts)
        return self._represent(counts)

    def transform(self, texts):
        return self._represent(self.vectorizer.transform(texts))

    def _represent(self, counts):
        binary = counts.copy()
        binary.data[:] = 1
        matrices = {"binary": binary, "count": counts, "tfidf": self.tfidf.transform(counts)}
        return {kind: matrices[kind] for kind in VECTORIZATIONS}

    @property
    def feature_names(self):
        return self.vectorizer.get_feature_names_out()

    @property
    def n_features(self):
        return len(self.vectorizer.vocabulary_)
