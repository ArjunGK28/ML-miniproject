"""NB-SVM / NB-LR: a linear classifier trained on Naive Bayes log-count ratios.

S. Wang and C. Manning, "Baselines and Bigrams: Simple, Good Sentiment and Topic
Classification", ACL 2012.

Each binary n-gram feature is scaled by r = log(p / q), where p and q are the smoothed,
normalised frequencies of that n-gram in positive and negative reviews. A linear SVM or
logistic regression is then trained on the scaled features. Its weights can be
interpolated with their mean magnitude (beta < 1), as the paper above recommends for
NB-SVM; beta = 1 means no interpolation, which worked better on our validation data.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class NBSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, alpha=1.0, beta=1.0, base="svm", random_state=None):
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        p = self.alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
        ratio = np.log((p / p.sum()) / (q / q.sum()))

        if self.base == "svm":
            linear = LinearSVC(C=self.C, tol=1e-4, random_state=self.random_state)
        else:
            linear = LogisticRegression(C=self.C, tol=1e-4, solver="liblinear",
                                        random_state=self.random_state)
        linear.fit(X.multiply(ratio).tocsr(), y)

        w = linear.coef_.ravel()
        w = (1 - self.beta) * np.abs(w).mean() + self.beta * w
        # Fold the ratio into the weights so the model scores plain binary features.
        self.coef_ = (ratio * w).reshape(1, -1)
        self.intercept_ = linear.intercept_
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.coef_.ravel()).ravel() + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
