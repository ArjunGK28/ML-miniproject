# IMDb Sentiment Analysis: reproducing and improving a CS229 paper

UE24CS352A Machine Learning mini-project, problem statement 10.

We reproduce the results of _"Machine Learning based classification for Sentimental analysis
of IMDb reviews"_ (C.-L. Wu and S.-L. Shin, Stanford CS229, [docs/original_work.pdf](docs/original_work.pdf))
and then improve on them. The paper classifies 50,000 IMDb movie reviews as positive or
negative using three text representations (binary, word count and tf-idf over 3-grams) and
six models (Logistic Regression, SVM, Naive Bayes, Random Forest, Boosting, DNN).

## Results

All numbers are accuracy on the 25,000 test reviews. The CSV files in `results/` hold
the full metrics.

### Reproduction of the paper's Table 2

| Model               | Features   | Paper | Ours   | Difference |
| ------------------- | ---------- | ----- | ------ | ---------- |
| Logistic Regression | binary     | 0.900 | 0.9004 | +0.0004    |
| Logistic Regression | word count | 0.897 | 0.8987 | +0.0017    |
| Logistic Regression | tf-idf     | 0.877 | 0.8802 | +0.0032    |
| SVM                 | binary     | 0.901 | 0.8990 | -0.0020    |
| SVM                 | word count | 0.898 | 0.8996 | +0.0016    |
| SVM                 | tf-idf     | 0.900 | 0.9023 | +0.0023    |
| Naive Bayes         | binary     | 0.881 | 0.8795 | -0.0015    |
| Naive Bayes         | word count | 0.874 | 0.8731 | -0.0009    |
| Naive Bayes         | tf-idf     | 0.879 | 0.8775 | -0.0015    |
| Random Forest       | binary     | 0.852 | 0.8616 | +0.0096    |
| Random Forest       | word count | 0.849 | 0.8598 | +0.0108    |
| Random Forest       | tf-idf     | 0.844 | 0.8548 | +0.0108    |
| Boosting            | binary     | 0.831 | 0.8151 | -0.0159    |
| Boosting            | word count | 0.834 | 0.8169 | -0.0171    |
| Boosting            | tf-idf     | 0.825 | 0.8165 | -0.0085    |
| DNN                 | binary     | 0.906 | 0.8985 | -0.0075    |
| DNN                 | word count | 0.898 | 0.8953 | -0.0027    |
| DNN                 | tf-idf     | 0.901 | 0.8989 | -0.0021    |

The Logistic Regression, SVM and Naive Bayes rows are within 0.35 points of the paper, and
Random Forest is about 1 point above it. Random Forest uses the n-grams found in at least 5
reviews (210,685 features). Boosting and the DNN use those found in at least 50 (17,671
features) so that they finish on a laptop, and end up 0.9 to 1.7 and 0.2 to 0.8 points below
the paper. In an earlier run with the 210,685-feature vocabulary the binary DNN scored
0.9057 (about ten minutes), level with the paper's 0.906.

### Improvements

The paper's best accuracy is 0.906 (DNN, binary features). Our best single model, chosen on validation data, reaches **0.916** (NB-LR). An ensemble of
NB-LR and a tf-idf SVM reaches **0.918**, but its 0.22-point lead is within the noise of one
test set (the standard error is about 0.17 points).

| Model                                      | Features | Cleaning             | Accuracy   | vs paper's best |
| ------------------------------------------ | -------- | -------------------- | ---------- | --------------- |
| **Ensemble: NB-LR + SVM (tf-idf)**         | mixed    | minimal, rich tokens | **0.9182** | +0.0122         |
| MLP, one ReLU layer (128 units)            | binary   | minimal, rich tokens | 0.8931     | -0.0129         |
| NB-LR                                      | binary   | minimal, rich tokens | **0.9160** | +0.0100         |
| NB-SVM                                     | binary   | minimal, rich tokens | 0.9157     | +0.0097         |
| SVM, wider C grid (C = 16)                 | tf-idf   | minimal, rich tokens | 0.9089     | +0.0029         |
| SVM, paper's grid                          | binary   | minimal, rich tokens | 0.9033     | -0.0027         |
| Logistic Regression, wider C grid (C = 64) | tf-idf   | minimal, rich tokens | 0.9029     | -0.0031         |
| Logistic Regression, paper's grid          | binary   | minimal, rich tokens | 0.9024     | -0.0036         |

What made the difference:

- **NB-SVM / NB-LR** (Wang and Manning, 2012) scale each binary n-gram by its Naive Bayes
  log-count ratio and then fit a linear model. This adds about 1.3 points over the same
  linear models on plain binary features. The weight interpolation that paper recommends for
  NB-SVM (beta = 0.25) hurt here (0.896), so we do not use it.
- **Cleaning matters, and less is better.** Same models, same C grid, binary features:

  | Cleaning                                                      | Logistic Regression | SVM    | Naive Bayes |
  | ------------------------------------------------------------- | ------------------- | ------ | ----------- |
  | Paper: short stop-word list, lemmatised, no numbers           | 0.9004              | 0.8990 | 0.8795      |
  | Same, with NLTK's full stop-word list (198 words)             | 0.8869              | 0.8857 | 0.8628      |
  | Minimal: lowercased, HTML line breaks removed                 | 0.8986              | 0.8986 | 0.8782      |
  | Minimal, rich tokens: also keep "!", "?" and one-letter words | 0.9024              | 0.9033 | 0.8784      |

  Removing the full stop-word list, which deletes "not" and "no", costs 1.3 to 1.7 points.
  The paper's cleaning and minimal cleaning differ by less than 0.2 points.

- **A wider C grid for tf-idf.** The paper's grid stops at C = 1; on tf-idf features the
  best C is 16 for the SVM and 64 (the largest we tried) for Logistic Regression.

- **Ensemble.** It averages the scores of NB-LR and the tf-idf SVM, the best pair on
  validation data out of all 11 combinations of NB-LR, NB-SVM, the tf-idf SVM and an MLP.
  Its ROC-AUC is 0.9739, against 0.9723 for NB-LR alone.
- **A different network did not help.** Our MLP with one wide ReLU layer scores 0.8931,
  below the paper-style DNN (0.8985 on a vocabulary of similar size). The two use different
  cleaning, so this is not a controlled comparison.

Every variant's test accuracy is listed for transparency, but nothing was chosen by it:
the tokenisation, beta and C were all selected on the validation split.

### What we found in the paper

- **Its "precision" columns are recall.** The paper defines positive precision as
  TP / (TP + FP), but its numbers match our per-class _recall_ (Naive Bayes, binary:
  paper 0.839 / 0.923, our recall 0.837 / 0.922, our precision 0.915 / 0.850). It also
  states that accuracy "is the average of negative and positive precision", which is true
  of recall on a balanced test set and not of precision; 15 of its 18 rows satisfy it to
  within rounding.
- **Two rows of Table 2 are inconsistent.** Naive Bayes tf-idf reports accuracy 0.879
  with class scores 0.819 and 0.868; accuracy cannot be above both, whether they are
  precision or recall. Random Forest tf-idf reports 0.844 with 0.864 and 0.786, whose
  mean is 0.825.
- **The SVM rows use C = 100 and C = 20**, values outside the grid the paper says it
  searched.

## Setup

Python 3.10 or newer (developed on 3.13). About 4 GB of free RAM is needed for the
full-vocabulary experiments.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The dataset ([Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/),
84 MB) and the NLTK stop-word and WordNet data are downloaded automatically the first time
they are needed.

## Running

Run everything from the repository root.

| Step                                               | Command                                                                                                                                                   | Time         | Output                                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------- |
| 1. Download and summarise the data                 | `python -m src.data`                                                                                                                                      | 1 min        | `data/`                                                                                      |
| 2. Reproduce Logistic Regression, SVM, Naive Bayes | `python -m src.experiments.linear_models`                                                                                                                 | 10 min       | `results/reproduction_linear.csv`, `models/`                                                 |
| 3. Improvements on the linear models               | `python -m src.experiments.improve_features`                                                                                                              | 20 min       | `results/improve_features.csv`, `models/`                                                    |
| 4. Random vs movie-grouped validation (optional)   | `python -m src.experiments.split_check`                                                                                                                   | 5 min        | `results/validation_split_check.csv`                                                         |
| 5. Reproduce Random Forest, Boosting, DNN          | `python -m src.experiments.nonlinear_models --models "Random Forest"` then `python -m src.experiments.nonlinear_models --models Boosting DNN --min-df 50` | about 10 min | `results/reproduction_nonlinear.csv`                                                         |
| 6. Better MLP and ensemble                         | `python -m src.experiments.improve_ensemble`                                                                                                              | about 5 min  | `results/improve_ensemble.csv`, `results/improve_ensemble_auc.csv`, `results/roc_curves.png` |
| 7. Comparison figures                              | `python -m src.plots`                                                                                                                                     | seconds      | `results/reproduction_vs_paper.png`, `results/improvements.png`                              |
| 8. Walkthrough notebook                            | open `notebooks/walkthrough.ipynb` and Run All                                                                                                            |              |                                                                                              |
| 9. Web demo                                        | `streamlit run app/app.py`                                                                                                                                |              | opens in the browser                                                                         |

Times are from a laptop with a 12th-gen Core i5 and 16 GB of RAM. Add `--quick` to an
experiment script to run it on a 2,000-review subset in under a minute (nothing is saved).
On binary and count features the SVM reaches liblinear's iteration limit for the larger C
values, and scikit-learn prints a convergence warning. The C values selected for the
reproduction (0.05, 0.01 and 1) all converge.

The web demo needs the models written by steps 2 and 3. Type a review to see each model's
prediction and the n-grams that drove it; the "Results vs paper" view shows our numbers
next to the paper's. The first prediction takes about 15 seconds while the vocabularies
load.

## Project structure

```
src/config.py                        paths, seed, C grid, the paper's Table 2
src/data.py                          download and load the dataset
src/preprocess.py                    text cleaning (three modes), cached in data/processed/
src/features.py                      binary / count / tf-idf matrices from one n-gram vocabulary;
                                     two tokenisations ("words" and "rich")
src/evaluate.py                      metrics, results tables, comparison with the paper
src/predict.py                       save and load fitted models; predict and explain raw text
src/nbsvm.py                         NB-SVM / NB-LR classifier
src/experiments/runner.py            shared loop: tune on validation data, test once
src/experiments/linear_models.py     reproduction: Logistic Regression, SVM, Naive Bayes
src/experiments/improve_features.py  improvements: cleaning ablation, NB-SVM, wider C grid
src/experiments/split_check.py       random vs movie-grouped validation split
app/app.py                           Streamlit demo
results/                             results of every experiment as CSV
docs/                                the paper, the assignment guidelines, the write-up
src/experiments/nonlinear_models.py  reproduction: Random Forest, Boosting, DNN (reduced vocabulary)
src/experiments/improve_ensemble.py  improvements: wider MLP and ensemble
src/plots.py                         comparison figures made from the results CSVs
notebooks/walkthrough.ipynb          walkthrough: data, results vs paper, live predictions
```

`data/` and `models/` are not in the repository; the scripts recreate them.

## Method

Every experiment follows the same procedure (`src/experiments/runner.py`):

1. Clean the text and build (1,3)-gram features.
2. Choose hyperparameters on a validation split: 20% of the training set, **grouped by
   movie**. The dataset's train and test sets share only one movie, so a random split,
   with reviews of the same movie on both sides, rewards memorising movie-specific words.
   `split_check.py` measures this: on a random split Naive Bayes scores 0.895, level with
   Logistic Regression (0.898) and above its own test accuracy (0.880). Grouped by movie
   it scores 0.866 against 0.882, close to the 2-point gap between them on the test set.
3. Refit on the full training set with the chosen hyperparameters and evaluate once on
   the 25,000 test reviews. No hyperparameter and no improvement was chosen on the test
   set.

The paper leaves some details open, and its code is no longer online. Our choices:

- **"3 grams"** means unigrams, bigrams and trigrams together (`ngram_range=(1, 3)`), as in
  the tutorial the paper cites for its method.
- **Stop words.** The paper does not name its list. We first used NLTK's full list and
  landed 1.3 points below the paper's Logistic Regression. The paper's examples ("a",
  "the", "of") come from the five-word list in that tutorial; with that list the
  reproduction matches. This is the one choice we made by comparing test results with
  the paper, and both lists are in the ablation table above.
- **Regularisation.** C is chosen from the paper's grid [0.01, 0.05, 0.25, 0.5, 1] on the
  validation split.
- **Vocabulary for Random Forest, Boosting and the DNN.** Millions of n-grams are too many
  for these models, so they keep only n-grams found in at least `min_df` reviews: 5 for
  Random Forest (210,685 features) and 50 for Boosting and the DNN (17,671 features). The
  paper does not say how it handled this, and gives no hyperparameter grid for these three
  models, so they are not tuned.

## Adding an experiment

`src/experiments/linear_models.py` is the template. `prepare()` returns the feature
matrices, `run()` trains one (model, vectorization) pair and returns a results row with
the shared columns, and `report()` writes `results/<name>.csv`. Models saved with
`save_model()` appear in the web demo automatically.
