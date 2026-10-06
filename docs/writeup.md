# IMDb Sentiment Analysis: Reproducing and Improving a CS229 Report

UE24CS352A Machine Learning mini-project, problem statement 10

## 1. Problem statement

We were asked to reproduce and improve on *"Machine Learning based classification for
Sentimental analysis of IMDb reviews"* by C.-L. Wu and S.-L. Shin (Stanford CS229). The
report classifies IMDb movie reviews as positive or negative. It cleans the text, turns
each review into a vector of 3-gram features in three ways (binary, word count, tf-idf)
and compares six classifiers: Logistic Regression, SVM, Naive Bayes, Random Forest,
Boosting and a five-layer neural network. Its best accuracy is 90.6% (neural network on
binary features).

Our goals were to (1) reproduce the report's Table 2, (2) check its methodology, and
(3) beat 90.6% on the same test set.

## 2. Dataset

We use the Large Movie Review Dataset v1.0 (Maas et al., 2011), the same data as the
report: 50,000 labelled IMDb reviews, split by its authors into 25,000 for training and
25,000 for testing, each half 50% positive. A review is negative if its rating is 4 or
lower and positive if it is 7 or higher; ratings of 5 and 6 are excluded. Ratings are
concentrated at the extremes: 1 and 10 make up about 40% of all reviews.

Reviews average 230 words (median 173, longest 2,470). No movie has more than 30
reviews. The training set covers 3,456 movies and the test set 3,581, with only one movie
in common, so a model is always tested on movies it has not seen.

## 3. Approach

**Reproduction.** We follow the report's pipeline. Text is lowercased; HTML line breaks,
punctuation, numbers and stop words are removed; each word is reduced to its root with
the WordNet lemmatiser. All unigrams, bigrams and trigrams of the training reviews form
one vocabulary, from which the binary, word-count and tf-idf matrices are derived, so the
three representations differ only in the values they hold. Each model uses the settings
listed in the report.

The report leaves three details open, and its code is no longer online:

- *Stop words.* The list is not named. The report's examples ("a", "the", "of") come from
  the five-word list of the tutorial it cites for its method, so we use that list, and
  test NLTK's full 198-word list separately.
- *"3 grams".* We read this as 1- to 3-grams together, as in the same tutorial.
- *Choosing C.* The report says only "choose the best performance". We choose C on a
  validation split and never on the test set.

**Evaluation protocol.** Hyperparameters are chosen on 20% of the training set, held out
*by movie*: because the test set contains only unseen movies, a random split, with reviews
of the same movie on both sides, rewards memorising movie-specific words. On a random
split Naive Bayes scores 89.5%, above its own test accuracy of 88.0%; grouped by movie it
scores 86.6%. The chosen model is refitted on all 25,000 training reviews and scored once
on the 25,000 test reviews.

**Improvements.** We changed one thing at a time and kept what helped on validation data:

1. *Cleaning ablation*: the report's cleaning against NLTK's full stop-word list, against
   lighter cleaning that keeps stop words, negations ("not", "no") and numbers ("10/10"),
   and against lighter cleaning that also keeps "!", "?" and one-letter words as tokens.
2. *NB-SVM / NB-LR* (Wang and Manning, 2012): a linear classifier trained on binary
   n-grams scaled by their Naive Bayes log-count ratios, which combines two of the
   report's own models.
3. *A wider C grid*: the report's grid stops at C = 1, which over-regularises tf-idf
   features whose values are much smaller than counts.
