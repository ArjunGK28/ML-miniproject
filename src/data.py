"""Download the Stanford Large Movie Review Dataset (aclImdb v1) and load it as DataFrames.

Run `python -m src.data` once to fetch the data and print a summary.
"""
import re
import tarfile
import urllib.request

import pandas as pd

from src.config import DATA_DIR, DATASET_URL

ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
# Labelled reviews are stored as aclImdb/<split>/<pos|neg>/<id>_<rating>.txt
_MEMBER = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")
# Line i of urls_<pos|neg>.txt is the IMDb page of the movie that review i is about.
_URLS = re.compile(r"aclImdb/(train|test)/urls_(pos|neg)\.txt$")
_MOVIE_ID = re.compile(r"tt\d+")


def download():
    if ARCHIVE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} (about 84 MB) ...")
    partial = ARCHIVE.with_suffix(".part")
    urllib.request.urlretrieve(DATASET_URL, partial)
    partial.rename(ARCHIVE)


def _extract():
    """Read the reviews straight out of the archive into one CSV per split."""
    rows = {"train": [], "test": []}
    movies = {}
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            urls = _URLS.match(member.name)
            if urls:
                lines = tar.extractfile(member).read().decode("utf-8").splitlines()
                movies[urls.groups()] = [_MOVIE_ID.search(line).group() for line in lines]
                continue
            match = _MEMBER.match(member.name)
            if not match:
                continue
            split, sentiment, review_id, rating = match.groups()
            text = tar.extractfile(member).read().decode("utf-8")
            rows[split].append((int(review_id), sentiment, int(rating), text))
    for split, items in rows.items():
        df = pd.DataFrame(items, columns=["id", "sentiment", "rating", "text"])
        df["movie"] = [movies[split, s][i] for s, i in zip(df.sentiment, df.id)]
        df["label"] = (df.sentiment == "pos").astype(int)
        df = df.sort_values(["label", "id"]).reset_index(drop=True)
        df[["id", "movie", "rating", "label", "text"]].to_csv(DATA_DIR / f"imdb_{split}.csv",
                                                              index=False)


def load_raw(split):
    """Return the raw reviews of `split` ("train" or "test"): id, movie, rating, label, text."""
    path = DATA_DIR / f"imdb_{split}.csv"
    if not path.exists():
        download()
        _extract()
    return pd.read_csv(path)


if __name__ == "__main__":
    for name in ("train", "test"):
        frame = load_raw(name)
        print(f"{name}: {len(frame)} reviews of {frame.movie.nunique()} movies, "
              f"{frame.label.mean():.0%} positive, ratings {sorted(frame.rating.unique().tolist())}")
    overlap = set(load_raw("train").movie) & set(load_raw("test").movie)
    print(f"movies that appear in both train and test: {len(overlap)}")
