"""Improvement: fine-tune pretrained transformers (DistilBERT and RoBERTa-base).

Every model in the paper, and every improvement so far, reads a review as a bag of
n-grams. A transformer reads the words in order and starts from a language model
pretrained on a large amount of English text, so it can handle negation ("not bad at
all"), contrast ("the first half is dull, but ...") and words it never saw labelled.

Protocol, the same as the other experiments:
- The 25,000 training reviews are split 80/20 by movie (src/experiments/runner.py).
- The model is fine-tuned on the 80% for up to EPOCHS epochs. After each epoch it is
  scored on the 20%; the epoch with the best validation accuracy is kept. That epoch
  count is the only setting chosen here (it is written to the "C" column of the results);
  the learning rates are the usual ones for these models.
- The kept model is scored once on the 25,000 test reviews.
Unlike the other experiments the model is not refitted on all 25,000 training reviews
afterwards (that would double the training time), so the transformers see 20% less
training data than the n-gram models.

Long reviews are cut to 512 tokens by keeping the first 128 and the last 382, which
Sun et al. (2019, "How to Fine-Tune BERT for Text Classification?") found works best on
IMDb: reviews tend to state the verdict at the start or at the end.

Needs an NVIDIA GPU (developed on an RTX 3050 with 6 GB): about 1 hour for DistilBERT and
3 hours for RoBERTa-base. Writes results/transformer_models.csv, the fine-tuned models to
models/<name>/ and each model's validation and test scores to models/scores/<name>.npz
(used by src/experiments/final_comparison.py).

    python -m src.experiments.transformer_models                      # both models
    python -m src.experiments.transformer_models --models DistilBERT  # one model
    python -m src.experiments.transformer_models --quick              # smoke test, saves nothing
"""
import argparse
import copy
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.config import MODELS_DIR, QUICK_N, RESULTS_DIR, SEED, VAL_SIZE
from src.data import load_raw
from src.evaluate import compute_metrics
from src.experiments.runner import report

# checkpoint on the Hugging Face hub, peak learning rate, reviews per step on the GPU,
# and steps to accumulate before each update (so both models update every 16 reviews).
MODELS = {
    "DistilBERT": dict(checkpoint="distilbert-base-uncased", lr=5e-5, batch=16, accumulate=1),
    "RoBERTa-base": dict(checkpoint="roberta-base", lr=2e-5, batch=8, accumulate=2),
}
EPOCHS = 2
MAX_TOKENS = 512
HEAD_TOKENS = 128            # kept from the start of a long review; the rest from its end
EVAL_BATCH = 64
SCORES_DIR = MODELS_DIR / "scores"

_BREAKS = re.compile(r"<br\s*/?>")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_split(quick):
    """Raw reviews (only the HTML line breaks removed) with the runner's grouped split."""
    train, test = load_raw("train"), load_raw("test")
    if quick:
        train = train.groupby("label").sample(QUICK_N // 2, random_state=SEED)
        test = test.groupby("label").sample(QUICK_N // 2, random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(train.text, train.label, groups=train.movie))
    parts = {"fit": train.iloc[fit_idx], "val": train.iloc[val_idx], "test": test}
    return {name: ([_BREAKS.sub(" ", t) for t in part.text], part.label.to_numpy())
            for name, part in parts.items()}


def encode(tokenizer, texts, max_tokens):
    """Token ids per review, cut to max_tokens by keeping its head and its tail.

    A single review is wrapped as [CLS] ... [SEP] for BERT models and <s> ... </s> for
    RoBERTa; both tokenizers expose these as cls_token_id and sep_token_id.
    """
    room = max_tokens - 2
    head = min(HEAD_TOKENS, room)
    encoded = []
    for ids in tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]:
        if len(ids) > room:
            ids = ids[:head] + ids[len(ids) - (room - head):]
        encoded.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    return encoded


def batches(encoded, order, size, pad_id, device):
    """Pad each batch only to its own longest review."""
    for start in range(0, len(order), size):
        rows = [encoded[i] for i in order[start:start + size]]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = torch.tensor(row)
            mask[r, :len(row)] = 1
        yield order[start:start + size], ids.to(device), mask.to(device)


@torch.no_grad()
def positive_scores(model, encoded, pad_id, device):
    """Probability of the positive class for every review."""
    model.eval()
    order = np.argsort([len(e) for e in encoded])   # similar lengths together: less padding
    scores = np.empty(len(encoded), dtype=np.float32)
    for idx, ids, mask in batches(encoded, order, EVAL_BATCH, pad_id, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        scores[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return scores


def finetune(name, settings, data, epochs, max_tokens, quick):
    device = "cuda"
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(settings["checkpoint"])
    model = AutoModelForSequenceClassification.from_pretrained(settings["checkpoint"],
                                                               num_labels=2).to(device)
    encoded = {part: encode(tokenizer, texts, max_tokens) for part, (texts, _) in data.items()}
    y = {part: labels for part, (_, labels) in data.items()}
    pad_id = tokenizer.pad_token_id

    batch, accumulate = settings["batch"], settings["accumulate"]
    steps = epochs * -(-len(encoded["fit"]) // (batch * accumulate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    rng = np.random.default_rng(SEED)

    best_epoch, best_val, best_state, val_scores = 0, -1.0, None, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(encoded["fit"]))
        for step, (idx, ids, mask) in enumerate(
                batches(encoded["fit"], order, batch, pad_id, device), 1):
            labels = torch.tensor(y["fit"][idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / accumulate).backward()
            if step % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
            if step % 200 == 0:
                print(f"    epoch {epoch} step {step}/{-(-len(order) // batch)} "
                      f"loss {loss.item():.3f} ({time.perf_counter() - start:.0f}s)", flush=True)
        scores = positive_scores(model, encoded["val"], pad_id, device)
        val_acc = float(((scores > 0.5) == y["val"]).mean())
        print(f"  {name} epoch {epoch}: validation accuracy {val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_epoch, best_val, val_scores = epoch, val_acc, scores
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    seconds = time.perf_counter() - start

    model.load_state_dict(best_state)
    test_scores = positive_scores(model, encoded["test"], pad_id, device)
    row = {"model": f"{name} (fine-tuned)", "vectorization": "subword tokens",
           "cleaning": "raw text", "C": best_epoch, "n_features": None, "val_accuracy": best_val,
           **compute_metrics(y["test"], (test_scores > 0.5).astype(int)),
           "train_seconds": round(seconds, 1)}
    print(f"  {name:<14} epochs={best_epoch} val={best_val:.4f} test={row['accuracy']:.4f} "
          f"({seconds / 60:.0f} min)", flush=True)

    if not quick:
        out = MODELS_DIR / slug(name)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(SCORES_DIR / f"{slug(name)}.npz", val=val_scores, test=test_scores,
                 y_val=y["val"], y_test=y["test"])
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return row


def main(names, quick=False):
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Install the CUDA build of PyTorch (see README).")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    data = load_split(quick)
    epochs, max_tokens = (1, 128) if quick else (EPOCHS, MAX_TOKENS)
    rows = [finetune(name, MODELS[name], data, epochs, max_tokens, quick) for name in names]
    if not quick:
        # Keep rows of models not run this time, so the CSV can be built one model at a time.
        path = RESULTS_DIR / "transformer_models.csv"
        if path.exists():
            previous = pd.read_csv(path)
            rows = [r for r in previous.to_dict("records")
                    if r["model"] not in {row["model"] for row in rows}] + rows
    report(rows, "transformer_models", quick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--quick", action="store_true", help="run on a small subset, save nothing")
    args = parser.parse_args()
    main(args.models, args.quick)
