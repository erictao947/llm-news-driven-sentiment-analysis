"""Baseline 1: FinBERT (ProsusAI/finbert) on the same headlines.

score = P(positive) - P(negative) in [-1, 1]; confidence = 1 - P(neutral).
FinBERT sees only the headline text, so each unique headline is scored once and mapped to every ticker it tags.

Usage: python -m src.score_finbert
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C

MODEL = "ProsusAI/finbert"


def device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def finbert_probs(texts, batch=64):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device()).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device())
            out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            if (i // batch) % 100 == 0:
                print(f"  finbert {i:,}/{len(texts):,}")
    return pd.DataFrame(np.vstack(out), columns=labels)


def main():
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    heads = ev["headline"].drop_duplicates().tolist()
    p = finbert_probs(heads)
    p["headline"] = heads
    p["score"] = p["positive"] - p["negative"]
    p["confidence"] = 1 - p["neutral"]
    out = ev[["event_id", "headline"]].merge(p, on="headline")[["event_id", "score", "confidence"]]
    out.to_parquet(C.SCORES / "finbert.parquet", index=False)
    print(f"FinBERT scored {len(heads):,} unique headlines -> {len(out):,} events")


if __name__ == "__main__":
    main()
