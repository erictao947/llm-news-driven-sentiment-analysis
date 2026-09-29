"""Baseline 2: logistic regression on all-MiniLM-L6-v2 headline embeddings.

Trained on in-sample events only, target = sign of the 30-minute SPY-excess return at the primary entry delay.
Regularization is chosen by cross-validation with folds grouped by trading day (same-day headlines are correlated).
score = 2 * P(up) - 1; confidence = |score|.

Usage: python -m src.score_embed
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGET = f"xret_{C.PRIMARY_EXIT}"


def embed(texts):
    m = SentenceTransformer(MODEL)
    return m.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)


def main():
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    ret = pd.read_parquet(C.INTERIM / "returns.parquet")
    ret = ret[ret["delay_s"] == C.ENTRY_DELAY_S][["event_id", "entry_date", "sample", TARGET]]

    heads = ev["headline"].drop_duplicates().tolist()
    E = embed(heads)
    row = {h: i for i, h in enumerate(heads)}
    X_all = E[ev["headline"].map(row).to_numpy()]

    train = ev.reset_index(drop=True).merge(ret, on="event_id", how="left")
    mask = (train["sample"] == "in").to_numpy() & train[TARGET].notna().to_numpy() & (train[TARGET] != 0).to_numpy()
    X, y = X_all[mask], (train.loc[mask, TARGET] > 0).astype(int).to_numpy()
    groups = train.loc[mask, "entry_date"].astype(str).to_numpy()

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    grid = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 0, 9)},
                        cv=GroupKFold(n_splits=5), scoring="roc_auc")
    grid.fit(X, y, groups=groups)
    print(f"embed logit: {mask.sum():,} in-sample training events, up-rate {y.mean():.3f}, "
          f"best C={grid.best_params_['logisticregression__C']:.2e}, CV AUC={grid.best_score_:.3f}")

    p_up = grid.best_estimator_.predict_proba(X_all)[:, 1]
    out = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "score": 2 * p_up - 1})
    out["confidence"] = out["score"].abs()
    out.to_parquet(C.SCORES / "embed.parquet", index=False)
    pd.DataFrame([{"n_train": int(mask.sum()), "up_rate": y.mean(), "best_C": grid.best_params_["logisticregression__C"],
                   "cv_auc": grid.best_score_}]).to_csv(C.TABLES / "embed_training.csv", index=False)


if __name__ == "__main__":
    main()
