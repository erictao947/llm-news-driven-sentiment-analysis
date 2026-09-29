"""Signal evaluation: information coefficient by horizon, block-bootstrap CIs, hit rates, breakdowns.

IC = Spearman rank correlation between score and forward SPY-excess return, pooled over out-of-sample events.
Confidence intervals resample whole trading days with replacement (headlines on the same day share market
shocks, so an event-level bootstrap would understate the error). Every replicate evaluates all models on the same
resampled days, which gives paired CIs for model differences.

All models are evaluated on the same event set: OOS events that every model scored and that have a valid entry.

Usage: python -m src.evaluate
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C


# ---------------------------------------------------------------- data
def load_panel(delay=C.ENTRY_DELAY_S, sample="oos"):
    """Wide panel: one row per event, columns score_<model>, conf_<model>, category, ret_*, xret_*."""
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    ret = pd.read_parquet(C.INTERIM / "returns.parquet")
    df = ev.merge(ret[ret["delay_s"] == delay], on="event_id")
    for m in ("llm", "finbert", "embed"):
        s = pd.read_parquet(C.SCORES / f"{m}.parquet").set_index("event_id")
        df[f"score_{m}"] = df["event_id"].map(s["score"])
        df[f"conf_{m}"] = df["event_id"].map(s["confidence"])
        if m == "llm":
            df["category"] = df["event_id"].map(s["category"])
    rng = np.random.default_rng(C.SEED)
    df["score_random"] = rng.choice([-1.0, 1.0], len(df))  # random-sign control
    df["conf_random"] = 1.0
    df = df.dropna(subset=[f"score_{m}" for m in C.MODELS])
    if sample != "all":
        df = df[df["sample"] == sample]
    return df.sort_values("entry_ts").reset_index(drop=True)


# ---------------------------------------------------------------- statistics
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def hit_rate(score, ret):
    s, r = np.sign(np.asarray(score, float)), np.sign(np.asarray(ret, float))
    ok = (s != 0) & (r != 0) & ~np.isnan(s) & ~np.isnan(r)
    return float((s[ok] == r[ok]).mean()) if ok.any() else np.nan


def day_bootstrap(df, stat_fns, reps=C.BOOTSTRAP_REPS, seed=C.SEED, day_col="entry_date"):
    """Resample trading days with replacement. stat_fns: {name: f(frame) -> float}. Returns {name: array(reps)}."""
    rng = np.random.default_rng(seed)
    codes, days = pd.factorize(df[day_col])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(days) + 1))
    blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(days))]
    out = {k: np.empty(reps) for k in stat_fns}
    for b in range(reps):
        pick = rng.integers(0, len(days), len(days))
        idx = np.concatenate([blocks[i] for i in pick])
        sub = df.iloc[idx]
        for k, f in stat_fns.items():
            out[k][b] = f(sub)
    return out


def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


# ---------------------------------------------------------------- tables
def ic_by_horizon(df, prefix="xret", reps=C.BOOTSTRAP_REPS):
    rows, diffs = [], []
    for h in C.HORIZONS:
        col = f"{prefix}_{h}"
        d = df.dropna(subset=[col])
        fns = {m: (lambda f, m=m, col=col: spearman(f[f"score_{m}"], f[col])) for m in C.MODELS}
        boot = day_bootstrap(d, fns, reps=reps)
        point = {m: spearman(d[f"score_{m}"], d[col]) for m in C.MODELS}
        for m in C.MODELS:
            lo, hi = ci(boot[m])
            rows.append({"model": m, "horizon": h, "n": len(d), "n_days": d["entry_date"].nunique(),
                         "ic": point[m], "ci_lo": lo, "ci_hi": hi,
                         "t_boot": point[m] / np.nanstd(boot[m]) if np.nanstd(boot[m]) > 0 else np.nan,
                         "hit_rate": hit_rate(d[f"score_{m}"], d[col])})
        for other in ("finbert", "embed", "random"):
            diff = boot["llm"] - boot[other]
            lo, hi = ci(diff)
            diffs.append({"horizon": h, "vs": other, "ic_llm": point["llm"], "ic_other": point[other],
                          "diff": point["llm"] - point[other], "ci_lo": lo, "ci_hi": hi,
                          "p_boot_le0": float((diff <= 0).mean()),
                          "ratio": point["llm"] / point[other] if point[other] else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def ic_by_group(df, group_col, horizon=C.PRIMARY_EXIT, reps=500, min_n=100):
    col = f"xret_{horizon}"
    rows = []
    for g, d in df.dropna(subset=[col]).groupby(group_col, observed=True):
        if len(d) < min_n:
            continue
        fns = {m: (lambda f, m=m: spearman(f[f"score_{m}"], f[col])) for m in C.MODELS}
        boot = day_bootstrap(d, fns, reps=reps)
        for m in C.MODELS:
            lo, hi = ci(boot[m])
            rows.append({group_col: g, "model": m, "n": len(d), "ic": spearman(d[f"score_{m}"], d[col]),
                         "ci_lo": lo, "ci_hi": hi, "hit_rate": hit_rate(d[f"score_{m}"], d[col]),
                         "mean_abs_ret_bps": d[col].abs().mean() * 1e4})
    return pd.DataFrame(rows)


def main():
    df = load_panel()
    print(f"OOS panel: {len(df):,} events over {df['entry_date'].nunique()} trading days")
    ic, diff = ic_by_horizon(df)
    ic.to_csv(C.TABLES / "ic_by_horizon.csv", index=False)
    diff.to_csv(C.TABLES / "ic_llm_vs_baselines.csv", index=False)
    raw, _ = ic_by_horizon(df, prefix="ret", reps=500)
    raw.to_csv(C.TABLES / "ic_by_horizon_raw_returns.csv", index=False)

    ic_by_group(df, "category").to_csv(C.TABLES / "ic_by_category.csv", index=False)
    ic_by_group(df.assign(session=np.where(df["headline_in_rth"], "regular_hours", "overnight")),
                "session").to_csv(C.TABLES / "ic_by_session.csv", index=False)
    df["llm_conf_bucket"] = pd.cut(df["conf_llm"], [0, 0.4, 0.6, 0.8, 1.0], include_lowest=True).astype(str)
    ic_by_group(df, "llm_conf_bucket").to_csv(C.TABLES / "ic_by_llm_confidence.csv", index=False)

    ins = load_panel(sample="in")
    ins_ic, _ = ic_by_horizon(ins, reps=200)
    ins_ic.to_csv(C.TABLES / "ic_by_horizon_insample.csv", index=False)

    counts = {"oos_events": len(df), "oos_days": int(df["entry_date"].nunique()),
              "oos_events_rth": int(df["headline_in_rth"].sum()),
              "insample_events": len(ins),
              "category_counts": df["category"].value_counts().to_dict()}
    (C.TABLES / "eval_counts.json").write_text(json.dumps(counts, indent=2, default=int))
    print(ic.pivot(index="horizon", columns="model", values="ic").loc[C.HORIZONS].round(4).to_string())
    print(diff[diff["vs"] == "finbert"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
