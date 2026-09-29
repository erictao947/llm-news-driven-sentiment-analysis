"""Event-driven long-short backtest.

One position per qualifying headline: long if the score is positive, short if negative. Equal weight: each
position is 1/MAX_CONCURRENT of capital, and a headline is skipped while MAX_CONCURRENT positions are open.
Entry at the aligned entry bar open; exit at the horizon close (30 min primary). Raw (not market-adjusted) returns.

Trade rule
  LLM:        |sentiment| >= 0.5 and confidence >= 0.6 (fixed in the brief)
  baselines:  |score| above the in-sample quantile that makes them trade the same fraction of in-sample
              headlines as the LLM rule does, so every model gets the same trade budget
  random:     the same fraction, chosen at random, random side

Costs per side: FEE_BPS_PER_SIDE + half-spread proxy (from align.py). Round trip = 2 x per side.

Usage: python -m src.backtest
"""
import heapq
import json

import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import load_panel


# ---------------------------------------------------------------- selection
def llm_rule(df):
    return (df["score_llm"].abs() >= C.SENT_THRESHOLD) & (df["conf_llm"] >= C.CONF_THRESHOLD)


def fit_thresholds(insample):
    """Per-model |score| cutoffs that match the LLM's in-sample trade rate. Uses in-sample data only."""
    rate = llm_rule(insample).mean()
    th = {"llm_trade_rate_insample": float(rate)}
    for m in ("finbert", "embed"):
        th[m] = float(insample[f"score_{m}"].abs().quantile(1 - rate))
    return th


def select(df, model, th):
    if model == "llm":
        return llm_rule(df)
    if model == "random":
        return pd.Series(np.random.default_rng(C.SEED + 1).random(len(df)) < th["llm_trade_rate_insample"], df.index)
    return df[f"score_{model}"].abs() >= th[model]


# ---------------------------------------------------------------- simulator
def trade_cost(half_spread_bps, fee_bps=C.FEE_BPS_PER_SIDE):
    """Round-trip cost as a return: pay fee + half-spread on entry and again on exit.
    A missing spread estimate is filled with the median of the batch."""
    hs = np.asarray(half_spread_bps, float)
    if np.isnan(hs).any():
        hs = np.where(np.isnan(hs), np.nanmedian(hs) if (~np.isnan(hs)).any() else 0.0, hs)
    return 2 * (fee_bps + hs) / 1e4


def simulate(cands, max_concurrent=C.MAX_CONCURRENT):
    """cands: rows with entry_ns, exit_ns sorted by entry. Returns a boolean mask of accepted trades."""
    open_exits, accepted = [], np.zeros(len(cands), bool)
    for k, (t_in, t_out) in enumerate(zip(cands["entry_ns"].to_numpy(), cands["exit_ns"].to_numpy())):
        while open_exits and open_exits[0] <= t_in:
            heapq.heappop(open_exits)
        if len(open_exits) < max_concurrent:
            heapq.heappush(open_exits, t_out)
            accepted[k] = True
    return accepted


def run(df, model, th, exit_h=C.PRIMARY_EXIT, fee_bps=C.FEE_BPS_PER_SIDE):
    d = df[select(df, model, th)].copy()
    d["exit_ns"] = d[f"exit_ns_{exit_h}"]
    d["ret"] = d[f"ret_{exit_h}"]
    d = d[(d["exit_ns"] >= 0) & d["ret"].notna()].sort_values("entry_ns")
    d = d[simulate(d)]
    d["side"] = np.sign(d[f"score_{model}"]).replace(0, 1)
    d["gross"] = d["side"] * d["ret"]
    d["cost"] = trade_cost(d["half_spread_bps"].to_numpy(), fee_bps)
    d["net"] = d["gross"] - d["cost"]
    return d


def summarize(trades, all_days, label):
    n = len(trades)
    years = len(all_days) / 252
    w = 1 / C.MAX_CONCURRENT
    daily = trades.groupby("entry_date")[["gross", "net"]].sum().reindex(all_days, fill_value=0) * w
    eq = daily.cumsum()  # additive: fixed 1/10-capital slots, no compounding
    dd = (eq["net"] - eq["net"].cummax()).min()
    out = {**label, "n_trades": n, "trades_per_day": n / len(all_days),
           "hit_rate": float((trades["gross"] > 0).mean()) if n else np.nan,
           "gross_bps": trades["gross"].mean() * 1e4, "cost_bps": trades["cost"].mean() * 1e4,
           "net_bps": trades["net"].mean() * 1e4}
    for k in ("gross", "net"):
        sd = trades[k].std()
        out[f"sharpe_trade_{k}"] = trades[k].mean() / sd * np.sqrt(n / years) if n > 2 and sd > 0 else np.nan
        dsd = daily[k].std()
        out[f"sharpe_daily_{k}"] = daily[k].mean() / dsd * np.sqrt(252) if dsd > 0 else np.nan
        out[f"total_return_{k}"] = daily[k].sum()
    out["max_drawdown_net"] = dd
    out["turnover_per_day"] = 2 * n * w / len(all_days)  # notional traded per day as a multiple of capital
    return out, daily.assign(**label).reset_index(names="date")


def main():
    ins = load_panel(sample="in")
    th = fit_thresholds(ins)
    (C.TABLES / "backtest_thresholds.json").write_text(json.dumps(th, indent=2))
    print("thresholds (fit in-sample):", th)

    panels = {d: load_panel(delay=d) for d in C.LATENCY_GRID_S}
    all_days = sorted(set().union(*(set(p["entry_date"]) for p in panels.values())))
    summary, curves, cat_rows = [], [], []

    for model in C.MODELS:
        for exit_h in C.EXIT_SENSITIVITY:
            t = run(panels[C.ENTRY_DELAY_S], model, th, exit_h)
            s, c = summarize(t, all_days, {"model": model, "exit": exit_h, "delay_s": C.ENTRY_DELAY_S})
            summary.append(s)
            curves.append(c)
            if exit_h == C.PRIMARY_EXIT:
                t.to_parquet(C.INTERIM / f"trades_{model}.parquet", index=False)
                g = t.groupby("category").agg(n_trades=("net", "size"), hit_rate=("gross", lambda x: (x > 0).mean()),
                                              gross_bps=("gross", "mean"), net_bps=("net", "mean"),
                                              pnl_net=("net", "sum"))
                g[["gross_bps", "net_bps"]] *= 1e4
                g["pnl_net"] /= C.MAX_CONCURRENT
                cat_rows.append(g.reset_index().assign(model=model))
        for delay in C.LATENCY_GRID_S:
            if delay == C.ENTRY_DELAY_S:
                continue
            t = run(panels[delay], model, th, C.PRIMARY_EXIT)
            s, c = summarize(t, all_days, {"model": model, "exit": C.PRIMARY_EXIT, "delay_s": delay})
            summary.append(s)
            curves.append(c)

    fee_rows = []
    for fee in (0.0, 1.0, 2.5, 5.0, 10.0):
        t = run(panels[C.ENTRY_DELAY_S], "llm", th, C.PRIMARY_EXIT, fee_bps=fee)
        s, _ = summarize(t, all_days, {"model": "llm", "exit": C.PRIMARY_EXIT, "delay_s": C.ENTRY_DELAY_S, "fee_bps": fee})
        fee_rows.append(s)

    S = pd.DataFrame(summary)
    S.to_csv(C.TABLES / "backtest_summary.csv", index=False)
    S[S["exit"] == C.PRIMARY_EXIT].sort_values(["model", "delay_s"]).to_csv(C.TABLES / "backtest_latency.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(C.TABLES / "backtest_fee_sensitivity.csv", index=False)
    pd.concat(cat_rows).to_csv(C.TABLES / "backtest_by_category.csv", index=False)
    pd.concat(curves).to_parquet(C.INTERIM / "equity_curves.parquet", index=False)
    cols = ["model", "exit", "delay_s", "n_trades", "hit_rate", "gross_bps", "cost_bps", "net_bps",
            "sharpe_trade_gross", "sharpe_trade_net", "sharpe_daily_net", "max_drawdown_net"]
    print(S[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
