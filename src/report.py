"""Render reports/writeup.md from reports/writeup_template.md and the result tables, then build the PDF.

Every number in the writeup is a {{placeholder}} filled from reports/tables/*. Verdict phrases (significant or
not, beat FinBERT or not) are computed from the confidence intervals, so the prose cannot overstate the result.
The rendered values are also saved to reports/results.json. The build fails on any unresolved placeholder.

Usage: python -m src.report [--no-pdf]
"""
import argparse
import json
import re
import subprocess

import numpy as np
import pandas as pd

from src import config as C
from src.figures import HORIZON_LABEL

T = C.TABLES


def f(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def md_table(df, cols, headers, fmts):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmts[i](r[c]) for i, c in enumerate(cols)) + " |")
    return "\n".join(lines)


def ci_str(r, nd=3):
    return f"[{r['ci_lo']:+.{nd}f}, {r['ci_hi']:+.{nd}f}]"


def build_values():
    V = {}
    ing = json.loads((T / "ingest_counts.json").read_text())
    ev = json.loads((T / "eval_counts.json").read_text())
    ic = pd.read_csv(T / "ic_by_horizon.csv")
    diff = pd.read_csv(T / "ic_llm_vs_baselines.csv")
    bt = pd.read_csv(T / "backtest_summary.csv")
    lat = pd.read_csv(T / "backtest_latency.csv")
    cat = pd.read_csv(T / "ic_by_category.csv")
    sess = pd.read_csv(T / "ic_by_session.csv")
    conf = pd.read_csv(T / "ic_by_llm_confidence.csv")
    th = json.loads((T / "backtest_thresholds.json").read_text())
    fee = pd.read_csv(T / "backtest_fee_sensitivity.csv")
    cost_log = pd.read_csv(C.REPORTS / "llm_cost_log.csv")

    V.update({k: f"{v:,}" for k, v in ing.items() if isinstance(v, int)})
    V["multi_ticker_share"] = f"{ing['multi_ticker_share']:.0%}"
    V["oos_events"] = f"{ev['oos_events']:,}"
    V["oos_days"] = f"{ev['oos_days']:,}"
    V["oos_rth_share"] = f"{ev['oos_events_rth'] / ev['oos_events']:.0%}"
    V["llm_spend_usd"] = f"{cost_log['est_usd'].sum():.2f}"
    V["llm_calls"] = f"{int(cost_log['calls'].sum()):,}"
    V["llm_trade_rate"] = f"{th['llm_trade_rate_insample']:.1%}"

    # IC
    g = lambda m, h: ic[(ic["model"] == m) & (ic["horizon"] == h)].iloc[0]
    for m in C.MODELS:
        for h in C.HORIZONS:
            r = g(m, h)
            V[f"ic_{m}_{h}"] = f(r["ic"], signed=True)
            V[f"ci_{m}_{h}"] = ci_str(r)
            V[f"hit_{m}_{h}"] = f"{r['hit_rate']:.1%}"
    llm = ic[ic["model"] == "llm"].set_index("horizon").loc[C.HORIZONS]
    peak_h = llm["ic"].idxmax()
    V["llm_peak_h"] = HORIZON_LABEL[peak_h]
    V["llm_peak_ic"] = f(llm.loc[peak_h, "ic"], signed=True)
    sig = llm[(llm["ci_lo"] > 0)]
    V["llm_sig_horizons"] = ", ".join(HORIZON_LABEL[h] for h in sig.index) if len(sig) else "none"
    r30 = llm.loc[C.PRIMARY_EXIT]
    V["llm_30m_sig_phrase"] = ("statistically different from zero (the 95% CI excludes zero)" if r30["ci_lo"] > 0 or r30["ci_hi"] < 0
                               else "not statistically different from zero (the 95% CI includes zero)")
    half = llm["ic"].iloc[0] / 2
    below = [h for h in C.HORIZONS if llm.loc[h, "ic"] < half]
    V["llm_half_life_phrase"] = (f"falls below half its 5-minute level by the {HORIZON_LABEL[below[0]]} horizon"
                                 if below and llm["ic"].iloc[0] > 0 else "does not show a clean decay from its 5-minute level")

    ic_tab = ic.pivot(index="horizon", columns="model", values="ic").loc[C.HORIZONS].reset_index()
    ci_tab = ic.assign(ci=ic.apply(ci_str, axis=1, nd=2)).pivot(index="horizon", columns="model", values="ci").loc[C.HORIZONS]
    rows = []
    for h in C.HORIZONS:
        rows.append({"h": HORIZON_LABEL[h], **{m: f"{g(m, h)['ic']:+.3f} {ci_tab.loc[h, m]}" for m in C.MODELS},
                     "n": f"{g('llm', h)['n']:,}"})
    V["table_ic"] = md_table(pd.DataFrame(rows), ["h"] + C.MODELS + ["n"],
                             ["Horizon"] + [C.MODEL_SHORT[m] for m in C.MODELS] + ["Events"],
                             [str] * (len(C.MODELS) + 2))

    d = diff[diff["vs"] == "finbert"].set_index("horizon")
    d30 = d.loc[C.PRIMARY_EXIT]
    V["diff_30m"] = f(d30["diff"], signed=True)
    V["diff_30m_ci"] = f"[{d30['ci_lo']:+.3f}, {d30['ci_hi']:+.3f}]"
    V["ratio_30m"] = f"{d30['ratio']:.1f}x" if d30["ic_other"] > 0 and d30["ic_llm"] > 0 else "not meaningful (a baseline IC is at or below zero)"
    V["p_diff_30m"] = f"{d30['p_boot_le0']:.3f}"
    V["llm_vs_finbert_verdict"] = ("The LLM beat FinBERT at 30 minutes, and the paired bootstrap CI on the difference excludes zero."
                                   if d30["ci_lo"] > 0 else
                                   "The LLM did not beat FinBERT by a statistically reliable margin at 30 minutes: the paired bootstrap CI on the difference includes zero."
                                   if d30["ci_hi"] > 0 else
                                   "FinBERT beat the LLM at 30 minutes, and the paired CI on the difference excludes zero.")
    dt = d.reset_index()
    dt["h"] = dt["horizon"].map(HORIZON_LABEL)
    V["table_diff"] = md_table(dt, ["h", "ic_llm", "ic_other", "diff", "ci_lo", "ci_hi", "p_boot_le0"],
                               ["Horizon", "LLM IC", "FinBERT IC", "Difference", "CI low", "CI high", "Bootstrap P(diff <= 0)"],
                               [str] + [lambda x: f"{x:+.3f}"] * 5 + [lambda x: f"{x:.3f}"])

    # backtest
    b = lambda m, e=C.PRIMARY_EXIT, dl=C.ENTRY_DELAY_S: bt[(bt["model"] == m) & (bt["exit"] == e) & (bt["delay_s"] == dl)].iloc[0]
    p = b("llm")
    for k in ("n_trades", "hit_rate", "gross_bps", "cost_bps", "net_bps", "sharpe_trade_gross", "sharpe_trade_net",
              "sharpe_daily_gross", "sharpe_daily_net", "max_drawdown_net", "turnover_per_day", "trades_per_day",
              "total_return_net", "total_return_gross"):
        V[f"bt_{k}"] = f"{p[k]:,.0f}" if k == "n_trades" else f"{p[k]:.1%}" if k in ("hit_rate", "max_drawdown_net", "total_return_net", "total_return_gross") \
            else f"{p[k]:+.2f}" if "sharpe" in k or "bps" in k else f"{p[k]:.2f}"
    V["bt_spread_bps"] = f"{p['cost_bps'] - 2 * C.FEE_BPS_PER_SIDE:.1f}"
    rows = []
    for m in C.MODELS:
        for e in C.EXIT_SENSITIVITY:
            r = b(m, e)
            rows.append({"model": C.MODEL_SHORT[m], "exit": HORIZON_LABEL[e], **r[["n_trades", "hit_rate", "gross_bps", "net_bps",
                         "sharpe_trade_gross", "sharpe_trade_net", "sharpe_daily_net", "max_drawdown_net"]].to_dict()})
    pct = lambda x: f"{x:.1%}"
    num = lambda x: f"{x:+.2f}"
    num3 = lambda x: f"{x:+.3f}"
    V["table_bt"] = md_table(pd.DataFrame(rows), ["model", "exit", "n_trades", "hit_rate", "gross_bps", "net_bps",
                                                  "sharpe_trade_gross", "sharpe_trade_net", "sharpe_daily_net", "max_drawdown_net"],
                             ["Model", "Exit", "Trades", "Hit", "Gross bps", "Net bps", "Sharpe gross", "Sharpe net",
                              "Daily SR net", "Max DD (additive)"],
                             [str, str, lambda x: f"{int(x):,}", pct, num, num, num, num, num, pct])

    lat = lat.assign(model_l=lat["model"].map(C.MODEL_SHORT), delay=lat["delay_s"].map(lambda s: f"{s}s"))
    V["table_latency"] = md_table(lat, ["model_l", "delay", "n_trades", "gross_bps", "net_bps", "sharpe_trade_gross", "sharpe_trade_net"],
                                  ["Model", "Entry delay", "Trades", "Gross bps", "Net bps", "Sharpe gross", "Sharpe net"],
                                  [str, str, lambda x: f"{int(x):,}", num, num, num, num])
    ll = lat[lat["model"] == "llm"].set_index("delay_s")
    for dl in C.LATENCY_GRID_S:
        V[f"lat_gross_{dl}"] = f"{ll.loc[dl, 'gross_bps']:+.2f}"
        V[f"lat_net_{dl}"] = f"{ll.loc[dl, 'net_bps']:+.2f}"
        V[f"lat_sharpe_net_{dl}"] = f"{ll.loc[dl, 'sharpe_trade_net']:+.2f}"
    gross_pos = [dl for dl in C.LATENCY_GRID_S if ll.loc[dl, "gross_bps"] > 0]
    net_pos = [dl for dl in C.LATENCY_GRID_S if ll.loc[dl, "net_bps"] > 0]
    V["latency_death_phrase"] = (
        "The strategy is net-negative at every entry delay tested, including zero." if not net_pos else
        f"Net edge stays positive through a {max(net_pos)}-second delay." if max(net_pos) == max(C.LATENCY_GRID_S) else
        f"Net edge is positive at {', '.join(f'{x}s' for x in net_pos)} and gone by {min(x for x in C.LATENCY_GRID_S if x > max(net_pos))} seconds.")
    V["latency_gross_phrase"] = (f"Gross edge is still positive at {max(gross_pos)} seconds." if gross_pos else
                                 "Gross edge is not positive at any delay tested.")

    fz = fee.set_index("fee_bps")
    V["net_bps_fee0"] = f"{fz.loc[0.0, 'net_bps']:+.2f}"
    V["sharpe_net_fee0"] = f"{fz.loc[0.0, 'sharpe_trade_net']:+.2f}"
    V["table_fee"] = md_table(fee, ["fee_bps", "gross_bps", "cost_bps", "net_bps", "sharpe_trade_net"],
                              ["Fee bps / side", "Gross bps", "Total cost bps", "Net bps", "Sharpe net"],
                              [lambda x: f"{x:.1f}", num, num, num, num])

    # what killed it
    gross, cost = p["gross_bps"], p["cost_bps"]
    if gross <= 0:
        V["killer_phrase"] = ("The signal. Even before costs, the trade rule loses money at a 30-minute exit, so costs and latency "
                              "are not the binding constraint.")
    elif gross < cost:
        V["killer_phrase"] = (f"Costs. The rule earns {gross:+.1f} bps gross per trade but pays {cost:.1f} bps round trip "
                              f"({2 * C.FEE_BPS_PER_SIDE:.0f} bps of fees and {cost - 2 * C.FEE_BPS_PER_SIDE:.1f} bps of estimated spread). "
                              "The signal is real enough to be positive gross, but too small to cover a retail-scale cost stack.")
    else:
        V["killer_phrase"] = (f"Nothing killed it at the 60-second delay: gross edge of {gross:+.1f} bps covers {cost:.1f} bps of costs. "
                              "Latency is the main risk, as the delay table shows.")

    # categories and sessions
    cl = cat[cat["model"] == "llm"].sort_values("ic", ascending=False)
    V["best_cat"], V["best_cat_ic"], V["best_cat_n"] = cl.iloc[0]["category"], f(cl.iloc[0]["ic"], signed=True), f"{cl.iloc[0]['n']:,}"
    V["best_cat_ci"] = ci_str(cl.iloc[0])
    V["worst_cat"], V["worst_cat_ic"] = cl.iloc[-1]["category"], f(cl.iloc[-1]["ic"], signed=True)
    V["best_cat_absret"] = f"{cl.iloc[0]['mean_abs_ret_bps']:.1f}"
    big = cl.sort_values("mean_abs_ret_bps", ascending=False).iloc[0]
    V["biggest_move_cat"], V["biggest_move_bps"] = big["category"], f"{big['mean_abs_ret_bps']:.1f}"
    ct = cat[cat["model"].isin(["llm", "finbert"])].pivot(index="category", columns="model", values="ic")
    ct = ct.join(cl.set_index("category")[["n", "hit_rate", "mean_abs_ret_bps", "ci_lo", "ci_hi"]]).sort_values("llm", ascending=False).reset_index()
    V["table_cat"] = md_table(ct, ["category", "n", "llm", "ci_lo", "ci_hi", "finbert", "mean_abs_ret_bps"],
                              ["Category", "Events", "LLM IC", "CI low", "CI high", "FinBERT IC", "Mean |move| bps"],
                              [str, lambda x: f"{int(x):,}", num3, num3, num3, num3, lambda x: f"{x:.1f}"])
    btc = pd.read_csv(T / "backtest_by_category.csv")
    btc = btc[btc["model"] == "llm"].sort_values("pnl_net", ascending=False)
    V["table_bt_cat"] = md_table(btc, ["category", "n_trades", "hit_rate", "gross_bps", "net_bps", "pnl_net"],
                                 ["Category", "Trades", "Hit", "Gross bps", "Net bps", "Net PnL, % capital"],
                                 [str, lambda x: f"{int(x):,}", pct, num, num, pct])
    sl = sess[sess["model"] == "llm"].set_index("session")
    for s in ("regular_hours", "overnight"):
        if s in sl.index:
            V[f"ic_sess_{s}"] = f(sl.loc[s, "ic"], signed=True)
            V[f"ci_sess_{s}"] = ci_str(sl.loc[s])
            V[f"n_sess_{s}"] = f"{sl.loc[s, 'n']:,}"
    cf = conf[conf["model"] == "llm"]
    V["table_conf"] = md_table(cf, ["llm_conf_bucket", "n", "ic", "ci_lo", "ci_hi"],
                               ["LLM confidence", "Events", "IC", "CI low", "CI high"],
                               [str, lambda x: f"{int(x):,}", num3, num3, num3])
    emb = pd.read_csv(T / "embed_training.csv").iloc[0]
    V["embed_cv_auc"] = f"{emb['cv_auc']:.3f}"
    V["embed_n_train"] = f"{int(emb['n_train']):,}"

    notes = C.REPORTS / "notes_post_run.md"
    if not notes.exists():
        raise SystemExit("reports/notes_post_run.md missing: write the 'what didn't work' notes after reading the results")
    V["post_run_notes"] = notes.read_text().strip()

    for k, v in [("sample_start", C.START), ("sample_end", C.END), ("insample_end", C.INSAMPLE_END), ("oos_start", C.OOS_START)]:
        V[k] = v
    return V


def render(V):
    tpl = (C.REPORTS / "writeup_template.md").read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(V))
    if missing:
        raise SystemExit(f"unresolved placeholders: {missing}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: V[m.group(1)], tpl)
    (C.REPORTS / "writeup.md").write_text(out)
    (C.REPORTS / "results.json").write_text(json.dumps({k: v for k, v in V.items() if not k.startswith("table_")}, indent=2))
    print("wrote reports/writeup.md and reports/results.json")


def pdf():
    subprocess.run(["pandoc", "writeup.md", "-o", "writeup.pdf", "--pdf-engine=typst", "--resource-path=."], cwd=C.REPORTS, check=True)
    print("wrote reports/writeup.pdf")


def readme(V):
    block = "\n".join([
        "<!-- results:start -->",
        "![IC by horizon](reports/figures/ic_decay.png)",
        "",
        f"- **Sample:** {V['events']} headline-ticker events, {V['oos_events']} out of sample over {V['oos_days']} trading days.",
        f"- **Decay:** LLM rank IC {V['ic_llm_5m']} at 5 min, {V['ic_llm_30m']} at 30 min {V['ci_llm_30m']}, {V['ic_llm_close']} by the close.",
        f"- **vs FinBERT at 30 min:** IC difference {V['diff_30m']}, 95% CI {V['diff_30m_ci']}.",
        f"- **Backtest (30-min exit, 60 s delay):** {V['bt_n_trades']} trades, {V['bt_gross_bps']} bps gross, {V['bt_net_bps']} bps net per trade; "
        f"Sharpe {V['bt_sharpe_trade_gross']} gross, {V['bt_sharpe_trade_net']} net.",
        f"- **Latency:** net bps per trade {V['lat_net_0']} at 0 s, {V['lat_net_60']} at 60 s, {V['lat_net_300']} at 300 s, {V['lat_net_900']} at 900 s.",
        f"- **LLM API spend:** ${V['llm_spend_usd']} for {V['llm_calls']} calls.",
        "<!-- results:end -->"])
    p = C.ROOT / "README.md"
    txt = p.read_text()
    txt = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, txt, flags=re.S)
    p.write_text(txt)


def sample_csv():
    from src.evaluate import load_panel
    cols = ["event_id", "ticker", "ts", "headline", "category", "score_llm", "conf_llm", "score_finbert", "score_embed",
            "entry_ts", "entry_px", "half_spread_bps", "xret_5m", "xret_30m", "xret_close"]
    load_panel()[cols].head(200).to_csv(C.DATA / "sample" / "oos_sample.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    V = build_values()
    render(V)
    readme(V)
    sample_csv()
    if not a.no_pdf:
        pdf()


if __name__ == "__main__":
    main()
