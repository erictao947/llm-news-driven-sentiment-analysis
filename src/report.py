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
from src.evaluate import load_panel
from src.figures import HORIZON_LABEL

T = C.TABLES


def f(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def md_table(df, cols, headers, fmts):
    """Pipe table whose separator dashes are proportional to content width, which pandoc uses for column widths."""
    body = [[fmts[i](r[c]) for i, c in enumerate(cols)] for _, r in df.iterrows()]
    widths = [max([len(h)] + [len(row[i]) for row in body]) for i, h in enumerate(headers)]
    sep = "|" + "|".join(("-" * (w + 2)) + ("" if i == 0 else ":") for i, w in enumerate(widths)) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep] + ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def ci_str(r, nd=3):
    return f"[{r['ci_lo']:+.{nd}f}, {r['ci_hi']:+.{nd}f}]"


def short(x, nd=3):
    """+0.013 -> +.013, compact for wide tables"""
    return f"{x:+.{nd}f}".replace("0.", ".", 1)


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
    V["llm_spend_estimated"] = f"{cost_log.loc[cost_log['window'].str.contains('estimated'), 'est_usd'].sum():.2f}"
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
        rows.append({"h": HORIZON_LABEL[h], **{m: f"{short(g(m, h)['ic'])} [{short(g(m, h)['ci_lo'], 2)}, {short(g(m, h)['ci_hi'], 2)}]"
                                                for m in C.MODELS}})
    V["table_ic"] = md_table(pd.DataFrame(rows), ["h"] + C.MODELS,
                             ["Horizon"] + [C.MODEL_SHORT[m] for m in C.MODELS], [str] * (len(C.MODELS) + 1))
    V["ic_n_events"] = f"{g('llm', C.PRIMARY_EXIT)['n']:,}"

    d = diff[diff["vs"] == "finbert"].set_index("horizon")
    d30 = d.loc[C.PRIMARY_EXIT]
    V["diff_30m"] = f(d30["diff"], signed=True)
    V["diff_30m_ci"] = f"[{d30['ci_lo']:+.3f}, {d30['ci_hi']:+.3f}]"
    fb30 = g("finbert", C.PRIMARY_EXIT)
    V["ratio_30m"] = (f"{d30['ratio']:.1f}x" if fb30["ci_lo"] > 0 and r30["ci_lo"] > 0
                      else "not meaningful, because the FinBERT IC is not distinguishable from zero")
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
            else f"{p[k]:+.2f}" if ("sharpe" in k or "bps" in k) and k != "cost_bps" else f"{p[k]:.2f}"
    V["bt_spread_bps"] = f"{p['cost_bps'] - 2 * C.FEE_BPS_PER_SIDE:.1f}"
    rows = []
    for m in C.MODELS:
        for e in C.EXIT_SENSITIVITY:
            r = b(m, e)
            rows.append({"model": C.MODEL_SHORT[m], "exit": HORIZON_LABEL[e],
                         "gross_ci": f"[{r['gross_bps_ci_lo']:+.0f}, {r['gross_bps_ci_hi']:+.0f}]",
                         **r[["n_trades", "hit_rate", "gross_bps", "net_bps",
                         "sharpe_trade_gross", "sharpe_trade_net", "max_drawdown_net"]].to_dict()})
    pct = lambda x: f"{x:.1%}"
    num = lambda x: f"{x:+.2f}"
    num3 = lambda x: f"{x:+.3f}"
    V["table_bt"] = md_table(pd.DataFrame(rows), ["model", "exit", "n_trades", "hit_rate", "gross_bps", "gross_ci", "net_bps",
                                                  "sharpe_trade_gross", "sharpe_trade_net", "max_drawdown_net"],
                             ["Model", "Exit", "Trades", "Hit", "Gross", "Gross CI", "Net", "SR gross", "SR net", "Max DD"],
                             [str, str, lambda x: f"{int(x):,}", pct, lambda x: f"{x:+.1f}", str, lambda x: f"{x:+.1f}", num, num, pct])

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

    # what killed it: test the gross edge before blaming costs
    gross, cost = p["gross_bps"], p["cost_bps"]
    glo, ghi = p["gross_bps_ci_lo"], p["gross_bps_ci_hi"]
    V["bt_gross_ci"] = f"[{glo:+.1f}, {ghi:+.1f}]"
    fee0 = fee.set_index("fee_bps").loc[0.0]
    if glo > 0 and gross < cost:
        V["killer_phrase"] = (f"Costs. The rule earns {gross:+.1f} bps gross per trade (95% CI {V['bt_gross_ci']}), "
                              f"but the round trip costs {cost:.1f} bps.")
    elif glo <= 0 < ghi:
        V["killer_phrase"] = (f"The signal first, then costs. At the 30-minute exit the gross edge is {gross:+.1f} bps per trade "
                              f"with a 95% CI of {V['bt_gross_ci']}, so it cannot be told apart from zero. Costs then remove any doubt: "
                              f"{cost:.1f} bps per round trip ({2 * C.FEE_BPS_PER_SIDE:.0f} bps of fees, {cost - 2 * C.FEE_BPS_PER_SIDE:.1f} bps of "
                              f"estimated spread), and even with zero fees the net edge is {fee0['net_bps']:+.1f} bps.")
    elif ghi <= 0:
        V["killer_phrase"] = f"The signal. The gross edge is {gross:+.1f} bps per trade (CI {V['bt_gross_ci']}), negative before costs."
    else:
        V["killer_phrase"] = f"Nothing at the 60-second delay: gross edge {gross:+.1f} bps (CI {V['bt_gross_ci']}) covers {cost:.1f} bps of costs."
    q = b("llm", "5m")
    V["bt5_gross"], V["bt5_net"], V["bt5_cost"] = f"{q['gross_bps']:+.1f}", f"{q['net_bps']:+.1f}", f"{q['cost_bps']:.1f}"
    V["bt5_gross_ci"] = f"[{q['gross_bps_ci_lo']:+.1f}, {q['gross_bps_ci_hi']:+.1f}]"
    V["bt5_n"] = f"{int(q['n_trades']):,}"
    V["n_bt_configs"] = str(len(C.EXIT_SENSITIVITY) + len(C.LATENCY_GRID_S) - 1)
    rth = pd.read_csv(T / "backtest_latency_rth.csv")
    r5 = rth[(rth["model"] == "llm") & (rth["exit"] == "5m") & (rth["delay_s"] == C.ENTRY_DELAY_S)].iloc[0]
    V["bt5_rth_gross"], V["bt5_rth_n"] = f"{r5['gross_bps']:+.1f}", f"{int(r5['n_trades']):,}"
    V["bt5_rth_gross_ci"] = f"[{r5['gross_bps_ci_lo']:+.1f}, {r5['gross_bps_ci_hi']:+.1f}]"
    tr = pd.read_parquet(C.INTERIM / "trades_llm.parquet")
    V["bt_rth_share"] = f"{tr['headline_in_rth'].mean():.0%}"

    # latency on regular-hours headlines
    icd = pd.read_csv(T / "ic_by_delay_rth.csv")
    icd = icd[icd["horizon"] == C.PRIMARY_EXIT]
    L = icd[icd["model"] == "llm"].set_index("delay_s")
    for dl in C.LATENCY_GRID_S:
        V[f"icd_llm_{dl}"] = f(L.loc[dl, "ic"], signed=True)
        V[f"icd_llm_ci_{dl}"] = ci_str(L.loc[dl])
    V["icd_n"] = f"{int(L.loc[C.ENTRY_DELAY_S, 'n']):,}"
    sig_d = [dl for dl in C.LATENCY_GRID_S if L.loc[dl, "ci_lo"] > 0]
    V["latency_ic_phrase"] = (f"the CI excludes zero only at {' and '.join(f'{x} s' for x in sig_d)}" if sig_d
                              else "the CI includes zero at every delay")
    rows = []
    for dl in C.LATENCY_GRID_S:
        rows.append({"d": f"{dl} s", **{m: (lambda r: f"{short(r['ic'])} [{short(r['ci_lo'], 2)}, {short(r['ci_hi'], 2)}]")(
        icd[(icd["model"] == m) & (icd["delay_s"] == dl)].iloc[0]) for m in C.MODELS}})
    V["table_ic_delay"] = md_table(pd.DataFrame(rows), ["d"] + C.MODELS, ["Entry delay"] + [C.MODEL_SHORT[m] for m in C.MODELS],
                                   [str] * (len(C.MODELS) + 1))

    # batch vs single-call agreement
    ag = pd.read_csv(T / "llm_batch_vs_single.csv").iloc[0]
    V["ag_n"] = f"{int(ag['n']):,}"
    V["ag_pearson"], V["ag_spearman"] = f"{ag['sentiment_pearson']:.2f}", f"{ag['sentiment_spearman']:.2f}"
    V["ag_sign"], V["ag_cat"], V["ag_trade"] = f"{ag['sign_agreement_nonzero']:.0%}", f"{ag['category_agreement']:.0%}", f"{ag['trade_flag_agreement']:.0%}"

    # embed baseline significance
    em = ic[(ic["model"] == "embed") & (ic["ci_lo"] > 0)]
    V["embed_sig_phrase"] = ("; ".join(f"{HORIZON_LABEL[h]}: {r:+.3f}" for h, r in zip(em["horizon"], em["ic"])) if len(em) else "none")
    V["ic_embed_5m_full"] = f"{g('embed', '5m')['ic']:+.3f} {ci_str(g('embed', '5m'))}"
    V["n_cat_tests"] = str(cat["category"].nunique() * len(C.MODELS))

    # categories and sessions
    cl = cat[cat["model"] == "llm"].sort_values("ic", ascending=False)
    V["best_cat"], V["best_cat_ic"], V["best_cat_n"] = cl.iloc[0]["category"], f(cl.iloc[0]["ic"], signed=True), f"{cl.iloc[0]['n']:,}"
    V["best_cat_ci"] = ci_str(cl.iloc[0])
    V["worst_cat"], V["worst_cat_ic"] = cl.iloc[-1]["category"], f(cl.iloc[-1]["ic"], signed=True)
    V["best_cat_absret"] = f"{cl.iloc[0]['mean_abs_ret_bps']:.1f}"
    big = cl.sort_values("mean_abs_ret_bps", ascending=False).iloc[0]
    V["biggest_move_cat"], V["biggest_move_bps"] = big["category"].replace("_", " ").capitalize(), f"{big['mean_abs_ret_bps']:.1f}"
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
    pan = load_panel()
    V["share_other"] = f"{(pan['category'] == 'other').mean():.0%}"
    cb = conf[conf["model"] == "llm"]
    V["share_conf_low"] = f"{cb.loc[cb['llm_conf_bucket'].str.endswith('0.4]'), 'n'].sum() / cb['n'].sum():.0%}"
    V["share_near_zero"] = f"{(pan['score_llm'].abs() < 0.1).mean():.0%}"
    single = cost_log[cost_log["prompt_version"] == C.PROMPT_VERSION]
    batch = cost_log[cost_log["prompt_version"].str.contains("batch")]
    V["single_usd_per_1k"] = f"{single['est_usd'].sum() / single['calls'].sum() * 1000:.2f}"
    n_batch_events = sum(1 for _ in open(C.CACHE / f"llm_{C.PROMPT_VERSION}_batch.jsonl"))
    V["batch_usd_per_1k"] = f"{batch['est_usd'].sum() / n_batch_events * 1000:.2f}"
    V["n_ic_tests"] = str(len(C.MODELS) * len(C.HORIZONS))

    for k, v in [("sample_start", C.START), ("sample_end", C.END), ("insample_end", C.INSAMPLE_END), ("oos_start", C.OOS_START)]:
        V[k] = v
    return V


def render(V):
    tpl = (C.REPORTS / "writeup_template.md").read_text().replace("{{post_run_notes}}", V["post_run_notes"])
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
        f"- **Backtest (30-min exit, 60 s delay):** {V['bt_n_trades']} trades, {V['bt_gross_bps']} bps gross per trade "
        f"(95% CI {V['bt_gross_ci']}), {V['bt_net_bps']} bps net after {V['bt_cost_bps']} bps of costs; "
        f"Sharpe {V['bt_sharpe_trade_gross']} gross, {V['bt_sharpe_trade_net']} net.",
        f"- **Latency (regular-hours headlines, 30-min IC):** {V['icd_llm_0']} at 0 s, {V['icd_llm_60']} at 60 s, "
        f"{V['icd_llm_300']} at 300 s, {V['icd_llm_900']} at 900 s.",
        f"- **Verdict:** {V['killer_phrase']}",
        f"- **LLM API spend:** ${V['llm_spend_usd']} for all {V['events']} events (Claude Haiku 4.5, Batches API).",
        "<!-- results:end -->"])
    p = C.ROOT / "README.md"
    txt = p.read_text()
    txt = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, txt, flags=re.S)
    p.write_text(txt)


def sample_csv():
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
