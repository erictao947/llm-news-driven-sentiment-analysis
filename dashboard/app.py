"""Streamlit dashboard: signal decay, backtest, headline explorer, live log.

Run: streamlit run dashboard/app.py
Reads only files written by the pipeline (reports/tables, data/interim, data/scores, live/logs).
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

COLOR = {"Claude Haiku 4.5": "#2a78d6", "FinBERT": "#eb6834", "MiniLM + logit": "#1baf7a", "Random sign": "#8a8983"}
SCALE = alt.Scale(domain=list(COLOR), range=list(COLOR.values()))
HORIZON_LABEL = {"5m": "5 min", "15m": "15 min", "30m": "30 min", "60m": "60 min", "close": "Close", "next_close": "Next close"}

st.set_page_config(page_title="LLM News Sentiment", layout="wide")
st.title("LLM news sentiment: signal study")
st.caption(f"Out of sample {C.OOS_START} to {C.END}. Every number comes from the historical backtest; the live tab is a demo.")


@st.cache_data
def load(name):
    p = C.TABLES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def panel():
    from src.evaluate import load_panel
    return load_panel()


@st.cache_data
def bars(sym):
    return pd.read_parquet(C.RAW / "bars_1min" / f"{sym}.parquet")


tab1, tab2, tab3, tab4 = st.tabs(["Signal decay", "Backtest", "Headline explorer", "Live log"])

with tab1:
    ic = load("ic_by_horizon.csv")
    if ic is None:
        st.info("Run `make eval` first.")
    else:
        ic["Model"] = ic["model"].map(C.MODEL_LABELS)
        ic["Horizon"] = ic["horizon"].map(HORIZON_LABEL)
        pick = st.multiselect("Models", list(COLOR), default=list(COLOR))
        d = ic[ic["Model"].isin(pick)]
        order = [HORIZON_LABEL[h] for h in C.HORIZONS]
        base = alt.Chart(d).encode(x=alt.X("Horizon:N", sort=order, axis=alt.Axis(labelAngle=0)), color=alt.Color("Model:N", scale=SCALE))
        tip = ["Model", "Horizon", alt.Tooltip("ic:Q", format="+.3f"), alt.Tooltip("ci_lo:Q", format="+.3f"),
               alt.Tooltip("ci_hi:Q", format="+.3f"), "n:Q", alt.Tooltip("hit_rate:Q", format=".1%")]
        chart = (base.mark_line(strokeWidth=2) + base.mark_point(size=60, filled=True).encode(y="ic:Q", tooltip=tip)
                 + base.mark_rule().encode(y="ci_lo:Q", y2="ci_hi:Q")).encode(y=alt.Y("ic:Q", title="Rank IC (SPY-excess)"))
        st.altair_chart(chart.properties(height=380), width="stretch")
        st.dataframe(d.pivot(index="Horizon", columns="Model", values="ic").reindex(order).round(4))
        cat = load("ic_by_category.csv")
        if cat is not None:
            st.subheader("IC at 30 min by LLM category")
            st.dataframe(cat.assign(Model=cat["model"].map(C.MODEL_LABELS)).pivot(index="category", columns="Model", values="ic").round(4))

with tab2:
    eqp = C.INTERIM / "equity_curves.parquet"
    if not eqp.exists():
        st.info("Run `make backtest` first.")
    else:
        eq = pd.read_parquet(eqp)
        c1, c2, c3 = st.columns(3)
        costs = c1.radio("Costs", ["net", "gross"], horizontal=True)
        delay = c2.select_slider("Entry delay (s)", options=C.LATENCY_GRID_S, value=C.ENTRY_DELAY_S)
        exit_h = c3.selectbox("Exit", C.EXIT_SENSITIVITY, index=C.EXIT_SENSITIVITY.index(C.PRIMARY_EXIT)) if delay == C.ENTRY_DELAY_S else C.PRIMARY_EXIT
        d = eq[(eq["delay_s"] == delay) & (eq["exit"] == exit_h)].sort_values("date").copy()
        d["Model"] = d["model"].map(C.MODEL_LABELS)
        d["cum"] = d.groupby("model")[costs].cumsum() * 100
        d["date"] = pd.to_datetime(d["date"])
        st.altair_chart(alt.Chart(d).mark_line(strokeWidth=2).encode(
            x="date:T", y=alt.Y("cum:Q", title=f"Cumulative {costs} return, % capital"),
            color=alt.Color("Model:N", scale=SCALE), tooltip=["Model", "date:T", alt.Tooltip("cum:Q", format="+.2f")]
        ).properties(height=380), width="stretch")
        s = load("backtest_summary.csv")
        s = s[(s["delay_s"] == delay) & (s["exit"] == exit_h)]
        st.dataframe(s.assign(model=s["model"].map(C.MODEL_LABELS)).set_index("model")[[
            "n_trades", "hit_rate", "gross_bps", "cost_bps", "net_bps", "sharpe_trade_gross", "sharpe_trade_net",
            "sharpe_daily_net", "max_drawdown_net", "turnover_per_day"]].round(3))

with tab3:
    try:
        df = panel()
    except FileNotFoundError:
        df = None
        st.info("Run the pipeline first.")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        tick = c1.multiselect("Ticker", C.UNIVERSE)
        cats = c2.multiselect("Category", sorted(df["category"].dropna().unique()))
        lo, hi = c3.slider("LLM sentiment", -1.0, 1.0, (-1.0, 1.0), 0.05)
        q = df[(df["score_llm"] >= lo) & (df["score_llm"] <= hi)]
        if tick:
            q = q[q["ticker"].isin(tick)]
        if cats:
            q = q[q["category"].isin(cats)]
        q = q.assign(time_et=q["ts"].dt.tz_convert(C.TZ).dt.strftime("%Y-%m-%d %H:%M:%S"))
        cols = ["time_et", "ticker", "headline", "category", "score_llm", "conf_llm", "score_finbert", "xret_30m", "xret_close"]
        st.caption(f"{len(q):,} events")
        view = q[cols].head(2000).reset_index(drop=True)
        sel = st.dataframe(view, on_select="rerun", selection_mode="single-row", width="stretch")
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            r = q.iloc[rows[0]]
            b = bars(r["ticker"])
            win = b[(b["ts"] >= r["entry_ts"] - pd.Timedelta(minutes=30)) & (b["ts"] <= r["entry_ts"] + pd.Timedelta(minutes=90))].copy()
            win["ret_bps"] = (win["close"] / r["entry_px"] - 1) * 1e4
            win["time_et"] = win["ts"].dt.tz_convert(C.TZ).dt.tz_localize(None)
            st.markdown(f"**{r['ticker']}**: {r['headline']}  \nLLM {r['score_llm']:+.2f} (conf {r['conf_llm']:.2f}, {r['category']}), "
                        f"FinBERT {r['score_finbert']:+.2f}")
            line = alt.Chart(win).mark_line(strokeWidth=2, color="#2a78d6").encode(
                x=alt.X("time_et:T", title="New York time"), y=alt.Y("ret_bps:Q", title="Return from entry, bps"),
                tooltip=[alt.Tooltip("time_et:T", format="%H:%M"), alt.Tooltip("ret_bps:Q", format="+.1f")])
            entry = alt.Chart(pd.DataFrame({"t": [r["entry_ts"].tz_convert(C.TZ).tz_localize(None)]})).mark_rule(
                color="#52514e", strokeDash=[4, 3]).encode(x="t:T")
            st.altair_chart((line + entry).properties(height=320), width="stretch")

with tab4:
    logs = sorted((C.ROOT / "live" / "logs").glob("*.jsonl"))
    if not logs:
        st.info("No live session logged yet. Run `python -m live.stream --dry-run` during market hours.")
    else:
        f = st.selectbox("Session", logs, format_func=lambda p: p.stem, index=len(logs) - 1)
        L = pd.DataFrame([json.loads(x) for x in f.read_text().splitlines()])
        scored = L[L["decision"] != "exit"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Headlines scored", len(scored))
        c2.metric("Orders", int(scored["decision"].isin(["order_submitted", "dry_run_order"]).sum()))
        if "latency_s" in scored:
            c3.metric("Median headline-to-score latency", f"{scored['latency_s'].median():.1f} s")
        st.dataframe(L, width="stretch")
