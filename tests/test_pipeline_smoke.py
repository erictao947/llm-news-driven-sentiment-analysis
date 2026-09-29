"""End-to-end smoke test on synthetic data in a temp directory: align -> evaluate -> backtest -> figures -> report.

Scores carry a planted signal, so the test also checks that the pipeline recovers a positive IC for the
informed model and roughly zero for the random control. Nothing here touches data/ or reports/.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C

TICKERS = ["AAPL", "MSFT", "JPM", "XOM"]


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("sandbox")
    mp = pytest.MonkeyPatch()
    for name in ("RAW", "INTERIM", "SCORES", "CACHE", "REPORTS", "TABLES", "FIGURES"):
        p = root / name.lower()
        p.mkdir()
        mp.setattr(C, name, p)
    (C.RAW / "bars_1min").mkdir()
    rng = np.random.default_rng(7)

    days = pd.bdate_range("2025-10-01", "2026-01-30")
    cal = pd.DataFrame({"date": days.date,
                        "open": [pd.Timestamp(f"{d.date()} 09:30", tz=C.TZ).tz_convert("UTC") for d in days],
                        "close": [pd.Timestamp(f"{d.date()} 16:00", tz=C.TZ).tz_convert("UTC") for d in days]})
    cal.to_parquet(C.RAW / "calendar.parquet", index=False)

    mins = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min", tz=C.TZ).tz_convert("UTC") for d in days]))
    mkt = rng.normal(0, 3e-4, len(mins))
    for sym in TICKERS + [C.MARKET]:
        r = mkt + (0 if sym == C.MARKET else rng.normal(0, 6e-4, len(mins)))
        px = 100 * np.exp(np.cumsum(r))
        pd.DataFrame({"ts": mins, "open": px / np.exp(r), "close": px, "high": px * 1.0004,
                      "low": px / np.exp(r) * 0.9996, "volume": 100}).to_parquet(C.RAW / "bars_1min" / f"{sym}.parquet")

    n = 3000
    ts = pd.Timestamp("2025-10-01 06:00", tz=C.TZ) + pd.to_timedelta(rng.integers(0, 120 * 86400, n), unit="s")
    ev = pd.DataFrame({"id": np.arange(n), "ticker": rng.choice(TICKERS, n), "ts": ts.tz_convert("UTC"),
                       "headline": [f"headline {i} about something" for i in range(n)], "source": "benzinga", "n_tickers": 1})
    ev["event_id"] = ev["id"].astype(str) + "_" + ev["ticker"]
    ev["headline_window"] = np.where(ev["ts"].dt.tz_convert(C.TZ).dt.date.astype(str) <= C.INSAMPLE_END, "in", "oos")
    ev.to_parquet(C.INTERIM / "events.parquet", index=False)
    yield root
    mp.undo()


def test_full_pipeline(sandbox):
    from src import align, backtest, evaluate, figures, report

    align.main()
    ret = pd.read_parquet(C.INTERIM / "returns.parquet")
    prim = ret[ret["delay_s"] == C.ENTRY_DELAY_S].set_index("event_id")
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    rng = np.random.default_rng(11)
    signal = ev["event_id"].map(prim["xret_30m"]).fillna(0).to_numpy()
    z = signal / (np.nanstd(signal) or 1)
    cats = rng.choice(["earnings", "analyst_rating", "product", "other"], len(ev))
    pd.DataFrame({"event_id": ev["event_id"], "score": np.tanh(0.8 * z + rng.normal(0, 1, len(ev))),
                  "confidence": rng.uniform(0.3, 1, len(ev)), "category": cats, "echo_ok": True}
                 ).to_parquet(C.SCORES / "llm.parquet")
    for m, k in (("finbert", 0.2), ("embed", 0.0)):
        s = np.tanh(k * z + rng.normal(0, 1, len(ev)))
        pd.DataFrame({"event_id": ev["event_id"], "score": s, "confidence": np.abs(s)}).to_parquet(C.SCORES / f"{m}.parquet")
    pd.DataFrame([{"n_train": 100, "up_rate": 0.5, "best_C": 0.01, "cv_auc": 0.5}]).to_csv(C.TABLES / "embed_training.csv", index=False)
    (C.TABLES / "ingest_counts.json").write_text(json.dumps({
        "articles": 3000, "articles_in_universe": 3000, "ticker_headlines_raw": 3000, "dropped_near_duplicates": 0,
        "events": 3000, "events_insample": int((ev["headline_window"] == "in").sum()),
        "events_oos": int((ev["headline_window"] == "oos").sum()), "multi_ticker_share": 0.0}))
    pd.DataFrame([{"est_usd": 1.0, "calls": 3000}]).to_csv(C.REPORTS / "llm_cost_log.csv", index=False)
    (C.REPORTS / "notes_post_run.md").write_text("Synthetic run.")
    tpl = (C.ROOT / "reports" / "writeup_template.md").read_text()
    (C.REPORTS / "writeup_template.md").write_text(tpl)

    evaluate.main()
    ic = pd.read_csv(C.TABLES / "ic_by_horizon.csv").set_index(["model", "horizon"])["ic"]
    assert ic[("llm", "30m")] > 0.3
    assert abs(ic[("random", "30m")]) < 0.08

    backtest.main()
    s = pd.read_csv(C.TABLES / "backtest_summary.csv")
    p = s[(s["model"] == "llm") & (s["exit"] == "30m") & (s["delay_s"] == 60)].iloc[0]
    assert p["n_trades"] > 0 and p["gross_bps"] > 0
    assert (s["n_trades"] >= 0).all()

    figures.main()
    assert len(list(C.FIGURES.glob("*.png"))) == 6

    report.render(report.build_values())
    text = (C.REPORTS / "writeup.md").read_text()
    assert "{{" not in text and "n/a" not in text.split("# Limitations")[0]
