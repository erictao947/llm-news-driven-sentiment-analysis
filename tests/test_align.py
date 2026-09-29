"""No-lookahead tests for src.align on synthetic bars with gaps, a half day, and overnight headlines."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.align import Bars, align_events, to_ns

TZ = C.TZ


def _calendar():
    days = [("2026-01-05", "09:30", "16:00"), ("2026-01-06", "09:30", "13:00"), ("2026-01-07", "09:30", "16:00")]
    rows = []
    for d, o, c in days:
        rows.append({"date": pd.Timestamp(d).date(),
                     "open": pd.Timestamp(f"{d} {o}", tz=TZ).tz_convert("UTC"),
                     "close": pd.Timestamp(f"{d} {c}", tz=TZ).tz_convert("UTC")})
    return pd.DataFrame(rows)


def _bars(cal, seed, drop_every=7):
    """Minute bars 08:00-18:00 ET (extended hours included, align must filter them), with holes."""
    rng = np.random.default_rng(seed)
    ts = []
    for d in cal["date"]:
        ts.extend(pd.date_range(f"{d} 08:00", f"{d} 17:59", freq="1min", tz=TZ).tz_convert("UTC"))
    ts = pd.DatetimeIndex(ts)
    ts = ts[np.arange(len(ts)) % drop_every != 3]  # IEX-style missing minutes
    px = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(ts))))
    o = px * (1 + rng.normal(0, 1e-4, len(ts)))
    return pd.DataFrame({"ts": ts, "open": o, "close": px,
                         "high": np.maximum(o, px) * 1.0003, "low": np.minimum(o, px) * 0.9997,
                         "volume": 100})


@pytest.fixture(scope="module")
def aligned():
    cal = _calendar()
    raw = {"AAPL": _bars(cal, 1), C.MARKET: _bars(cal, 2, drop_every=11)}
    bars = {k: Bars(v, cal) for k, v in raw.items()}
    stamps = ["2026-01-05 07:15", "2026-01-05 09:29:30", "2026-01-05 10:00:00", "2026-01-05 10:00:59",
              "2026-01-05 15:10", "2026-01-05 15:58:50", "2026-01-05 19:00", "2026-01-06 12:59:30",
              "2026-01-06 14:00", "2026-01-07 11:03:17", "2026-01-07 15:59:59"]
    rng = np.random.default_rng(3)
    stamps += [str(pd.Timestamp("2026-01-05 06:00") + pd.Timedelta(seconds=int(s)))
               for s in rng.integers(0, 3 * 86400, 300)]
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(len(stamps))], "ticker": "AAPL",
                       "ts": pd.to_datetime(stamps, format="mixed").tz_localize(TZ).tz_convert("UTC")})
    out = align_events(ev, bars, cal)
    return ev, out, raw, bars, cal


def test_entry_never_before_decision_time(aligned):
    _, out, *_ = aligned
    assert len(out) > 0
    assert (out["entry_ns"] >= out["decision_ns"]).all()


def test_no_forward_return_uses_a_bar_before_the_entry_bar(aligned):
    _, out, *_ = aligned
    for h in C.HORIZONS:
        valid = out[f"exit_ns_{h}"] >= 0
        assert (out.loc[valid, f"exit_ns_{h}"] >= out.loc[valid, "entry_ns"]).all(), h
        assert out.loc[~valid, f"ret_{h}"].isna().all(), h


def test_intraday_exit_bar_ends_by_horizon(aligned):
    _, out, *_ = aligned
    for h, m in C.INTRADAY_HORIZONS.items():
        v = out[out[f"exit_ns_{h}"] >= 0]
        bar_end = v[f"exit_ns_{h}"] + 60 * 10**9
        assert (bar_end <= v["entry_ns"] + m * 60 * 10**9).all(), h


def test_entry_and_exit_bars_are_regular_hours_same_session(aligned):
    _, out, _, _, cal = aligned
    closes = to_ns(cal["close"])
    opens = to_ns(cal["open"])
    s_entry = np.searchsorted(closes, out["entry_ns"].to_numpy(), side="right")
    assert (out["entry_ns"].to_numpy() >= opens[s_entry]).all()
    for h in list(C.INTRADAY_HORIZONS) + ["close"]:
        v = out[f"exit_ns_{h}"].to_numpy()
        ok = v >= 0
        assert (np.searchsorted(closes, v[ok], side="right") == s_entry[ok]).all(), h


def test_returns_match_raw_prices(aligned):
    _, out, raw, *_ = aligned
    px = raw["AAPL"].set_index(to_ns(raw["AAPL"]["ts"]))
    for _, r in out.sample(50, random_state=0).iterrows():
        assert r["entry_px"] == pytest.approx(px.loc[r["entry_ns"], "open"])
        if r["exit_ns_30m"] >= 0:
            assert r["ret_30m"] == pytest.approx(px.loc[r["exit_ns_30m"], "close"] / r["entry_px"] - 1)


def test_overnight_headline_enters_at_next_open(aligned):
    ev, out, *_ = aligned
    prim = out[out["delay_s"] == 60].set_index("event_id")
    et = lambda eid: pd.Timestamp(prim.loc[eid, "entry_ns"], tz="UTC").tz_convert(TZ)
    assert et("e0") >= pd.Timestamp("2026-01-05 09:30", tz=TZ)            # pre-market
    assert et("e0") < pd.Timestamp("2026-01-05 09:35", tz=TZ)
    assert et("e6").date() == pd.Timestamp("2026-01-06").date()             # after close -> next day
    assert et("e7").date() == pd.Timestamp("2026-01-07").date()             # 12:59:30 + 60s on a half day
    assert not prim.loc["e6", "headline_in_rth"] and prim.loc["e2", "headline_in_rth"]


def test_mid_session_entry_is_first_bar_after_delay(aligned):
    _, out, *_ = aligned
    prim = out[out["delay_s"] == 60].set_index("event_id")
    # 10:00:59 + 60 s = 10:01:59, so the 10:01 bar (opens 10:01:00) is too early; must be >= 10:02
    assert pd.Timestamp(prim.loc["e3", "entry_ns"], tz="UTC").tz_convert(TZ) >= pd.Timestamp("2026-01-05 10:02", tz=TZ)


def test_spread_window_is_strictly_before_entry(aligned):
    *_, bars, _ = aligned
    b = bars["AAPL"]
    i = 200
    before = b.half_spread_bps(i)
    b.hl[i:] = 10.0          # poison the entry bar and everything after it
    assert b.half_spread_bps(i) == before


def test_latency_is_monotone(aligned):
    _, out, *_ = aligned
    w = out.pivot(index="event_id", columns="delay_s", values="entry_ns").dropna()
    for a, b in zip(C.LATENCY_GRID_S[:-1], C.LATENCY_GRID_S[1:]):
        assert (w[b] >= w[a]).all()


def test_sample_split_uses_entry_date_not_headline_date():
    """A headline after the Nov 28 2025 half-day close enters Dec 1 (OOS) and must be tagged OOS."""
    from src.align import assign_sample
    days = [("2025-11-26", "16:00"), ("2025-11-28", "13:00"), ("2025-12-01", "16:00")]
    cal = pd.DataFrame([{"date": pd.Timestamp(d).date(),
                         "open": pd.Timestamp(f"{d} 09:30", tz=TZ).tz_convert("UTC"),
                         "close": pd.Timestamp(f"{d} {c}", tz=TZ).tz_convert("UTC")} for d, c in days])
    bars = {"AAPL": Bars(_bars(cal, 5), cal)}
    ev = pd.DataFrame({"event_id": ["fri_morning", "fri_after_close", "sat"], "ticker": "AAPL",
                       "ts": pd.to_datetime(["2025-11-28 10:00", "2025-11-28 14:30", "2025-11-29 12:00"])
                       .tz_localize(TZ).tz_convert("UTC")})
    out = align_events(ev, bars, cal)
    out["sample"] = assign_sample(out)
    tag = out.drop_duplicates("event_id").set_index("event_id")["sample"]
    assert tag["fri_morning"] == "in"
    assert tag["fri_after_close"] == "oos" and tag["sat"] == "oos"
    # the tag is per event, identical across latency variants
    assert out.groupby("event_id")["sample"].nunique().max() == 1
