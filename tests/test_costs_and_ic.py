"""Cost model, position limits, IC and bootstrap checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import config as C
from src.backtest import simulate, summarize, trade_cost
from src.evaluate import ci, day_bootstrap, hit_rate, spearman
from src.ingest import dedup


# ---------------------------------------------------------------- costs
def test_round_trip_cost_is_two_sides_of_fee_plus_half_spread():
    assert trade_cost(np.array([3.0]), fee_bps=5.0)[0] == pytest.approx(2 * (5 + 3) / 1e4)


def test_zero_fee_zero_spread_is_free():
    assert trade_cost(np.array([0.0]), fee_bps=0.0)[0] == 0.0


def test_missing_spread_filled_with_median():
    c = trade_cost(np.array([2.0, np.nan, 4.0]), fee_bps=0.0)
    assert c[1] == pytest.approx(2 * 3.0 / 1e4)


def test_net_equals_gross_minus_cost_in_summary():
    days = [pd.Timestamp("2026-01-05").date(), pd.Timestamp("2026-01-06").date()]
    t = pd.DataFrame({"entry_date": [days[0], days[0], days[1]], "gross": [0.01, -0.004, 0.002],
                      "cost": [0.0012] * 3})
    t["net"] = t["gross"] - t["cost"]
    s, daily = summarize(t, days, {"model": "x"})
    assert s["net_bps"] == pytest.approx(s["gross_bps"] - 12.0)
    assert daily["net"].sum() == pytest.approx(t["net"].sum() / C.MAX_CONCURRENT)
    assert s["turnover_per_day"] == pytest.approx(2 * 3 / C.MAX_CONCURRENT / 2)


# ---------------------------------------------------------------- position limit
def test_max_concurrent_positions_enforced():
    n = 25
    cands = pd.DataFrame({"entry_ns": np.arange(n), "exit_ns": np.arange(n) + 100})
    acc = simulate(cands, max_concurrent=10)
    assert acc.sum() == 10 and acc[:10].all()


def test_slot_frees_when_position_exits():
    cands = pd.DataFrame({"entry_ns": [0, 1, 5, 6], "exit_ns": [5, 100, 100, 100]})
    assert simulate(cands, max_concurrent=2).tolist() == [True, True, True, False]


# ---------------------------------------------------------------- IC
def test_spearman_matches_scipy_with_ties_and_nans():
    rng = np.random.default_rng(0)
    x = rng.integers(-3, 4, 500).astype(float)
    y = 0.3 * x + rng.normal(size=500)
    x[::17] = np.nan
    ok = ~np.isnan(x)
    assert spearman(x, y) == pytest.approx(spearmanr(x[ok], y[ok]).statistic)


def test_ic_sign_and_bounds():
    x = np.arange(100.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_hit_rate_ignores_zeros():
    assert hit_rate([1, -1, 0, 1], [0.1, 0.2, 0.3, 0.0]) == pytest.approx(0.5)


def test_day_bootstrap_ci_covers_truth_and_widens_with_day_clustering():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(60), 50)
    shock = rng.normal(size=60)[days]            # common daily shock hits score and return
    s = rng.normal(size=len(days)) + shock
    r = 0.05 * s + rng.normal(size=len(days)) + 2 * shock
    df = pd.DataFrame({"s": s, "r": r, "entry_date": days})
    b = day_bootstrap(df, {"ic": lambda f: spearman(f["s"], f["r"])}, reps=300)["ic"]
    lo, hi = ci(b)
    assert lo < spearman(s, r) < hi
    # event-level bootstrap (every event its own day) gives a narrower band when days are clustered
    df2 = df.assign(entry_date=np.arange(len(df)))
    b2 = day_bootstrap(df2, {"ic": lambda f: spearman(f["s"], f["r"])}, reps=300)["ic"]
    assert np.std(b) > np.std(b2)


# ---------------------------------------------------------------- dedup
def test_dedup_drops_near_duplicates_within_window_only():
    t0 = pd.Timestamp("2026-01-05 14:00", tz="UTC")
    ev = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "AAPL"],
        "ts": [t0, t0 + pd.Timedelta(minutes=3), t0 + pd.Timedelta(minutes=30),
               t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=4)],
        "headline": ["Apple Q1 EPS $2.10 Beats $1.98 Estimate, Sales $124B Beat",
                     "Apple Q1 EPS $2.10 Beats $1.98 Estimate, Sales $124B Beat",   # dup within 10 min
                     "Apple Q1 EPS $2.10 Beats $1.98 Estimate, Sales $124B Beat",   # same text, 30 min later
                     "Apple Q1 EPS $2.10 Beats $1.98 Estimate, Sales $124B Beat",   # different ticker
                     "Apple shares slide after iPhone supply warning"],
    })
    kept = dedup(ev)
    assert list(kept.index) == [0, 2, 3, 4]
