"""Join each headline to its entry bar and compute forward returns.

Entry rule: the first regular-hours minute bar whose *start* is at or after (headline ts + delay).
The entry price is that bar's open, which is the first price printed at or after the decision time.
Headlines that land outside regular hours (or too late to fill before the close) enter at the next
session's first bar. Bar timestamps mark the bar start, so a bar stamped 10:01 covers [10:01, 10:02).

Exit for an intraday horizon h: close of the last bar that *ends* at or before entry_bar_ts + h.
Horizons that would run past the session close are left NaN rather than truncated.
'close' = last regular-hours bar of the entry session. 'next_close' = last bar of the following session.

Excess returns subtract SPY's return over the identical window.
Half-spread proxy: 0.5 x median((high - low) / close) over the 30 bars strictly before the entry bar.

Usage: python -m src.align
"""
import numpy as np
import pandas as pd

from src import config as C

NS_MIN = 60_000_000_000


def to_ns(ts):
    """Timestamps as int64 nanoseconds. pandas 3 defaults to microsecond resolution, so never astype directly."""
    return pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


class Bars:
    """Regular-hours minute bars for one symbol, as numpy arrays, with session bookkeeping."""

    def __init__(self, df, cal):
        opens = to_ns(cal["open"])
        closes = to_ns(cal["close"])
        t = to_ns(df["ts"])
        sess = np.searchsorted(closes, t, side="right")          # first session whose close > bar start
        ok = (sess < len(closes))
        ok[ok] &= t[ok] >= opens[sess[ok]]
        self.t, self.sess = t[ok], sess[ok]
        self.open = df["open"].to_numpy()[ok]
        self.close = df["close"].to_numpy()[ok]
        self.hl = ((df["high"] - df["low"]) / df["close"]).to_numpy()[ok]
        self.opens, self.closes = opens, closes
        n_sess = len(closes)
        self.last_bar = np.full(n_sess, -1)
        np.maximum.at(self.last_bar, self.sess, np.arange(len(self.t)))

    def in_session(self, t_ns):
        s = np.searchsorted(self.closes, t_ns, side="right")
        return bool(s < len(self.closes) and self.opens[s] <= t_ns)

    def entry(self, target_ns, max_stale_min=C.MAX_ENTRY_STALENESS_MIN):
        """Return (bar_index, session) of the entry bar for a decision time, or (-1, -1)."""
        s = np.searchsorted(self.closes, target_ns, side="right")
        if s >= len(self.closes):
            return -1, -1
        ref = max(target_ns, self.opens[s])
        i = np.searchsorted(self.t, ref, side="left")
        if i >= len(self.t):
            return -1, -1
        if self.sess[i] != s:          # nothing printed before this session closed: roll to next printed session
            s = self.sess[i]
            ref = self.opens[s]
        if self.t[i] - ref > max_stale_min * NS_MIN:
            return -1, -1
        return i, s

    def exit_index(self, i, s, horizon):
        """Index of the exit bar for horizon, or -1. Always >= i by construction."""
        if horizon in C.INTRADAY_HORIZONS:
            end = self.t[i] + C.INTRADAY_HORIZONS[horizon] * NS_MIN
            if end > self.closes[s]:
                return -1
            return np.searchsorted(self.t, end - NS_MIN, side="right") - 1
        if horizon == "close":
            return self.last_bar[s]
        if horizon == "next_close":
            return self.last_bar[s + 1] if s + 1 < len(self.last_bar) else -1
        raise ValueError(horizon)

    def half_spread_bps(self, i):
        lo = max(0, i - C.SPREAD_LOOKBACK_BARS)
        window = self.hl[lo:i]  # strictly before the entry bar
        if len(window) < 5:
            return np.nan
        return min(0.5 * np.median(window) * 1e4, C.SPREAD_CAP_BPS)


def _market_leg(mkt, entry_ns, s, horizon):
    """SPY return over the same window: open of first SPY bar at/after entry, close at the matching exit."""
    k = np.searchsorted(mkt.t, entry_ns, side="left")
    if k >= len(mkt.t) or mkt.sess[k] != s or mkt.t[k] - entry_ns > 5 * NS_MIN:
        return np.nan
    j = mkt.exit_index(k, s, horizon)
    if j < k:
        return np.nan
    return mkt.close[j] / mkt.open[k] - 1


def align_events(events, bars_by_symbol, cal, delays=C.LATENCY_GRID_S):
    """Pure function used by main() and by the tests. Returns one row per (event, delay)."""
    mkt = bars_by_symbol.get(C.MARKET)
    rows = []
    for ticker, g in events.groupby("ticker"):
        if ticker not in bars_by_symbol:
            continue
        b = bars_by_symbol[ticker]
        ts_ns = to_ns(g["ts"])
        for delay in delays:
            for eid, t0 in zip(g["event_id"], ts_ns):
                target = t0 + delay * 1_000_000_000
                i, s = b.entry(target)
                if i < 0:
                    continue
                r = {"event_id": eid, "delay_s": delay, "decision_ns": target,
                     "entry_ns": b.t[i], "entry_px": b.open[i],
                     "half_spread_bps": b.half_spread_bps(i),
                     "headline_in_rth": b.in_session(t0)}
                for h in C.HORIZONS:
                    j = b.exit_index(i, s, h)
                    if j < 0:
                        r[f"ret_{h}"] = r[f"xret_{h}"] = np.nan
                        r[f"exit_ns_{h}"] = -1
                        continue
                    ret = b.close[j] / b.open[i] - 1
                    r[f"ret_{h}"] = ret
                    r[f"exit_ns_{h}"] = b.t[j]
                    m = _market_leg(mkt, b.t[i], s, h) if mkt is not None else np.nan
                    r[f"xret_{h}"] = ret - m
                rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["entry_ts"] = pd.to_datetime(out["entry_ns"], unit="ns", utc=True)
    out["entry_date"] = out["entry_ts"].dt.tz_convert(C.TZ).dt.date
    return out


def load_bars(symbols, cal):
    out = {}
    for sym in symbols:
        p = C.RAW / "bars_1min" / f"{sym}.parquet"
        if p.exists():
            out[sym] = Bars(pd.read_parquet(p), cal)
    return out


def main():
    cal = pd.read_parquet(C.RAW / "calendar.parquet").sort_values("open").reset_index(drop=True)
    events = pd.read_parquet(C.INTERIM / "events.parquet")
    bars = load_bars(C.UNIVERSE + [C.MARKET], cal)
    ret = align_events(events, bars, cal)
    ret.to_parquet(C.INTERIM / "returns.parquet", index=False)
    prim = ret[ret["delay_s"] == C.ENTRY_DELAY_S]
    print(f"aligned {prim['event_id'].nunique():,} of {len(events):,} events at {C.ENTRY_DELAY_S}s delay; "
          f"{prim['headline_in_rth'].mean():.1%} of headlines arrived during regular hours")
    print(prim[[f"ret_{h}" for h in C.HORIZONS]].notna().sum().rename("non-null").to_string())


if __name__ == "__main__":
    main()
