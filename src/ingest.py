"""Pull historical news, minute bars, daily bars and the trading calendar from Alpaca.

Outputs (all timestamps UTC):
  data/raw/news.parquet            one row per article
  data/raw/bars_1min/{SYM}.parquet  IEX minute bars incl. extended hours
  data/raw/bars_1day.parquet       IEX daily bars
  data/raw/calendar.parquet        session open/close per trading date (handles half days)
  data/interim/events.parquet      one row per (article, universe ticker), deduplicated

Usage: python -m src.ingest [--skip-news] [--skip-bars]
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src import config as C

DATA_URL = "https://data.alpaca.markets"
load_dotenv(C.ROOT / ".env")


def _headers():
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing. Copy .env.example to .env and fill it in.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(url, params, retries=6):
    for attempt in range(retries):
        r = requests.get(url, headers=_headers(), params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _month_chunks(start, end):
    edges = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1), freq="MS", tz="UTC")
    edges = [pd.Timestamp(start, tz="UTC")] + [e for e in edges if e > pd.Timestamp(start, tz="UTC")]
    last = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    if edges[-1] < last:
        edges.append(last)
    return list(zip(edges[:-1], edges[1:]))


# ---------------------------------------------------------------- news
def pull_news():
    rows = []
    for lo, hi in _month_chunks(C.START, C.END):
        token, n0 = None, len(rows)
        while True:
            params = {"symbols": ",".join(C.UNIVERSE), "start": lo.isoformat(), "end": hi.isoformat(),
                      "limit": 50, "sort": "asc", "include_content": "false"}
            if token:
                params["page_token"] = token
            js = _get(f"{DATA_URL}/v1beta1/news", params)
            rows.extend(js.get("news", []))
            token = js.get("next_page_token")
            if not token:
                break
        print(f"news {lo.date()}..{hi.date()}: {len(rows) - n0}")
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df = df.drop_duplicates("id").sort_values("created_at")
    keep = ["id", "created_at", "updated_at", "headline", "summary", "author", "source", "url", "symbols"]
    df[keep].to_parquet(C.RAW / "news.parquet", index=False)
    return df


# ---------------------------------------------------------------- bars
def _pull_bars(symbol, timeframe):
    out = []
    for lo, hi in _month_chunks(C.START, C.END_BARS):
        token = None
        while True:
            params = {"symbols": symbol, "timeframe": timeframe, "start": lo.isoformat(), "end": hi.isoformat(),
                      "limit": 10000, "feed": "iex", "adjustment": "all", "sort": "asc"}
            if token:
                params["page_token"] = token
            js = _get(f"{DATA_URL}/v2/stocks/bars", params)
            out.extend(js.get("bars", {}).get(symbol, []))
            token = js.get("next_page_token")
            if not token:
                break
    df = pd.DataFrame(out).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                                            "c": "close", "v": "volume", "n": "trades", "vw": "vwap"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def pull_bars():
    (C.RAW / "bars_1min").mkdir(exist_ok=True)
    daily = []
    for sym in C.UNIVERSE + [C.MARKET]:
        m = _pull_bars(sym, "1Min")
        m.to_parquet(C.RAW / "bars_1min" / f"{sym}.parquet", index=False)
        d = _pull_bars(sym, "1Day").assign(symbol=sym)
        daily.append(d)
        print(f"bars {sym}: {len(m):,} minute, {len(d)} daily")
    pd.concat(daily).to_parquet(C.RAW / "bars_1day.parquet", index=False)


def pull_calendar():
    js = requests.get("https://paper-api.alpaca.markets/v2/calendar", headers=_headers(),
                      params={"start": C.START, "end": C.END_BARS}, timeout=60).json()
    cal = pd.DataFrame(js)
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    for col in ("open", "close"):
        local = pd.to_datetime(cal["date"].astype(str) + " " + cal[col]).dt.tz_localize(C.TZ)
        cal[col] = local.dt.tz_convert("UTC")
    cal[["date", "open", "close"]].to_parquet(C.RAW / "calendar.parquet", index=False)
    print(f"calendar: {len(cal)} sessions, {(cal['close'].dt.tz_convert(C.TZ).dt.hour < 16).sum()} half days")
    return cal


# ---------------------------------------------------------------- events + dedup
_TOKEN = re.compile(r"[a-z0-9$%.]+")


def tokens(text):
    return frozenset(_TOKEN.findall(str(text).lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedup(events, window_min=C.DEDUP_WINDOW_MIN, thresh=C.DEDUP_JACCARD):
    """Drop a headline if the same ticker had a kept headline within `window_min` minutes with token Jaccard >= thresh."""
    keep = []
    window = pd.Timedelta(minutes=window_min)
    for _, g in events.sort_values("ts").groupby("ticker", sort=False):
        recent = []  # (ts, tokens) of kept rows
        for idx, ts, head in zip(g.index, g["ts"], g["headline"]):
            tok = tokens(head)
            recent = [(t, k) for t, k in recent if ts - t <= window]
            if any(jaccard(tok, k) >= thresh for _, k in recent):
                continue
            recent.append((ts, tok))
            keep.append(idx)
    return events.loc[sorted(keep)]


def build_events(news=None):
    news = pd.read_parquet(C.RAW / "news.parquet") if news is None else news
    ev = news.explode("symbols").rename(columns={"symbols": "ticker", "created_at": "ts"})
    ev = ev[ev["ticker"].isin(C.UNIVERSE)]
    ev = ev[ev["headline"].str.strip().str.len() > 0]
    ev["n_tickers"] = ev.groupby("id")["ticker"].transform("size")
    ev["event_id"] = ev["id"].astype(str) + "_" + ev["ticker"]
    n_raw = len(ev)
    ev = dedup(ev.reset_index(drop=True))
    ev = ev[["event_id", "id", "ticker", "ts", "headline", "source", "n_tickers"]].sort_values("ts")
    ev["sample"] = (ev["ts"].dt.tz_convert(C.TZ).dt.date.astype(str) <= C.INSAMPLE_END).map({True: "in", False: "oos"})
    ev.to_parquet(C.INTERIM / "events.parquet", index=False)
    counts = {"articles": int(news["id"].nunique()),
              "articles_in_universe": int(ev["id"].nunique()),
              "ticker_headlines_raw": int(n_raw), "dropped_near_duplicates": int(n_raw - len(ev)),
              "events": int(len(ev)), "events_insample": int((ev["sample"] == "in").sum()),
              "events_oos": int((ev["sample"] == "oos").sum()),
              "multi_ticker_share": float((ev["n_tickers"] > 1).mean())}
    (C.TABLES / "ingest_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"events: {n_raw:,} ticker-headlines, {n_raw - len(ev):,} dropped as near-duplicates, {len(ev):,} kept "
          f"({(ev['sample'] == 'in').sum():,} in-sample, {(ev['sample'] == 'oos').sum():,} OOS)")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-bars", action="store_true")
    a = ap.parse_args()
    pull_calendar()
    if not a.skip_news:
        news = pull_news()
        print(f"news total: {len(news):,} articles")
    if not a.skip_bars:
        pull_bars()
    build_events()


if __name__ == "__main__":
    main()
