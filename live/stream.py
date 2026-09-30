"""Live demo: stream Alpaca news, score each headline with the frozen prompt, paper-trade the same rule.

This is a demo of the pipeline running in real time. It is NOT a source of results; every reported number comes
from the historical backtest. Orders go to the Alpaca PAPER endpoint only (hard-coded; live URLs are refused).

Rule (same as the backtest): |sentiment| >= 0.5 and confidence >= 0.6, one position per headline, max 10 open,
1/10 of paper equity per position, exit after 30 minutes, flatten at 15:55 ET.

Log: live/logs/YYYY-MM-DD.jsonl, one line per scored (headline, ticker) with timing:
  created_at (Benzinga stamp) -> received_at (websocket) -> scored_at (LLM back) -> order_at

Usage:
  python -m live.stream --dry-run   # score and log, no orders
  python -m live.stream             # score, log, and paper trade until the close
"""
import argparse
import asyncio
import datetime as dt
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import websockets
from dotenv import load_dotenv

from src import config as C
from src.score_llm import make_client, request_kwargs, user_message

load_dotenv(C.ROOT / ".env")
NEWS_WS = "wss://stream.data.alpaca.markets/v1beta1/news"
PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
ET = ZoneInfo(C.TZ)
LOG_DIR = Path(os.getenv("LIVE_LOG_DIR", C.ROOT / "live" / "logs"))
HOLD = dt.timedelta(minutes=C.INTRADAY_HORIZONS[C.PRIMARY_EXIT])

if os.getenv("ALPACA_BASE_URL") and "paper-api" not in os.getenv("ALPACA_BASE_URL"):
    raise SystemExit("Refusing to run: ALPACA_BASE_URL is not the paper endpoint.")


def H():
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}


def now():
    return dt.datetime.now(dt.UTC)


class Book:
    """Paper positions opened by this process, keyed by symbol, with their scheduled exit time."""

    def __init__(self, dry):
        self.dry = dry
        self.open = {}
        acct = requests.get(f"{PAPER}/v2/account", headers=H(), timeout=10).json()
        self.slot_usd = float(acct["equity"]) / C.MAX_CONCURRENT
        print(f"paper equity ${float(acct['equity']):,.0f}; ${self.slot_usd:,.0f} per position")

    def market_open(self):
        return requests.get(f"{PAPER}/v2/clock", headers=H(), timeout=10).json()["is_open"]

    def last_price(self, sym):
        js = requests.get(f"{DATA}/v2/stocks/{sym}/trades/latest", headers=H(), params={"feed": "iex"}, timeout=10).json()
        return float(js["trade"]["p"])

    def enter(self, sym, side):
        if sym in self.open:
            return {"action": "skip", "reason": "already holding symbol"}
        if len(self.open) >= C.MAX_CONCURRENT:
            return {"action": "skip", "reason": "max concurrent positions"}
        if not self.market_open():
            return {"action": "skip", "reason": "market closed"}
        px = self.last_price(sym)
        qty = math.floor(self.slot_usd / px)
        if qty < 1:
            return {"action": "skip", "reason": "slot below one share"}
        order = {"symbol": sym, "qty": str(qty), "side": "buy" if side > 0 else "sell", "type": "market", "time_in_force": "day"}
        if self.dry:
            self.open[sym] = {"exit_at": now() + HOLD, "side": side}
            return {"action": "dry_run_order", **order, "ref_px": px}
        r = requests.post(f"{PAPER}/v2/orders", headers=H(), json=order, timeout=10)
        if r.status_code >= 300:
            return {"action": "order_rejected", "status": r.status_code, "detail": r.text[:200]}
        self.open[sym] = {"exit_at": now() + HOLD, "side": side, "order_id": r.json()["id"]}
        return {"action": "order_submitted", **order, "ref_px": px, "order_id": r.json()["id"]}

    def exit_due(self, flatten=False):
        closed = []
        for sym, pos in list(self.open.items()):
            if flatten or now() >= pos["exit_at"]:
                if not self.dry:
                    requests.delete(f"{PAPER}/v2/positions/{sym}", headers=H(), timeout=10)
                closed.append(sym)
                del self.open[sym]
        return closed


def log(rec):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{now().astimezone(ET).date()}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("ticker", "sentiment", "confidence", "category", "decision", "latency_s")}))


async def handle(item, client, book):
    received = now()
    tickers = [s for s in item.get("symbols", []) if s in C.UNIVERSE]
    created = dt.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    for t in tickers:
        others = [s for s in tickers if s != t]
        resp = await asyncio.to_thread(client.messages.parse, **request_kwargs(user_message(t, item["headline"], others)))
        s = resp.parsed_output
        scored = now()
        trade = abs(s.sentiment) >= C.SENT_THRESHOLD and s.confidence >= C.CONF_THRESHOLD
        decision = await asyncio.to_thread(book.enter, t, 1 if s.sentiment > 0 else -1) if trade else {"action": "no_trade"}
        log({"news_id": item["id"], "ticker": t, "headline": item["headline"], "created_at": created,
             "received_at": received, "scored_at": scored, "order_at": now() if trade else None,
             "sentiment": s.sentiment, "confidence": s.confidence, "category": s.category,
             "decision": decision["action"], "detail": decision,
             "latency_s": round((scored - created).total_seconds(), 2)})


async def exits(book, stop_at):
    while True:
        closing = now() >= stop_at
        for sym in await asyncio.to_thread(book.exit_due, closing):
            log({"ticker": sym, "decision": "exit", "flatten": closing, "at": now()})
        if closing:
            return
        await asyncio.sleep(10)


async def main(dry):
    client = make_client(max_retries=3)
    book = Book(dry)
    today = now().astimezone(ET).date()
    stop_at = dt.datetime.combine(today, dt.time(15, 55), ET).astimezone(dt.UTC)
    exit_task = asyncio.create_task(exits(book, stop_at))
    async with websockets.connect(NEWS_WS) as ws:
        await ws.send(json.dumps({"action": "auth", "key": os.environ["ALPACA_API_KEY"], "secret": os.environ["ALPACA_SECRET_KEY"]}))
        await ws.send(json.dumps({"action": "subscribe", "news": C.UNIVERSE}))
        print(f"subscribed to news for {len(C.UNIVERSE)} tickers; running until {stop_at.astimezone(ET):%H:%M} ET")
        while not exit_task.done():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                continue
            for msg in json.loads(raw):
                if msg.get("T") == "n":
                    asyncio.create_task(handle(msg, client, book))
                elif msg.get("T") in ("error", "subscription", "success"):
                    print(msg)
    await exit_task


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main(ap.parse_args().dry_run))
