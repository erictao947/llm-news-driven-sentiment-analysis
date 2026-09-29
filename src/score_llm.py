"""Score each (headline, ticker) event with Claude Haiku 4.5.

- temperature 0, structured JSON output validated by a Pydantic schema
- the frozen system prompt (prompts/llm_{PROMPT_VERSION}.md) is prompt-cached, so repeat input costs ~0.1x
- every response is appended to data/cache/llm_{PROMPT_VERSION}.jsonl keyed by event id; reruns skip cached keys
- token usage and estimated spend are appended to reports/llm_cost_log.csv per run

Usage:
  python -m src.score_llm --estimate                 # count tokens and project the cost, no scoring
  python -m src.score_llm --window in --spotcheck 50 # score 50 random in-sample events, write reports/llm_spotcheck.md
  python -m src.score_llm --window all --max-usd 150 # score everything, stop if spend passes the cap
  python -m src.score_llm --export                   # rebuild data/scores/llm.parquet from the cache only
"""
import argparse
import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from src import config as C

load_dotenv(C.ROOT / ".env")

SYSTEM = (C.ROOT / "prompts" / f"llm_{C.PROMPT_VERSION}.md").read_text()
CACHE_FILE = C.CACHE / f"llm_{C.PROMPT_VERSION}.jsonl"
COST_LOG = C.REPORTS / "llm_cost_log.csv"
HAIKU_CACHE_MIN_TOKENS = 4096

Category = Literal["earnings", "guidance", "analyst_rating", "product", "legal_regulatory",
                   "macro", "m_and_a", "management", "other"]


class Score(BaseModel):
    ticker: str
    sentiment: float
    confidence: float
    category: Category


def user_message(ticker, headline, others):
    return f"Ticker: {ticker}\nOther tickers tagged: {', '.join(others) if others else 'none'}\nHeadline: {headline}"


def request_kwargs(msg):
    return dict(
        model=C.LLM_MODEL,
        max_tokens=128,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": msg}],
        output_format=Score,
        extra_body={"temperature": 0},  # SDK 1.x dropped the kwarg; Haiku 4.5 still honors it
    )


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    out = {}
    with open(CACHE_FILE) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["event_id"]] = rec
    return out


def make_client(**kw):
    """Pin the public endpoint and the project key so an ambient ANTHROPIC_BASE_URL (e.g. a tool proxy) is ignored."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY missing. Copy .env.example to .env and fill it in.")
    return anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com", **kw)


def cost_usd(u):
    p = C.LLM_PRICE
    return (u["input"] * p["input"] + u["cache_write"] * p["cache_write"]
            + u["cache_read"] * p["cache_read"] + u["output"] * p["output"]) / 1e6


class Scorer:
    def __init__(self):
        self.client = make_client(max_retries=8)
        self.lock = threading.Lock()
        self.usage = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0, "errors": 0}

    def score_one(self, ev):
        msg = user_message(ev["ticker"], ev["headline"], ev["others"])
        resp = self.client.messages.parse(**request_kwargs(msg))
        u = resp.usage
        s = resp.parsed_output
        rec = {"event_id": ev["event_id"], "ticker": ev["ticker"],
               "sentiment": max(-1.0, min(1.0, s.sentiment)),
               "confidence": max(0.0, min(1.0, s.confidence)),
               "category": s.category, "echo_ok": s.ticker.upper() == ev["ticker"],
               "model": resp.model, "prompt_version": C.PROMPT_VERSION}
        with self.lock:
            self.usage["input"] += u.input_tokens
            self.usage["cache_write"] += u.cache_creation_input_tokens or 0
            self.usage["cache_read"] += u.cache_read_input_tokens or 0
            self.usage["output"] += u.output_tokens
            self.usage["calls"] += 1
            with open(CACHE_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def run(self, todo, workers=8, max_usd=None):
        if not todo:
            return
        self.score_one(todo[0])  # warm the prompt cache before fanning out
        if self.usage["cache_write"] + self.usage["cache_read"] == 0:
            print("WARNING: prompt was not cached; check the system prompt clears the model's minimum")
        done = 1
        with ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(self.score_one, ev) for ev in todo[1:]]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except anthropic.APIStatusError as e:
                    self.usage["errors"] += 1
                    print(f"API error {e.status_code}: {e.message[:120]}")
                done += 1
                if done % 1000 == 0:
                    print(f"  {done:,}/{len(todo):,}  spend so far ${cost_usd(self.usage):.2f}")
                if max_usd and cost_usd(self.usage) > max_usd:
                    print(f"Spend cap ${max_usd} reached, cancelling remaining requests")
                    for f in futs:
                        f.cancel()
                    break

    def log(self, window):
        row = {"run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "window": window,
               "model": C.LLM_MODEL, "prompt_version": C.PROMPT_VERSION, **self.usage,
               "est_usd": round(cost_usd(self.usage), 4)}
        pd.DataFrame([row]).to_csv(COST_LOG, mode="a", header=not COST_LOG.exists(), index=False)
        print(f"run usage: {self.usage}  est spend ${row['est_usd']:.2f}")


def load_events(window):
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    others = ev.groupby("id")["ticker"].agg(list)
    ev["others"] = [[t for t in others[i] if t != tk] for i, tk in zip(ev["id"], ev["ticker"])]
    if window != "all":
        ev = ev[ev["headline_window"] == window]
    return ev


def estimate(ev):
    client = make_client()
    n_sys = client.messages.count_tokens(model=C.LLM_MODEL, system=SYSTEM,
                                         messages=[{"role": "user", "content": "x"}]).input_tokens
    sample = ev.sample(min(200, len(ev)), random_state=C.SEED)
    per_msg = sum(len(user_message(r.ticker, r.headline, r.others)) for r in sample.itertuples()) / len(sample) / 3.5
    n = len(ev)
    u = {"input": n * per_msg, "cache_write": n_sys, "cache_read": (n - 1) * n_sys, "output": n * 45}
    print(f"system prompt: {n_sys:,} tokens (cache minimum for Haiku 4.5 is {HAIKU_CACHE_MIN_TOKENS:,}; "
          f"{'OK' if n_sys >= HAIKU_CACHE_MIN_TOKENS else 'TOO SHORT, WILL NOT CACHE'})")
    print(f"{n:,} events -> projected spend ${cost_usd(u):.2f} (assumes ~45 output tokens per call, full cache hits)")


def spotcheck(recs, ev, path=C.REPORTS / "llm_spotcheck.md"):
    df = ev.set_index("event_id").join(pd.DataFrame(recs).set_index("event_id")[["sentiment", "confidence", "category"]],
                                        how="inner")
    lines = ["# LLM spot check (in-sample)", "", f"Prompt version `{C.PROMPT_VERSION}`, {len(df)} random in-sample events.",
             "", "| Ticker | Headline | Sentiment | Conf | Category |", "|---|---|---|---|---|"]
    for eid, r in df.iterrows():
        lines.append(f"| {r['ticker']} | {r['headline'].replace('|', '/')} | {r['sentiment']:+.2f} | "
                     f"{r['confidence']:.2f} | {r['category']} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def export():
    cache = load_cache()
    df = pd.DataFrame(cache.values())
    df = df.rename(columns={"sentiment": "score"})[["event_id", "score", "confidence", "category", "echo_ok"]]
    df.to_parquet(C.SCORES / "llm.parquet", index=False)
    print(f"exported {len(df):,} LLM scores; {(~df['echo_ok']).sum()} ticker-echo mismatches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["in", "oos", "all"], default="all")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--spotcheck", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.export:
        return export()
    ev = load_events(a.window)
    if a.estimate:
        return estimate(ev)
    cache = load_cache()
    if a.spotcheck:
        ev = ev.sample(a.spotcheck, random_state=C.SEED)
    todo = [r for r in ev.to_dict("records") if r["event_id"] not in cache]
    print(f"{len(ev):,} events in window '{a.window}', {len(ev) - len(todo):,} cached, {len(todo):,} to score")
    s = Scorer()
    try:
        s.run(todo, workers=a.workers, max_usd=a.max_usd)
    finally:
        s.log(a.window)
    if a.spotcheck:
        spotcheck(list(load_cache().values()), ev)
    export()


if __name__ == "__main__":
    main()
