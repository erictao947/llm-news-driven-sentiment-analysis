"""Cost-efficient LLM scoring: 20 events per request through the Message Batches API (50% price).

Same model (Claude Haiku 4.5, the cheapest current Claude model), same frozen prompt, temperature 0, JSON-schema
output. Two changes versus src/score_llm.py, which scores one event per call:
  1. Each request carries GROUP_SIZE events, so the ~5k-token system prompt is paid once per 20 events
     instead of once per event (prompt re-reads were ~70% of single-call cost).
  2. Requests go through the Batches API, which bills all tokens at 50%.

Lookahead guard: events are assigned to groups at random across the whole year (fixed seed), and the model is told
to score each item as if it saw it alone. Chronological grouping could put a follow-up headline ("shares slump
after...") in the same request as the news it describes.

Validation: --validate scores the events already scored one-per-call and writes the agreement table
reports/tables/llm_batch_vs_single.csv.

Usage:
  python -m src.score_llm_batch --validate            # batch-score the single-call set, compare
  python -m src.score_llm_batch --max-usd 15          # score everything not yet in the batch cache
  python -m src.score_llm_batch --resume <batch_id>   # collect results of a submitted batch
"""
import argparse
import datetime as dt
import json
import time

import anthropic
import numpy as np
import pandas as pd

from src import config as C
from src.score_llm import COST_LOG, SYSTEM, load_events, make_client, user_message
from src.score_llm import load_cache as load_single_cache

GROUP_SIZE = 20
CACHE_FILE = C.CACHE / f"llm_{C.PROMPT_VERSION}_batch.jsonl"
BATCH_LOG = C.CACHE / "batches.jsonl"
CATEGORIES = ["earnings", "guidance", "analyst_rating", "product", "legal_regulatory",
              "macro", "m_and_a", "management", "other"]
SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "ticker": {"type": "string"},
                       "sentiment": {"type": "number"}, "confidence": {"type": "number"},
                       "category": {"type": "string", "enum": CATEGORIES}},
        "required": ["item", "ticker", "sentiment", "confidence", "category"],
        "additionalProperties": False}}},
    "required": ["results"],
    "additionalProperties": False,
}
PREAMBLE = ("Score each numbered item independently, exactly as if it were the only headline you had seen. "
            "Never use one item to inform another. Return one result per item with its item number, in order.")


def group_message(rows):
    parts = [PREAMBLE]
    for i, r in enumerate(rows, 1):
        parts.append(f"[{i}]\n{user_message(r['ticker'], r['headline'], r['others'])}")
    return "\n\n".join(parts)


def params(rows):
    return {
        "model": C.LLM_MODEL,
        "max_tokens": 80 * len(rows) + 200,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": group_message(rows)}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        "temperature": 0,  # forwarded as-is in batch params; Haiku 4.5 honors it
    }


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    with open(CACHE_FILE) as f:
        return {r["event_id"]: r for r in map(json.loads, f)}


def batch_cost(u):
    p = C.LLM_PRICE
    return 0.5 * (u["input"] * p["input"] + u["cache_write"] * p["cache_write"]
                  + u["cache_read"] * p["cache_read"] + u["output"] * p["output"]) / 1e6


def projected_usd(n_events):
    n_req = int(np.ceil(n_events / GROUP_SIZE))
    sys_tokens = 4980
    u = {"input": n_req * sys_tokens + n_events * 55, "cache_write": 0, "cache_read": 0, "output": n_events * 40}
    return batch_cost(u)  # worst case: no prompt-cache hits inside the batch


def submit(todo, tag):
    rng = np.random.default_rng(C.SEED)
    todo = [todo[i] for i in rng.permutation(len(todo))]
    groups = [todo[i:i + GROUP_SIZE] for i in range(0, len(todo), GROUP_SIZE)]
    reqs, members = [], {}
    for g, rows in enumerate(groups):
        cid = f"{tag}-{g:05d}"
        reqs.append({"custom_id": cid, "params": params(rows)})
        members[cid] = [r["event_id"] for r in rows] + [r["ticker"] for r in rows]
    client = make_client()
    b = client.messages.batches.create(requests=reqs)
    with open(BATCH_LOG, "a") as f:
        f.write(json.dumps({"batch_id": b.id, "tag": tag, "submitted_at": dt.datetime.now(dt.UTC).isoformat(),
                            "n_requests": len(reqs), "n_events": len(todo), "members": members}) + "\n")
    print(f"submitted batch {b.id}: {len(reqs):,} requests, {len(todo):,} events")
    return b.id


def collect(batch_id, poll_s=30):
    client = make_client()
    info = next(r for r in map(json.loads, open(BATCH_LOG)) if r["batch_id"] == batch_id)
    while True:
        try:
            b = client.messages.batches.retrieve(batch_id)
        except anthropic.APIConnectionError as e:  # includes timeouts; the batch keeps running server-side
            print(f"  poll failed ({type(e).__name__}), retrying")
            time.sleep(poll_s)
            continue
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"  {b.processing_status}: {c.succeeded} ok, {c.processing} processing, {c.errored} errored")
        time.sleep(poll_s)
    usage = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0, "errors": 0}
    missing, n_ok = 0, 0
    with open(CACHE_FILE, "a") as f:
        for res in client.messages.batches.results(batch_id):
            mem = info["members"][res.custom_id]
            k = len(mem) // 2
            ids, tickers = mem[:k], mem[k:]
            if res.result.type != "succeeded":
                usage["errors"] += 1
                missing += k
                continue
            msg = res.result.message
            u = msg.usage
            usage["input"] += u.input_tokens
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["output"] += u.output_tokens
            usage["calls"] += 1
            try:
                out = json.loads(next(b.text for b in msg.content if b.type == "text"))["results"]
            except (StopIteration, json.JSONDecodeError, KeyError):
                missing += k
                continue
            got = {}
            for r in out:
                i = r.get("item", 0) - 1
                if 0 <= i < k and i not in got:
                    got[i] = r
            for i in range(k):
                r = got.get(i)
                if r is None:
                    missing += 1
                    continue
                f.write(json.dumps({"event_id": ids[i], "ticker": tickers[i],
                                    "sentiment": max(-1.0, min(1.0, r["sentiment"])),
                                    "confidence": max(0.0, min(1.0, r["confidence"])),
                                    "category": r["category"], "echo_ok": r["ticker"].upper() == tickers[i],
                                    "model": msg.model, "prompt_version": C.PROMPT_VERSION,
                                    "mode": f"batch{GROUP_SIZE}", "batch_id": batch_id}) + "\n")
                n_ok += 1
    row = {"run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "window": info["tag"],
           "model": C.LLM_MODEL, "prompt_version": f"{C.PROMPT_VERSION}-batch{GROUP_SIZE}", **usage,
           "est_usd": round(batch_cost(usage), 4)}
    pd.DataFrame([row]).to_csv(COST_LOG, mode="a", header=not COST_LOG.exists(), index=False)
    print(f"collected {n_ok:,} scores, {missing:,} missing (rerun to retry them); spend ${row['est_usd']:.2f}")


def validate():
    from src.evaluate import spearman
    single, batch = load_single_cache(), load_cache()
    both = sorted(set(single) & set(batch))
    s = pd.DataFrame([single[k] for k in both]).set_index("event_id")
    b = pd.DataFrame([batch[k] for k in both]).set_index("event_id")
    trade = lambda d: (d["sentiment"].abs() >= C.SENT_THRESHOLD) & (d["confidence"] >= C.CONF_THRESHOLD)
    row = {"n": len(both),
           "sentiment_spearman": spearman(s["sentiment"], b["sentiment"]),
           "sentiment_pearson": float(np.corrcoef(s["sentiment"], b["sentiment"])[0, 1]),
           "confidence_spearman": spearman(s["confidence"], b["confidence"]),
           "category_agreement": float((s["category"] == b["category"]).mean()),
           "sign_agreement_nonzero": float((np.sign(s["sentiment"]) == np.sign(b["sentiment"]))[
               (s["sentiment"] != 0) & (b["sentiment"] != 0)].mean()),
           "mean_abs_sentiment_diff": float((s["sentiment"] - b["sentiment"]).abs().mean()),
           "trade_flag_agreement": float((trade(s) == trade(b)).mean()),
           "trade_rate_single": float(trade(s).mean()), "trade_rate_batch": float(trade(b).mean())}
    pd.DataFrame([row]).to_csv(C.TABLES / "llm_batch_vs_single.csv", index=False)
    print(json.dumps(row, indent=2))


def export():
    df = pd.DataFrame(load_cache().values()).drop_duplicates("event_id", keep="last")
    df = df.rename(columns={"sentiment": "score"})[["event_id", "score", "confidence", "category", "echo_ok"]]
    df.to_parquet(C.SCORES / "llm.parquet", index=False)
    print(f"exported {len(df):,} LLM scores (batch mode); {(~df['echo_ok']).sum()} ticker-echo mismatches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--max-usd", type=float, default=15.0)
    a = ap.parse_args()
    if a.resume:
        collect(a.resume)
        return export()
    ev = load_events("all")
    done = load_cache()
    if a.validate:
        keys = set(load_single_cache())
        todo = [r for r in ev.to_dict("records") if r["event_id"] in keys and r["event_id"] not in done]
        tag = "validate"
    else:
        todo = [r for r in ev.to_dict("records") if r["event_id"] not in done]
        tag = "full"
    proj = projected_usd(len(todo))
    print(f"{len(todo):,} events to score; worst-case projected spend ${proj:.2f} (cap ${a.max_usd:.2f})")
    if proj > a.max_usd:
        raise SystemExit("projected spend exceeds cap; nothing submitted")
    if todo:
        collect(submit(todo, tag))
    if a.validate:
        validate()
    else:
        export()


if __name__ == "__main__":
    main()
