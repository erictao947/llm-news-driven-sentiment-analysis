---
title: "Does LLM News Sentiment Predict Intraday Returns?"
subtitle: "An out-of-sample test on 30 large caps, with costs and latency"
author: "Eric Tao"
papersize: us-letter
fontsize: 10pt
margin:
  x: 0.8in
  y: 0.7in
---

# Question

Can a small, cheap LLM read a real-time equity headline and tell you which way the stock moves next? If it can, how long does the information last, and is anything left after paying to trade it?

I test this on Benzinga headlines for 30 high-news-volume S&P 500 stocks from {{sample_start}} to {{sample_end}}. Claude Haiku 4.5 scores each headline. I compare it against FinBERT, a supervised embedding model, and a random-sign control. The prompt and every fitted parameter were frozen on {{sample_start}} to {{insample_end}}. **Every result below is out of sample: {{oos_start}} to {{sample_end}}, {{oos_events}} events over {{oos_days}} trading days.**

# Answers in brief

1. **Hypothesis and lookahead.** Headline sentiment predicts the stock's return relative to SPY after the headline. Each trade enters at the open of the first minute bar that starts at least 60 seconds after the headline timestamp. Every exit bar starts at or after the entry bar. A unit test on synthetic bars proves both properties, along with the half-day and overnight handling.
2. **Decay.** The LLM's rank IC is {{ic_llm_5m}} at 5 minutes, {{ic_llm_30m}} at 30 minutes and {{ic_llm_close}} by the close. It peaks at {{llm_peak_h}} ({{llm_peak_ic}}) and {{llm_half_life_phrase}}. Horizons where the 95% CI excludes zero: {{llm_sig_horizons}}.
3. **LLM vs FinBERT.** At 30 minutes the IC difference is {{diff_30m}}, 95% CI {{diff_30m_ci}}; LLM-to-FinBERT ratio: {{ratio_30m}}. {{llm_vs_finbert_verdict}}
4. **What killed net returns.** {{killer_phrase}} {{latency_death_phrase}}
5. **Next.** Tick-level quotes for true spreads, more names, event-specific exits, and a cheaper execution venue. Details in the last section.

# Data

- **News.** Alpaca News API (Benzinga feed): {{articles}} articles tagged to the universe. An article tagged to several universe tickers becomes one event per ticker and is scored separately for each ({{multi_ticker_share}} of events come from multi-ticker articles). I drop a headline when the same ticker had a headline with token Jaccard similarity of 0.8 or more within the prior 10 minutes: {{dropped_near_duplicates}} near-duplicates removed, {{events}} events kept ({{events_insample}} in sample, {{events_oos}} out of sample).
- **Prices.** Alpaca IEX minute bars and daily bars, split-adjusted, for the 30 stocks and SPY. The trading calendar comes from Alpaca, so half days close at 13:00.
- **Timestamps.** Stored in UTC. Displayed in New York time.

# Method

**Entry and exit.** The decision time is the headline timestamp plus a delay (60 seconds primary). Entry is the open of the first regular-hours bar starting at or after the decision time. A headline outside 09:30 to 16:00 enters at the next session's first bar, with the same horizons measured from there; {{oos_rth_share}} of OOS headlines arrived during regular hours. Intraday exits are the close of the last bar that ends by entry + horizon. A horizon that would run past the close is dropped, not truncated. "Close" is the entry session's last bar; "next close" is the following session's last bar. Returns are measured net of SPY over the identical window.

**Scores.**

- *Claude Haiku 4.5*: temperature 0, structured JSON output (`ticker`, `sentiment` in [-1, 1], `confidence` in [0, 1], `category`), a frozen few-shot prompt with 12 hand-written examples, and a universe reference table. Responses are cached to disk by event id. Total API spend: ${{llm_spend_usd}} over {{llm_calls}} calls.
- *FinBERT* (ProsusAI/finbert): P(positive) minus P(negative).
- *Embedding logit*: all-MiniLM-L6-v2 headline embeddings into an L2 logistic regression predicting the sign of the 30-minute excess return. It is trained on {{embed_n_train}} in-sample events with day-grouped cross-validation (CV AUC {{embed_cv_auc}}).
- *Random sign*: a coin flip per event.

**Sample split.** An event belongs to the in-sample or OOS window by the date its trade enters, not the date of the headline. A Friday-evening headline on the last in-sample weekend enters on the first OOS session, so it counts as OOS.

**Evaluation.** The IC is the Spearman correlation between score and forward excess return, pooled over OOS events. Confidence intervals come from a block bootstrap that resamples trading days (2,000 replicates). Every replicate scores all models on the same days, which gives paired CIs for model differences.

**Backtest.** One position per qualifying headline: long if the score is positive, short if negative, each at 1/10 of capital, with at most 10 open positions. The LLM trades when |sentiment| is at least 0.5 and confidence at least 0.6, which selects {{llm_trade_rate}} of in-sample events. Each baseline gets the |score| cutoff that trades the same fraction of in-sample events, so every model has the same trade budget. Costs per side are 5 bps plus half the spread, with the spread proxied by the median (high - low) / close over the 30 minute bars before entry. Sharpe is annualized from per-trade returns as the brief specifies. Because positions overlap, I also report a daily-PnL Sharpe.

# Results

## Signal decay

![IC by horizon with 95% day-block bootstrap CIs.](figures/ic_decay.png)

{{table_ic}}

Cells show the IC with its 95% CI. Haiku = Claude Haiku 4.5, Embed = MiniLM embedding logit, Random = random-sign control.

## LLM versus FinBERT

{{table_diff}}

{{llm_vs_finbert_verdict}}

## Backtest

![Cumulative net return. The dotted line is the LLM before costs.](figures/equity_curves.png)

Primary configuration (LLM, 30-minute exit, 60-second delay): {{bt_n_trades}} trades, hit rate {{bt_hit_rate}}, {{bt_gross_bps}} bps gross and {{bt_net_bps}} bps net per trade after {{bt_cost_bps}} bps of round-trip cost. Per-trade Sharpe is {{bt_sharpe_trade_gross}} gross and {{bt_sharpe_trade_net}} net; daily-PnL Sharpe is {{bt_sharpe_daily_net}} net. Max drawdown is {{bt_max_drawdown_net}} of capital, measured on the additive (non-compounded) equity curve that fixed 1/10-capital slots imply, and turnover is {{bt_turnover_per_day}}x capital per day.

{{table_bt}}

![Per-trade edge, from gross to net.](figures/cost_bridge.png)

## Latency

![Edge per trade by entry delay.](figures/latency.png)

{{table_latency}}

The LLM earns {{lat_gross_0}} bps gross with no delay, {{lat_gross_60}} at 60 seconds, {{lat_gross_300}} at 5 minutes and {{lat_gross_900}} at 15 minutes. {{latency_gross_phrase}} {{latency_death_phrase}}

Cost sensitivity for the LLM at the 30-minute exit:

{{table_fee}}

## Which news moves prices

![IC at 30 minutes by the category the LLM assigned.](figures/ic_by_category.png)

{{table_cat}}

The LLM's strongest category is **{{best_cat}}** (IC {{best_cat_ic}}, CI {{best_cat_ci}}, n = {{best_cat_n}}), and its weakest is **{{worst_cat}}** ({{worst_cat_ic}}). The largest average 30-minute moves follow **{{biggest_move_cat}}** headlines ({{biggest_move_bps}} bps mean absolute excess move).

Backtest PnL by category (LLM, 30-minute exit):

{{table_bt_cat}}

Headlines arriving during regular hours have IC {{ic_sess_regular_hours}} {{ci_sess_regular_hours}} (n = {{n_sess_regular_hours}}). Overnight headlines, measured from the next open, have IC {{ic_sess_overnight}} {{ci_sess_overnight}} (n = {{n_sess_overnight}}). A headline that arrived overnight has already had the whole opening auction to be priced in.

Is the LLM's confidence informative?

{{table_conf}}

# What didn't work

{{post_run_notes}}

# Limitations

- **Spread proxy.** Minute-bar high minus low includes price movement as well as the spread, so it overstates the true quoted spread, probably most in volatile minutes right after news. Net results are conservative on spread and optimistic on market impact, which the model ignores.
- **IEX bars.** IEX prints a few percent of consolidated volume. Some minutes have no bar, and bar prices can differ from the consolidated tape by a few cents.
- **Headline timestamps.** The Benzinga `created_at` stamp is the only arrival time available. A professional feed would stamp arrival at the reader, which can differ by seconds.
- **One year, 30 names.** With about {{oos_days}} OOS days, a daily-level effect needs to be large to show up. Category-level CIs are wide.
- **Model knowledge.** Haiku 4.5's training data ends before the sample begins, so it cannot recall these outcomes. It still carries general priors about the companies.

# Next steps

1. **Quote data.** Replace the high-low proxy with NBBO quotes to measure true spreads and to test mid-quote returns in the first minutes.
2. **Condition the trade.** Trade only the categories and sessions where the OOS IC is reliably positive, then re-validate on a fresh window rather than this one.
3. **Faster entry.** The latency table prices speed directly. The websocket demo in `live/` shows end-to-end scoring latency on real headlines.
4. **Surprise, not tone.** Feed the model the consensus estimate alongside earnings headlines so it can score the surprise directly.
5. **Breadth.** Extend to the Russell 1000, where coverage is thinner and news may be priced more slowly.

*Every number in this document is rendered by `make report` from files in `reports/tables/`; see `reports/results.json`.*
