---
title: "Does LLM News Sentiment Predict Intraday Returns?"
subtitle: "An out-of-sample test on 30 large caps, with costs and latency"
author: "Eric Tao"
papersize: us-letter
fontsize: 9.5pt
margin:
  x: 0.8in
  y: 0.7in
---

# Question

Can a small, cheap LLM read a real-time equity headline and tell you which way the stock moves next? If it can, how long does the information last, and is anything left after paying to trade it?

I test this on Benzinga headlines for 30 high-news-volume S&P 500 stocks from {{sample_start}} to {{sample_end}}. Claude Haiku 4.5 scores each headline, and I compare it with FinBERT, a supervised embedding model and a random-sign control. The prompt and every fitted parameter were frozen using data through {{insample_end}}. **Every result below is out of sample: {{oos_events}} events over {{oos_days}} trading days.**

# Answers in brief

1. **Hypothesis and lookahead.** The hypothesis: headline sentiment predicts the stock's return relative to SPY after the headline. Each trade enters at the open of the first minute bar that starts at least 60 seconds after the headline timestamp, and every exit bar starts at or after the entry bar. Events are assigned to the in-sample or OOS window by the date the trade enters, not the date of the headline. Unit tests on synthetic bars check all three properties, including half days and overnight headlines.
2. **Decay.** Pooled over all OOS headlines, the LLM's rank IC is {{ic_llm_5m}} at 5 minutes, {{ic_llm_30m}} at 30 minutes and {{ic_llm_close}} at the close. Horizons whose 95% CI excludes zero: {{llm_sig_horizons}}. The pooled signal is too weak to trace a decay curve. On headlines that arrive during market hours, the 30-minute IC falls from {{icd_llm_0}} with no delay to {{icd_llm_60}} at 60 s, {{icd_llm_300}} at 5 minutes and {{icd_llm_900}} at 15 minutes. The point estimates fall steadily and {{latency_ic_phrase}}, but the intervals overlap heavily, so the decline is suggestive rather than established. If there is an edge, it belongs to whoever reads the headline within about a minute.
3. **LLM vs FinBERT.** At 30 minutes the LLM's IC exceeds FinBERT's by {{diff_30m}}, with a paired 95% CI of {{diff_30m_ci}}. {{llm_vs_finbert_verdict}} An IC ratio is {{ratio_30m}}.
4. **What killed net returns.** {{killer_phrase}} Latency did not: pooled gross edge is {{lat_gross_0}} bps with no delay and {{lat_gross_900}} bps at 15 minutes, because most LLM trades come from off-hours headlines that enter at the open regardless.
5. **Next.** Condition on the few places the signal showed up (regular-hours headlines, guidance news, a 5-minute exit) and test them on a fresh window; replace the spread proxy with quotes; add the consensus estimate to earnings headlines. Details in the last section.

# Data

- **News.** Alpaca News API (Benzinga feed): {{articles}} articles tagged to the universe. An article tagged to several universe tickers becomes one event per ticker, scored separately for each; {{multi_ticker_share}} of events come from multi-ticker articles. I drop a headline when the same ticker had one with token Jaccard similarity of 0.8 or more in the prior 10 minutes, which removed only {{dropped_near_duplicates}}; {{events}} events remain.
- **Prices.** Alpaca IEX minute bars and daily bars, split-adjusted, for the 30 stocks and SPY. The trading calendar comes from Alpaca, so half days close at 13:00.
- **Timestamps.** Stored in UTC, displayed in New York time.

# Method

**Entry and exit.** The decision time is the headline timestamp plus a delay (60 seconds primary). Entry is the open of the first regular-hours bar starting at or after the decision time. A headline outside 09:30 to 16:00 enters at the next session's first bar; {{oos_rth_share}} of OOS headlines arrived during regular hours. Intraday exits are the close of the last bar that ends by entry plus the horizon. A horizon that would run past the close is dropped, not truncated. Returns are measured net of SPY over the identical window.

**Scores.**

- *Claude Haiku 4.5* (the cheapest current Claude model): temperature 0, JSON-schema output (`ticker`, `sentiment` in [-1, 1], `confidence` in [0, 1], `category`), a frozen prompt with a scoring rubric, a 30-name reference table and 12 hand-written examples. Two prompt versions were tried on 50 in-sample headlines; v2 fixed two category rules and was frozen (`reports/prompt_log.md`). To cut cost, events go 20 per request through the Message Batches API, grouped at random across the year so no request holds a follow-up story next to the news it describes. On {{ag_n}} events scored both ways, batch and one-at-a-time scoring agree: sentiment correlation {{ag_pearson}}, sign agreement {{ag_sign}}, category agreement {{ag_cat}}, and trade-decision agreement {{ag_trade}}. Total API spend: ${{llm_spend_usd}}, of which ${{llm_spend_estimated}} is estimated for a one-at-a-time run I stopped before it logged usage.
- *FinBERT* (ProsusAI/finbert): P(positive) minus P(negative).
- *Embedding logit*: all-MiniLM-L6-v2 headline embeddings into an L2 logistic regression for the sign of the 30-minute excess return, trained on {{embed_n_train}} in-sample events with day-grouped cross-validation (CV AUC {{embed_cv_auc}}).
- *Random sign*: a coin flip per event.

**Evaluation.** IC is the Spearman correlation between score and forward excess return over OOS events. Confidence intervals resample whole trading days (2,000 replicates), because headlines on the same day share market shocks. Every replicate scores all models on the same days, which gives paired CIs for model differences.

**Backtest.** One position per qualifying headline, long if the score is positive and short if negative, each at 1/10 of capital, with at most 10 open. The LLM trades when |sentiment| is at least 0.5 and confidence at least 0.6, which selected {{llm_trade_rate}} of in-sample events. Each baseline gets the |score| cutoff that trades the same fraction of in-sample events. Costs per side are 5 bps plus half the spread, proxied by the median (high - low) / close over the 30 bars before entry. Sharpe is annualized from per-trade returns as specified; because positions overlap, I also report a daily-PnL Sharpe.

# Results

## Signal decay

![Rank IC by horizon, out of sample, with 95% day-block bootstrap CIs.](figures/ic_decay.png)

{{table_ic}}

Each cell is the rank IC with its 95% CI; {{ic_n_events}} events at 30 minutes. Haiku = Claude Haiku 4.5, Embed = MiniLM embedding logit, Random = random-sign control.

The embedding baseline is the only model whose pooled CI excludes zero at any horizon ({{embed_sig_phrase}}), despite a near-chance in-sample AUC. I have not established why; see "What didn't work".

## LLM versus FinBERT

{{table_diff}}

## Latency

![Latency on regular-hours headlines. Off-hours headlines enter at the next open whatever the delay, so they are excluded here.](figures/latency.png)

Rank IC at 30 minutes by entry delay, regular-hours headlines only (n = {{icd_n}} at 60 s):

{{table_ic_delay}}

## Backtest

![Cumulative return, 30-minute exit, 60-second delay. The dotted line is the LLM before costs.](figures/equity_curves.png)

Primary configuration (LLM, 30-minute exit, 60-second delay): {{bt_n_trades}} trades, hit rate {{bt_hit_rate}}, {{bt_gross_bps}} bps gross per trade (95% CI {{bt_gross_ci}}) and {{bt_net_bps}} bps net after {{bt_cost_bps}} bps of round-trip cost. Per-trade Sharpe is {{bt_sharpe_trade_gross}} gross and {{bt_sharpe_trade_net}} net; daily-PnL Sharpe is {{bt_sharpe_daily_net}} net. Max drawdown is {{bt_max_drawdown_net}} of capital on the additive equity curve that fixed 1/10-capital slots imply. Turnover is {{bt_turnover_per_day}}x capital per day. {{bt_rth_share}} of LLM trades come from regular-hours headlines; the rest enter at the open.

{{table_bt}}

Gross and net are bps per trade; SR is per-trade Sharpe, annualized; Max DD is on the additive equity curve. Daily-PnL Sharpe for every configuration is in `reports/tables/backtest_summary.csv`.

The one configuration with a gross edge that clears zero is the LLM at a 5-minute exit: {{bt5_gross}} bps per trade, CI {{bt5_gross_ci}}, over {{bt5_n}} trades. It still loses {{bt5_net}} bps net after {{bt5_cost}} bps of costs. It is also one of {{n_bt_configs}} exit and delay configurations I ran for the LLM, and the edge comes from headlines traded at the open: restricted to regular-hours headlines it is {{bt5_rth_gross}} bps, CI {{bt5_rth_gross_ci}} ({{bt5_rth_n}} trades).

![Per-trade edge from gross to net, LLM, 30-minute exit.](figures/cost_bridge.png)

Cost sensitivity, LLM at the 30-minute exit:

{{table_fee}}

## Which news moves prices

![IC at 30 minutes by the category the LLM assigned.](figures/ic_by_category.png)

{{table_cat}}

**{{biggest_move_cat}}** headlines carry the largest moves: {{biggest_move_bps}} bps mean absolute 30-minute excess move. The LLM's best category is **{{best_cat}}** (IC {{best_cat_ic}}, CI {{best_cat_ci}}, n = {{best_cat_n}}), and its worst is **{{worst_cat}}** ({{worst_cat_ic}}). This table holds {{n_cat_tests}} category-by-model tests, so one or two CIs that barely exclude zero are what chance alone would produce.

LLM backtest PnL by category (30-minute exit):

{{table_bt_cat}}

By arrival time, headlines during regular hours have LLM IC {{ic_sess_regular_hours}} {{ci_sess_regular_hours}} (n = {{n_sess_regular_hours}}); overnight headlines, measured from the next open, have {{ic_sess_overnight}} {{ci_sess_overnight}} (n = {{n_sess_overnight}}).

Does the LLM's own confidence sort the signal?

{{table_conf}}

# What didn't work

{{post_run_notes}}

# Limitations

- **Spread proxy.** Minute-bar high minus low mixes price movement with the spread, so it overstates the quoted spread of these megacaps, most of all in the volatile minutes after news. Costs here are conservative on spread and ignore market impact.
- **IEX bars.** IEX prints a few percent of consolidated volume. Some minutes have no bar, and bar prices can differ from the consolidated tape by a few cents.
- **Headline timestamps.** The Benzinga `created_at` stamp is the only arrival time available; a professional feed stamps arrival at the reader.
- **One year, 30 names.** {{oos_days}} OOS days and several hundred trades per configuration leave per-trade CIs of roughly plus or minus 14 bps. An edge of a few bps is invisible at this size.
- **Batch scoring.** Scores came from 20-headline requests, which agree with one-at-a-time scoring on {{ag_trade}} of trade decisions but not all of them.
- **Model knowledge.** Haiku 4.5's training data ends before the sample starts, so it cannot recall these outcomes, but it carries general priors about the companies.

# Next steps

1. **Pre-register the survivors.** Regular-hours headlines, guidance news and the 5-minute exit are the only places the LLM showed anything. Fix those rules now and test them on data from September 2026 onward, untouched by this study. The guidance result is the weakest of the three and may be one of the chance hits the category table warns about.
2. **Quote data.** Replace the high-low proxy with NBBO quotes, measure true spreads, and test mid-quote returns in the first minutes after a headline.
3. **Score the surprise.** Give the model the consensus estimate with each earnings and guidance headline, so it scores the surprise rather than the tone.
4. **More breadth.** Mid caps with thinner coverage, where news may be priced more slowly and a few-bps edge matters less relative to the move.

*Every number in this document is rendered by `make report` from files in `reports/tables/`; see `reports/results.json`.*
