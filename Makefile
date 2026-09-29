PY := .venv/bin/python
LLM_MAX_USD ?= 150

.PHONY: setup data score score-llm-estimate spotcheck eval backtest report test dashboard live live-dry all

setup:
	uv venv --python 3.13 .venv && uv pip install --python $(PY) -r requirements.txt

data:                     ## pull news + bars + calendar, dedupe, align entry bars and forward returns
	$(PY) -m src.ingest
	$(PY) -m src.align

score-llm-estimate:       ## count prompt tokens and project the LLM bill; no scoring
	$(PY) -m src.score_llm --estimate

spotcheck:                ## score 50 random in-sample headlines and write reports/llm_spotcheck.md
	$(PY) -m src.score_llm --window in --spotcheck 50

score:                    ## all three models; the LLM stops at LLM_MAX_USD
	$(PY) -m src.score_llm --window all --max-usd $(LLM_MAX_USD)
	$(PY) -m src.score_finbert
	$(PY) -m src.score_embed

eval:
	$(PY) -m src.evaluate

backtest:
	$(PY) -m src.backtest

report:                   ## figures + writeup.md + writeup.pdf, every number from reports/tables
	$(PY) -m src.figures
	$(PY) -m src.report

test:
	$(PY) -m pytest tests -q

dashboard:
	.venv/bin/streamlit run dashboard/app.py

live-dry:
	$(PY) -m live.stream --dry-run

live:
	$(PY) -m live.stream

all: data score eval backtest report
