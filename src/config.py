"""Single source of truth for study parameters. Change nothing here after the in-sample freeze."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
SCORES = DATA / "scores"
CACHE = DATA / "cache"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "NFLX",
    "JPM", "BAC", "GS", "MS", "WFC", "XOM", "CVX", "PFE", "JNJ", "UNH",
    "LLY", "WMT", "COST", "HD", "NKE", "DIS", "BA", "CAT", "KO", "PEP",
]
MARKET = "SPY"  # benchmark for excess returns; never traded, never scored

START = "2025-09-01"
END = "2026-08-31"          # inclusive, last headline date
END_BARS = "2026-09-04"     # bars run a few sessions past END so next-day-close horizons resolve
INSAMPLE_END = "2025-11-30"  # prompt design and all fitted parameters use data up to here only
OOS_START = "2025-12-01"

TZ = "America/New_York"

# Forward-return horizons. Intraday ones are minutes after the entry bar open.
INTRADAY_HORIZONS = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
HORIZONS = list(INTRADAY_HORIZONS) + ["close", "next_close"]

ENTRY_DELAY_S = 60                   # primary: first bar open at/after headline + 60 s
LATENCY_GRID_S = [0, 60, 300, 900]   # latency sensitivity
MAX_ENTRY_STALENESS_MIN = 10         # drop an event if no bar prints within 10 min of the target entry time

DEDUP_WINDOW_MIN = 10
DEDUP_JACCARD = 0.8

# Trading rule
SENT_THRESHOLD = 0.5
CONF_THRESHOLD = 0.6
MAX_CONCURRENT = 10
FEE_BPS_PER_SIDE = 5.0
SPREAD_LOOKBACK_BARS = 30
SPREAD_CAP_BPS = 50.0
PRIMARY_EXIT = "30m"
EXIT_SENSITIVITY = ["5m", "30m", "close"]

BOOTSTRAP_REPS = 2000
SEED = 20250901

LLM_MODEL = "claude-haiku-4-5"
# USD per million tokens (Haiku 4.5 list price). Cache write = 1.25x, cache read = 0.1x input.
LLM_PRICE = {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10}
PROMPT_VERSION = "v2"

MODELS = ["llm", "finbert", "embed", "random"]
MODEL_LABELS = {"llm": "Claude Haiku 4.5", "finbert": "FinBERT",
                "embed": "MiniLM + logit", "random": "Random sign"}
MODEL_SHORT = {"llm": "Haiku", "finbert": "FinBERT", "embed": "Embed", "random": "Random"}

for p in (RAW, INTERIM, SCORES, CACHE, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
