"""All report charts, written to reports/figures/. Reads only files produced by evaluate.py and backtest.py.

Usage: python -m src.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

COLOR = {"llm": "#2a78d6", "finbert": "#eb6834", "embed": "#1baf7a", "random": "#8a8983"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HORIZON_LABEL = {"5m": "5 min", "15m": "15 min", "30m": "30 min", "60m": "60 min",
                 "close": "Close", "next_close": "Next close"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 10,
})


def _style_line(m):
    return dict(color=COLOR[m], lw=2, ls="--" if m == "random" else "-", label=C.MODEL_LABELS[m])


def ic_decay():
    ic = pd.read_csv(C.TABLES / "ic_by_horizon.csv")
    x = np.arange(len(C.HORIZONS))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axhline(0, color=MUTED, lw=0.8)
    offsets = dict(zip(C.MODELS, np.linspace(-0.12, 0.12, len(C.MODELS))))
    for m in C.MODELS:
        d = ic[ic["model"] == m].set_index("horizon").loc[C.HORIZONS]
        xs = x + offsets[m]
        ax.plot(xs, d["ic"], marker="o", ms=5, **_style_line(m))
        ax.vlines(xs, d["ci_lo"], d["ci_hi"], color=COLOR[m], lw=1.2, alpha=0.7)
    ax.set_xticks(x, [HORIZON_LABEL[h] for h in C.HORIZONS])
    ax.set_ylabel("Rank IC vs SPY-excess return")
    ax.set_title("Signal decay: out-of-sample IC by horizon, 95% CI", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "ic_decay.png")
    plt.close(fig)


def equity():
    eq = pd.read_parquet(C.INTERIM / "equity_curves.parquet")
    eq = eq[(eq["exit"] == C.PRIMARY_EXIT) & (eq["delay_s"] == C.ENTRY_DELAY_S)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axhline(0, color=MUTED, lw=0.8)
    for m in C.MODELS:
        d = eq[eq["model"] == m].sort_values("date")
        dates = pd.to_datetime(d["date"])
        ax.plot(dates, d["net"].cumsum() * 100, **_style_line(m))
        if m == "llm":
            ax.plot(dates, d["gross"].cumsum() * 100, color=COLOR[m], lw=1.2, ls=":", label="Claude Haiku 4.5, gross")
    ax.set_ylabel("Cumulative return, % of capital")
    ax.set_title("Backtest equity, 30-min exit, 60 s entry delay, net of costs", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "equity_curves.png")
    plt.close(fig)


def latency():
    lat = pd.read_csv(C.TABLES / "backtest_latency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3))
    for ax, col, title in ((axes[0], "gross_bps", "Gross edge per trade (bps)"),
                           (axes[1], "net_bps", "Net edge per trade (bps)")):
        ax.axhline(0, color=MUTED, lw=0.8)
        for m in C.MODELS:
            d = lat[lat["model"] == m].sort_values("delay_s")
            ax.plot(d["delay_s"], d[col], marker="o", ms=5, **_style_line(m))
        ax.set_xscale("symlog", linthresh=60)
        ax.set_xticks(C.LATENCY_GRID_S, [f"{s}s" for s in C.LATENCY_GRID_S])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Entry delay after headline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(C.FIGURES / "latency.png")
    plt.close(fig)


def category():
    cat = pd.read_csv(C.TABLES / "ic_by_category.csv")
    cat = cat[cat["model"].isin(["llm", "finbert"])]
    order = (cat[cat["model"] == "llm"].sort_values("ic")["category"]).tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(order) + 1.0))
    ax.axvline(0, color=MUTED, lw=0.8)
    for k, m in enumerate(["llm", "finbert"]):
        d = cat[cat["model"] == m].set_index("category").reindex(order)
        yy = y + (0.18 if k == 0 else -0.18)
        ax.errorbar(d["ic"], yy, xerr=[d["ic"] - d["ci_lo"], d["ci_hi"] - d["ic"]], fmt="o", ms=5,
                    color=COLOR[m], elinewidth=1.2, capsize=0, label=C.MODEL_LABELS[m])
    n = cat[cat["model"] == "llm"].set_index("category")["n"]
    ax.set_yticks(y, [f"{c}  (n={n[c]:,})" for c in order])
    ax.set_xlabel("Rank IC at 30 min (SPY-excess)")
    ax.set_title("IC by LLM-assigned news category, out of sample", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "ic_by_category.png")
    plt.close(fig)


def cost_bridge():
    s = pd.read_csv(C.TABLES / "backtest_summary.csv")
    r = s[(s["model"] == "llm") & (s["exit"] == C.PRIMARY_EXIT) & (s["delay_s"] == C.ENTRY_DELAY_S)].iloc[0]
    fee = 2 * C.FEE_BPS_PER_SIDE
    spread = r["cost_bps"] - fee
    steps = [("Gross edge", r["gross_bps"]), ("Fees (2 x 5 bps)", -fee), ("Spread (2 x half)", -spread)]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    level = 0.0
    for i, (lab, v) in enumerate(steps):
        ax.bar(i, v, bottom=level, color=COLOR["llm"] if i == 0 else "#b8b7b0", width=0.6)
        ax.text(i, level + v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", color=INK, fontsize=8)
        level += v
    ax.bar(len(steps), level, color=COLOR["llm"] if level > 0 else "#e34948", width=0.6)
    ax.text(len(steps), level + (0.3 if level >= 0 else -0.3), f"{level:+.1f}", ha="center",
            va="bottom" if level >= 0 else "top", color=INK, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(steps) + 1), [l for l, _ in steps] + ["Net edge"], fontsize=8)
    ax.margins(y=0.18)
    ax.set_ylabel("bps per trade")
    ax.set_title("Where the LLM edge goes: 30-min exit, 60 s delay", loc="left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "cost_bridge.png")
    plt.close(fig)


def score_distribution():
    from src.evaluate import load_panel
    df = load_panel()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    axes[0].hist(df["score_llm"], bins=41, color=COLOR["llm"], edgecolor="white", linewidth=0.5)
    axes[0].set_title("LLM sentiment, OOS events", loc="left")
    axes[1].hist(df["score_finbert"], bins=41, color=COLOR["finbert"], edgecolor="white", linewidth=0.5)
    axes[1].set_title("FinBERT P(pos) - P(neg)", loc="left")
    for ax in axes:
        ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "score_distribution.png")
    plt.close(fig)


def main():
    for f in (ic_decay, equity, latency, category, cost_bridge, score_distribution):
        f()
        print(f"figure: {f.__name__}")
    (C.FIGURES / "manifest.json").write_text(json.dumps(sorted(p.name for p in C.FIGURES.glob("*.png")), indent=2))


if __name__ == "__main__":
    main()
