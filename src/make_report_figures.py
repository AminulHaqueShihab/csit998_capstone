"""Draw the report figures from saved results (no training).

Run: python -m src.make_report_figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIG = config.ROOT / "report" / "figures"
RESULTS = config.ROOT / "results"
COL = {"A": "#1f5f99", "BV": "#d9822b", "BF": "#2a9d8f", "naive": "#6b7280", "ok": "#2e7d32", "bad": "#c62828",
       "truth": "#111827"}
LABEL = {"A": "LSTM A (price only)", "BV": "LSTM B (price + VADER)", "BF": "LSTM B (price + FinBERT)",
         "naive": "Majority rule"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "none", "axes.titlesize": 10, "axes.titleweight": "bold"})


def rolling_accuracy(y, p, window):
    hit = ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).astype(float)
    return pd.Series(hit).rolling(window).mean().to_numpy()


def predicted_vs_actual():
    d = pd.read_csv(RESULTS / "finbert_test_predictions.csv", parse_dates=["date"])
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"]).set_index("date")["close"].reindex(d["date"])
    cols = {"A": "proba_lstm_a_price_only", "BV": "proba_lstm_b_price_plus_sentiment", "BF": "proba_lstm_b_finbert",
            "naive": "proba_naive_majority"}
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1, 1]})

    a = ax[0]
    a.plot(d["date"], price.values / 1000, color=COL["truth"], lw=1.3)
    hit = (d[cols["A"]] >= 0.5).astype(int) == d["y_true"]
    a.scatter(d["date"][hit], price.values[hit] / 1000, s=9, color=COL["ok"], zorder=3, label="LSTM A called next day correctly")
    a.scatter(d["date"][~hit], price.values[~hit] / 1000, s=14, marker="x", color=COL["bad"], zorder=3, linewidths=0.9,
              label="LSTM A called next day wrongly")
    a.set_ylabel("BTCUSDT close (thousand USDT)")
    a.set_title("(a) Actual price, with the right and wrong daily calls of the price-only LSTM")
    a.legend(loc="upper right", fontsize=8)

    b = ax[1]
    truth7 = d["y_true"].rolling(7, min_periods=7).mean()
    b.fill_between(d["date"], 0, 1, where=d["y_true"] == 1, color=COL["ok"], alpha=0.06, step="mid", linewidth=0)
    b.plot(d["date"], truth7, color=COL["truth"], lw=1.6, label="Actual: share of up days, trailing 7 days")
    b.plot(d["date"], d[cols["A"]].rolling(7, min_periods=7).mean(), color=COL["A"], lw=1.6, label="LSTM A: predicted P(up), 7-day mean")
    b.plot(d["date"], d[cols["BV"]].rolling(7, min_periods=7).mean(), color=COL["BV"], lw=1.6, label="LSTM B (VADER): predicted P(up), 7-day mean")
    b.axhline(0.5, color=COL["naive"], ls="--", lw=0.9)
    b.set_ylim(0, 1)
    b.set_ylabel("Probability / share of days")
    b.set_title("(b) Predicted probability of a rise against what actually happened")
    b.legend(loc="lower right", fontsize=8, ncol=1)

    c = ax[2]
    for key in ("A", "BV", "BF", "naive"):
        c.plot(d["date"], rolling_accuracy(d["y_true"], d[cols[key]], 30), color=COL[key], lw=1.5,
               ls="--" if key == "naive" else "-", label=LABEL[key])
    c.axhline(0.5, color="black", lw=0.8, ls=":")
    c.set_ylim(0.25, 0.75)
    c.set_ylabel("Accuracy, trailing 30 days")
    c.set_title("(c) Rolling accuracy of each model")
    c.legend(loc="lower left", fontsize=8, ncol=2)
    c.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    c.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.tight_layout()
    fig.savefig(FIG / "predicted_vs_actual.pdf")
    plt.close(fig)


def walkforward_rolling():
    d = pd.read_csv(RESULTS / "walk_forward" / "walk_forward_predictions.csv", parse_dates=["date"])
    cols = {"A": "proba_lstm_a_price_only", "BV": "proba_lstm_b_vader", "BF": "proba_lstm_b_finbert",
            "naive": "proba_naive_majority"}
    folds = json.load(open(RESULTS / "walk_forward" / "walk_forward_results.json"))["design"]["folds"]
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    a = ax[0]
    for key in ("A", "BV", "BF", "naive"):
        a.plot(d["date"], rolling_accuracy(d["y_true"], d[cols[key]], 90), color=COL[key], lw=1.5,
               ls="--" if key == "naive" else "-", label=LABEL[key])
    a.axhline(0.5, color="black", lw=0.8, ls=":")
    a.set_ylim(0.3, 0.7)
    a.set_ylabel("Accuracy, trailing 90 days")
    a.set_title("(a) Rolling accuracy across the eight walk-forward periods")
    a.legend(loc="lower left", fontsize=8, ncol=2)
    b = ax[1]
    b.plot(d["date"], d["y_true"].rolling(90).mean(), color=COL["truth"], lw=1.7, label="Actual share of up days")
    b.plot(d["date"], (d[cols["A"]] >= 0.5).astype(int).rolling(90).mean(), color=COL["A"], lw=1.5, label="LSTM A predicted up")
    b.plot(d["date"], (d[cols["BV"]] >= 0.5).astype(int).rolling(90).mean(), color=COL["BV"], lw=1.5, label="LSTM B (VADER) predicted up")
    b.axhline(0.5, color="black", lw=0.8, ls=":")
    b.set_ylim(0.1, 0.9)
    b.set_ylabel("Share of days, trailing 90 days")
    b.set_title("(b) How often each model calls a rise, against how often one happened")
    b.legend(loc="lower left", fontsize=8, ncol=1)
    for s, _ in folds[1:]:
        for x in ax:
            x.axvline(pd.Timestamp(s), color="grey", lw=0.6, ls=":")
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    b.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.tight_layout()
    fig.savefig(FIG / "walkforward_rolling.pdf")
    plt.close(fig)


def walkforward_folds():
    r = json.load(open(RESULTS / "walk_forward" / "walk_forward_results.json"))
    keys = [("naive_majority", "naive"), ("lstm_a_price_only", "A"), ("lstm_b_vader", "BV"), ("lstm_b_finbert", "BF")]
    labels = [f"{f['test_start'][:7]}\nto {f['test_end'][:7]}" for f in r["folds"]]
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, (k, c) in enumerate(keys):
        ax.bar(x + (i - 1.5) * w, [f["models"][k]["accuracy_ensemble"] * 100 for f in r["folds"]], w, color=COL[c], label=LABEL[c])
    ax.axhline(50, color="black", lw=0.8, ls=":")
    ax.set_ylim(35, 65)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Accuracy of averaged predictions (%)")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(FIG / "walkforward_folds.pdf")
    plt.close(fig)


def lag_heatmap():
    t = pd.read_csv(RESULTS / "lag_analysis" / "lag_crosscorrelation.csv")
    t = t[t["target"] == "ret"]
    order = ["vader_mean", "finbert_mean", "vader_net_share", "vader_surprise", "finbert_surprise", "log_article_count"]
    names = ["VADER mean", "FinBERT mean", "VADER net share", "VADER surprise", "FinBERT surprise", "Article count (log)"]
    lags = sorted(t["lag_k"].unique())
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    grid = np.array([[t[(t.measure == m) & (t.lag_k == k)]["rho"].iloc[0] for k in lags] for m in order])
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i, m in enumerate(order):
        for j, k in enumerate(lags):
            row = t[(t.measure == m) & (t.lag_k == k)].iloc[0]
            ax.text(j, i, f"{row.rho:.2f}" + ("*" if row.nominal_p05 else ""), ha="center", va="center", fontsize=7,
                    fontweight="bold" if row.bonferroni_sig else "normal", color="white" if abs(row.rho) > 0.3 else "black")
    ax.set_xticks(range(len(lags)), [f"{k:+d}" if k else "0" for k in lags])
    ax.set_yticks(range(len(order)), names)
    ax.set_xlabel("Lag k in days: negative means sentiment follows the price move, positive means it comes first")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman correlation with the day's return")
    fig.tight_layout()
    fig.savefig(FIG / "lag_heatmap.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (predicted_vs_actual, walkforward_rolling, walkforward_folds, lag_heatmap):
        fn()
        print(f"[make_report_figures] {fn.__name__} done")


if __name__ == "__main__":
    main()
