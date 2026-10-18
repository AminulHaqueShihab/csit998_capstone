"""Stage 12: descriptive lag analysis between daily sentiment and price moves.

Run: python -m src.lag_analysis
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config, evaluate_finbert, features, make_splits
from .config import PipelineError

OUT = config.ROOT / "results" / "lag_analysis"
START, END = pd.Timestamp("2021-01-10"), pd.Timestamp("2025-06-14")
LAGS = list(range(-5, 6))
N_PERM = 5000
MEASURES = ["vader_mean", "finbert_mean", "vader_net_share", "vader_surprise", "finbert_surprise",
            "log_article_count"]
TARGETS = ["ret", "abs_ret"]
SEED = 20261017


def build_daily(price: pd.DataFrame, vader: pd.DataFrame, finbert: pd.DataFrame) -> pd.DataFrame:
    close = price.set_index("date")["close"]
    v, f = vader.set_index("date"), finbert.set_index("date")
    df = pd.DataFrame({"close": close})
    df["ret"] = df["close"].pct_change()
    df["abs_ret"] = df["ret"].abs()
    df["vader_mean"] = v["vader_mean"]
    df["finbert_mean"] = f["finbert_mean"]
    n = v["article_count"].where(v["article_count"] > 0)
    df["vader_net_share"] = (v["vader_pos_count"] - v["vader_neg_count"]) / n
    df["log_article_count"] = np.log1p(v["article_count"])
    # causal surprise: today minus the mean of the previous 7 days (today excluded)
    for src, dst in (("vader_mean", "vader_surprise"), ("finbert_mean", "finbert_surprise")):
        df[dst] = df[src] - df[src].shift(1).rolling(7).mean()
    return df


def aligned(x: np.ndarray, y: np.ndarray, k: int):
    """Pairs (x_i, y_{i+k}): k > 0 means x (sentiment) comes BEFORE y (the price move)."""
    n = len(x)
    return (x[: n - k], y[k:]) if k >= 0 else (x[-k:], y[: n + k])


def spearman_perm(x, y, n_perm: int, rng):
    """Spearman rho, two-sided circular-shift permutation p-value, and the 95% null band."""
    rx, ry = rankdata(x), rankdata(y)
    m = len(rx)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
    rho = float((rxc * ryc).sum() / denom)
    shifts = rng.integers(20, m - 20, n_perm)
    idx = (shifts[:, None] + np.arange(m)[None, :]) % m
    null = (rxc[idx] * ryc).sum(axis=1) / denom
    p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (n_perm + 1))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return rho, p, float(lo), float(hi)


def lag_table(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target in TARGETS:
        for measure in MEASURES:
            for k in LAGS:
                x, y = aligned(df[measure].to_numpy(), df[target].to_numpy(), k)
                rho, p, lo, hi = spearman_perm(x, y, n_perm, rng)
                rows.append({"target": target, "measure": measure, "lag_k": k, "n": len(x), "rho": rho,
                             "p_perm": p, "null95_low": lo, "null95_high": hi})
    t = pd.DataFrame(rows)
    t["nominal_p05"] = t["p_perm"] < 0.05
    t["bonferroni_sig"] = t["p_perm"] < 0.05 / len(t)
    return t


def quintile_table(df: pd.DataFrame, measures=("vader_mean", "finbert_mean", "log_article_count")) -> pd.DataFrame:
    d = df.copy()
    d["next_ret"] = d["ret"].shift(-1)
    d = d.dropna(subset=["next_ret"])
    rows = []
    for m in measures:
        d["q"] = pd.qcut(d[m], 5, labels=False, duplicates="drop") + 1
        for q, g in d.groupby("q"):
            rows.append({"measure": m, "quintile": int(q), "n": len(g), "measure_mean": g[m].mean(),
                         "next_day_mean_return_pct": 100 * g["next_ret"].mean(),
                         "next_day_share_up": float((g["next_ret"] > 0).mean()),
                         "next_day_mean_abs_return_pct": 100 * g["next_ret"].abs().mean()})
    return pd.DataFrame(rows)


def yearly_table(df: pd.DataFrame, measures=("vader_mean", "finbert_mean")) -> pd.DataFrame:
    d = df.copy()
    d["next_ret"] = d["ret"].shift(-1)
    rows = []
    for year, g in d.groupby(d.index.year):
        for m in measures:
            for label, target in (("same_day_return", "ret"), ("next_day_return", "next_ret")):
                gg = g[[m, target]].dropna()
                rho = float(gg[m].corr(gg[target], method="spearman"))
                se = 1.0 / np.sqrt(len(gg) - 3)  # Fisher approximation, ignores autocorrelation
                z = np.arctanh(rho)
                rows.append({"year": int(year), "measure": m, "target": label, "n": len(gg), "rho": rho,
                             "approx_ci95_low": float(np.tanh(z - 1.96 * se)),
                             "approx_ci95_high": float(np.tanh(z + 1.96 * se))})
    return pd.DataFrame(rows)


def feature_screen(table: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    """Univariate association of each model feature with next-day direction (descriptive only)."""
    rng = np.random.default_rng(seed + 1)
    cols = (features.MODEL_PRICE_FEATURES
            + [c for c in features.MODEL_SENTIMENT_FEATURES if c != "no_news_flag"]
            + [c for c in features.FINBERT_SENTIMENT_FEATURES if c not in features.MODEL_SENTIMENT_FEATURES
               and c != "no_news_flag"])
    y = table["target_direction"].to_numpy()
    rows = []
    for c in cols:
        x = table[c].to_numpy(dtype=float)
        rho, p, _, _ = spearman_perm(x, y.astype(float), n_perm, rng)
        rows.append({"feature": c, "group": ("price" if c in features.MODEL_PRICE_FEATURES else
                                             "finbert" if c.startswith("finbert") else "vader/attention"),
                     "n": len(x), "spearman_rho_with_next_day_up": rho, "p_perm": p,
                     "auc": float(roc_auc_score(y, x))})
    out = pd.DataFrame(rows)
    out["nominal_p05"] = out["p_perm"] < 0.05
    out["bonferroni_sig"] = out["p_perm"] < 0.05 / len(out)
    return out


def heatmap(t: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    titles = {"ret": "Daily return of day t+k", "abs_ret": "Absolute return (volatility) of day t+k"}
    for ax, target in zip(axes, TARGETS):
        piv = t[t["target"] == target].pivot(index="measure", columns="lag_k", values="rho").loc[MEASURES]
        sig = t[t["target"] == target].pivot(index="measure", columns="lag_k", values="nominal_p05").loc[MEASURES]
        im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_xticks(range(len(LAGS)), [f"{k:+d}" for k in LAGS])
        ax.set_yticks(range(len(MEASURES)), MEASURES)
        for i in range(len(MEASURES)):
            for j in range(len(LAGS)):
                ax.text(j, i, f"{piv.iloc[i, j]:.2f}" + ("*" if sig.iloc[i, j] else ""), ha="center",
                        va="center", fontsize=7)
        ax.set_title(titles[target], fontsize=10)
        ax.set_xlabel("lag k (days)")
    axes[0].set_ylabel("sentiment / attention measure on day t")
    fig.colorbar(im, ax=axes, label="Spearman correlation", shrink=0.85)
    fig.suptitle("Lag analysis, 2021-01 to 2025-06 (* = nominal permutation p < 0.05, 132 tests; "
                 "descriptive, not causal)", fontsize=10)
    fig.text(0.45, -0.02, "k > 0: sentiment on day t comes BEFORE the move on day t+k (prediction); "
             "k < 0: sentiment comes AFTER the move (reaction)", ha="center", fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    for p, hint in ((config.PRICE_CLEAN, "prepare_price"), (config.DAILY_SENTIMENT, "build_sentiment"),
                    (config.DAILY_FINBERT, "build_finbert_features"), (config.MODELLING_TABLE, "build_modelling_table")):
        config.require_file(p, "lag_analysis", f"Run `python -m src.{hint}` first.")
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])
    vader = pd.read_csv(config.DAILY_SENTIMENT, parse_dates=["date"])
    finbert = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])
    df = build_daily(price, vader, finbert).loc[START:END]
    if df[MEASURES + TARGETS].isna().any().any():
        raise PipelineError("[lag_analysis] unexpected missing values in the analysis window")
    if (vader.set_index("date").loc[START:END, "no_news_flag"] != 0).any():
        raise PipelineError("[lag_analysis] analysis window contains days without news")

    OUT.mkdir(parents=True, exist_ok=True)
    lags = lag_table(df)
    lags.to_csv(OUT / "lag_crosscorrelation.csv", index=False)
    quint = quintile_table(df)
    quint.to_csv(OUT / "quintiles_next_day.csv", index=False)
    yearly = yearly_table(df)
    yearly.to_csv(OUT / "stability_by_year.csv", index=False)
    table = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    daily_fb = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])
    t = evaluate_finbert.add_finbert_inputs(make_splits.prepare_table(table), daily_fb)
    t = t[t["date"] <= END]
    screen = feature_screen(t)
    screen.to_csv(OUT / "feature_screening_next_day_direction.csv", index=False)
    (OUT / "figures").mkdir(exist_ok=True)
    heatmap(lags, OUT / "figures" / "lag_profile.png")

    k1 = lags[(lags["target"] == "ret") & (lags["lag_k"] == 1)].set_index("measure")
    k0 = lags[(lags["target"] == "ret") & (lags["lag_k"] == 0)].set_index("measure")
    km1 = lags[(lags["target"] == "ret") & (lags["lag_k"] == -1)].set_index("measure")
    summary = {
        "sample": {"start": str(START.date()), "end": str(END.date()), "days": len(df),
                   "note": "ends before the 17-day news gap; the test period (2025-10 onward) is not used"},
        "tests": {"n_lag_tests": len(lags), "nominal_p05": int(lags["nominal_p05"].sum()),
                  "expected_by_chance": round(0.05 * len(lags), 1),
                  "bonferroni_threshold": 0.05 / len(lags), "bonferroni_significant": lags[lags["bonferroni_sig"]][
                      ["target", "measure", "lag_k", "rho", "p_perm"]].round(4).to_dict("records")},
        "return_correlation_by_lag": {
            "same_day_k0": k0["rho"].round(3).to_dict(),
            "sentiment_after_move_k_minus1": km1["rho"].round(3).to_dict(),
            "sentiment_before_move_k_plus1_(prediction_target)": k1["rho"].round(3).to_dict(),
            "k_plus1_p_values": k1["p_perm"].round(3).to_dict()},
        "feature_screening": {"n_features": len(screen), "nominal_p05": int(screen["nominal_p05"].sum()),
                              "bonferroni_significant": screen[screen["bonferroni_sig"]]["feature"].tolist(),
                              "max_abs_auc_deviation_from_0.5": float((screen["auc"] - 0.5).abs().max())},
        "interpretation_rules": "Correlation is not causation. A same-day link can arise because headlines describe "
                                "the day's price move. Only k > 0 speaks to prediction.",
    }
    config.write_report("lag_analysis_summary.json", summary)
    print(f"[lag_analysis] {len(df)} days; {summary['tests']['nominal_p05']}/{len(lags)} lag tests nominal p<0.05 "
          f"(~{summary['tests']['expected_by_chance']} expected by chance); "
          f"Bonferroni-significant: {len(summary['tests']['bonferroni_significant'])}")
    print("same-day rho:", summary["return_correlation_by_lag"]["same_day_k0"])
    print("k=+1 rho   :", summary["return_correlation_by_lag"]["sentiment_before_move_k_plus1_(prediction_target)"])


if __name__ == "__main__":
    main()
