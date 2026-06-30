"""
src/visualize.py
----------------
Produces publication-quality plots for the coastal flooding project.

Figures generated
─────────────────
1. time_series.png      - Observed water level + predicted flood probability
2. roc_curves.png       - ROC curves for all three models
3. pr_curves.png       - Precision-Recall curves
4. confusion_matrix.png - Ensemble confusion matrix heatmap
5. feature_importance.png - XGBoost top-20 features (gain)
6. shap_summary.png     - SHAP beeswarm (if shap installed)
7. flood_events.png     - Zoomed view of individual storm events
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

log = logging.getLogger(__name__)

PALETTE = {
    "water_level": "#1a8c8c",
    "flood_prob":  "#c89b3c",
    "flood_event": "#e05c5c",
    "lstm":        "#5b7fa6",
    "xgb":         "#c89b3c",
    "ensemble":    "#1a8c8c",
    "threshold":   "#888888",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       120,
})


def _save(fig, path: Path, name: str):
    dest = path / name
    fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved %s", dest)


def plot_time_series(df: pd.DataFrame, ensemble, out: Path):
    """Water level time series with flood probability overlay."""
    test_start = df.index[int(len(df) * 0.9)]
    subset = df[df.index >= test_start].copy()

    try:
        _, p_ens = ensemble.predict_proba_aligned(subset)
        idx = subset.index[-len(p_ens):]
    except Exception as e:
        log.warning("Could not generate predictions for time series: %s", e)
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Water level
    ax1.plot(subset.index, subset["water_level_m"],
             color=PALETTE["water_level"], linewidth=0.8, label="Observed water level")
    flood_mask = subset["flood"] == 1
    ax1.scatter(subset.index[flood_mask], subset["water_level_m"][flood_mask],
                color=PALETTE["flood_event"], s=8, zorder=5, label="Flood event")
    ax1.set_ylabel("Water Level (m above MHHW)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Coastal Flood Prediction — Test Period", fontsize=13, fontweight="bold")

    # Flood probability
    ax2.fill_between(idx, p_ens, alpha=0.4, color=PALETTE["flood_prob"])
    ax2.plot(idx, p_ens, color=PALETTE["flood_prob"], linewidth=0.9, label="Flood probability")
    thr = getattr(ensemble, "_threshold", 0.5)
    ax2.axhline(thr, color=PALETTE["threshold"], linestyle="--",
                linewidth=1.2, label=f"Threshold ({thr:.2f})")
    ax2.set_ylabel("P(Flood)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    _save(fig, out, "time_series.png")


def plot_roc_curves(df: pd.DataFrame, ensemble, metrics: dict, out: Path):
    """ROC curves for LSTM, XGBoost, and Ensemble."""
    test = df.iloc[int(len(df) * 0.9):]
    y_true = test["flood"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    model_map = {
        "LSTM":     (lambda: ensemble.lstm.predict_proba(df)[-len(y_true):], PALETTE["lstm"]),
        "XGBoost":  (lambda: ensemble.xgb.predict_proba(test),               PALETTE["xgb"]),
        "Ensemble": (lambda: ensemble.predict_proba(df)[-len(y_true):],       PALETTE["ensemble"]),
    }
    for name, (pred_fn, color) in model_map.items():
        try:
            proba = pred_fn()
            y_a   = y_true[-len(proba):]
            fpr, tpr, _ = roc_curve(y_a, proba)
            auc = metrics.get(name, {}).get("roc_auc", 0)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name}  (AUC={auc:.4f})")
        except Exception as e:
            log.warning("ROC curve failed for %s: %s", name, e)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Flood Prediction Models", fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out, "roc_curves.png")


def plot_pr_curves(df: pd.DataFrame, ensemble, metrics: dict, out: Path):
    """Precision-Recall curves."""
    test   = df.iloc[int(len(df) * 0.9):]
    y_true = test["flood"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    climo = float(y_true.mean())
    ax.axhline(climo, color="k", linestyle="--", linewidth=0.8,
               label=f"Baseline (P={climo:.3f})")

    model_map = {
        "XGBoost":  (lambda: ensemble.xgb.predict_proba(test),               PALETTE["xgb"]),
        "Ensemble": (lambda: ensemble.predict_proba(df)[-len(y_true):],       PALETTE["ensemble"]),
    }
    for name, (pred_fn, color) in model_map.items():
        try:
            proba = pred_fn()
            y_a   = y_true[-len(proba):]
            prec, rec, _ = precision_recall_curve(y_a, proba)
            ap = metrics.get(name, {}).get("pr_auc", 0)
            ax.plot(rec, prec, color=color, linewidth=2,
                    label=f"{name}  (AP={ap:.4f})")
        except Exception as e:
            log.warning("PR curve failed for %s: %s", name, e)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Flood Prediction", fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out, "pr_curves.png")


def plot_confusion_matrix(metrics: dict, out: Path):
    """Ensemble confusion matrix heatmap."""
    m = metrics.get("Ensemble", {})
    if not m:
        return

    cm = np.array([[m.get("tn", 0), m.get("fp", 0)],
                   [m.get("fn", 0), m.get("tp", 0)]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Flood", "Flood"])
    ax.set_yticklabels(["No Flood", "Flood"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Ensemble — Confusion Matrix", fontweight="bold")
    _save(fig, out, "confusion_matrix.png")


def plot_feature_importance(ensemble, out: Path, top_n: int = 20):
    """XGBoost top-N feature importances (gain)."""
    try:
        imp = ensemble.xgb.feature_importance().head(top_n)
    except Exception as e:
        log.warning("Feature importance unavailable: %s", e)
        return

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors = [PALETTE["ensemble"]] * top_n
    imp.sort_values().plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Gain")
    ax.set_title(f"XGBoost — Top {top_n} Features by Gain", fontweight="bold")
    _save(fig, out, "feature_importance.png")


def plot_shap_summary(ensemble, df: pd.DataFrame, out: Path):
    """SHAP beeswarm plot (requires shap package)."""
    try:
        import shap
        if ensemble.xgb.shap_values is None:
            return
        feature_cols = ensemble.xgb.feature_cols
        X_sample = df[feature_cols].fillna(0).sample(min(500, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(ensemble.xgb.shap_values, X_sample,
                          plot_type="dot", show=False, max_display=20)
        plt.title("SHAP Feature Importance — XGBoost", fontweight="bold")
        _save(fig, out, "shap_summary.png")
    except ImportError:
        log.info("shap not installed — skipping SHAP plot.")
    except Exception as e:
        log.warning("SHAP plot failed: %s", e)


def plot_results(df: pd.DataFrame, ensemble, metrics: dict, out: Path):
    """Generate all figures."""
    out = Path(out)
    log.info("Generating visualisations …")
    plot_time_series(df, ensemble, out)
    plot_roc_curves(df, ensemble, metrics, out)
    plot_pr_curves(df, ensemble, metrics, out)
    plot_confusion_matrix(metrics, out)
    plot_feature_importance(ensemble, out)
    plot_shap_summary(ensemble, df, out)
    log.info("All figures saved to %s/", out)
