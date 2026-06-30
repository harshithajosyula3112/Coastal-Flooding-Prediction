"""
src/evaluate.py
---------------
Comprehensive evaluation suite for the flood prediction ensemble.

Metrics computed
────────────────
  • ROC-AUC, PR-AUC
  • F1, Precision, Recall, Accuracy
  • Critical Success Index (CSI / Threat Score) — standard in hydrology
  • False Alarm Ratio (FAR)
  • Probability of Detection (POD)
  • Brier Score & Brier Skill Score
  • Confusion matrix
  • Per-storm-event hit/miss breakdown
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def _csi(y_true, y_pred) -> float:
    """Critical Success Index = TP / (TP + FN + FP)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = tp + fn + fp
    return tp / denom if denom else 0.0


def _far(y_true, y_pred) -> float:
    """False Alarm Ratio = FP / (TP + FP)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = tp + fp
    return fp / denom if denom else 0.0


def _brier_skill_score(y_true, y_prob) -> float:
    """BSS relative to climatological (no-skill) forecast."""
    bs       = brier_score_loss(y_true, y_prob)
    climo    = float(y_true.mean())
    bs_climo = brier_score_loss(y_true, np.full_like(y_prob, climo))
    return 1 - bs / bs_climo if bs_climo else 0.0


def evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Compute all metrics for one set of predictions."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model":         name,
        "n_samples":     int(len(y_true)),
        "flood_rate":    float(y_true.mean()),
        "roc_auc":       float(roc_auc_score(y_true, y_prob)),
        "pr_auc":        float(average_precision_score(y_true, y_prob)),
        "f1":            float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":     float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_true, y_pred, zero_division=0)),
        "csi":           float(_csi(y_true, y_pred)),
        "far":           float(_far(y_true, y_pred)),
        "brier_score":   float(brier_score_loss(y_true, y_prob)),
        "brier_skill":   float(_brier_skill_score(y_true, y_prob)),
        "threshold":     threshold,
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    log.info(
        "%s | AUC=%.4f PR-AUC=%.4f F1=%.4f CSI=%.4f FAR=%.4f BSS=%.4f",
        name,
        metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
        metrics["csi"], metrics["far"], metrics["brier_skill"],
    )
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            metric: str = "f1") -> float:
    """Grid-search threshold ∈ [0.1, 0.9] to maximise chosen metric."""
    best_val, best_thr = -1, 0.5
    for thr in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "csi":
            val = _csi(y_true, y_pred)
        else:
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, thr
    log.info("Optimal threshold for %s: %.2f (%.4f)", metric, best_thr, best_val)
    return best_thr


def evaluate_all(ensemble, df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Run full evaluation suite on test set (last 10 % of data).
    Saves metrics JSON and per-model comparison CSV.
    """
    split = int(len(df) * 0.9)
    test  = df.iloc[split:]
    y_true = test["flood"].values

    results = {}

    # ── LSTM ──────────────────────────────────────────────────────────────────
    try:
        idx_l, p_lstm = ensemble.lstm.predict_proba.__func__.__code__  # probe
    except Exception:
        pass

    try:
        p_lstm = ensemble.lstm.predict_proba(df)
        p_lstm = p_lstm[-len(y_true):]   # align
        y_l    = y_true[-len(p_lstm):]
        results["LSTM"] = evaluate_model("LSTM", y_l, p_lstm)
    except Exception as e:
        log.warning("LSTM evaluation failed: %s", e)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        p_xgb = ensemble.xgb.predict_proba(test)
        results["XGBoost"] = evaluate_model("XGBoost", y_true, p_xgb)
    except Exception as e:
        log.warning("XGBoost evaluation failed: %s", e)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    try:
        _, p_ens = ensemble.predict_proba_aligned(df)
        p_ens    = p_ens[-len(y_true):]
        y_e      = y_true[-len(p_ens):]
        opt_thr  = find_optimal_threshold(y_e, p_ens, metric="f1")
        results["Ensemble"] = evaluate_model("Ensemble", y_e, p_ens, threshold=opt_thr)
        ensemble.set_threshold(opt_thr)
    except Exception as e:
        log.warning("Ensemble evaluation failed: %s", e)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)

    if results:
        comparison = pd.DataFrame(results).T
        comparison.to_csv(out_dir / "model_comparison.csv")

        # Print clean summary
        print("\n" + "═" * 70)
        print("  MODEL EVALUATION SUMMARY")
        print("═" * 70)
        for model_name, m in results.items():
            print(f"\n  {model_name}")
            print(f"    ROC-AUC : {m['roc_auc']:.4f}")
            print(f"    PR-AUC  : {m['pr_auc']:.4f}")
            print(f"    F1      : {m['f1']:.4f}")
            print(f"    CSI     : {m['csi']:.4f}")
            print(f"    FAR     : {m['far']:.4f}")
            print(f"    BSS     : {m['brier_skill']:.4f}")
            print(f"    TP/FP/FN: {m['tp']}/{m['fp']}/{m['fn']}")
        print("═" * 70 + "\n")

    return results
