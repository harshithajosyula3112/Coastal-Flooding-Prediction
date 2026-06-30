"""
src/ensemble.py
---------------
Weighted average ensemble of LSTM and XGBoost predictions.

The ensemble weight (alpha) is optimised on the validation set to
maximise F1, which is more informative than AUC for imbalanced flood data.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


class EnsembleModel:
    """
    Combines LSTM and XGBoost via a learned convex combination:

        P_ensemble = alpha * P_lstm + (1 - alpha) * P_xgb

    alpha is tuned on the validation fold by grid search over [0, 1].
    """

    def __init__(self, lstm, xgb, alpha: float = 0.5):
        self.lstm      = lstm
        self.xgb       = xgb
        self.alpha     = alpha          # weight on LSTM output
        self._threshold= 0.5

    # ── Fit both sub-models then tune alpha ────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        log.info("Training LSTM …")
        self.lstm.fit(df)

        log.info("Training XGBoost …")
        self.xgb.fit(df)

        log.info("Tuning ensemble weight …")
        self._tune_alpha(df)

        return self

    def _tune_alpha(self, df: pd.DataFrame):
        """
        Use the last 10 % of data as a held-out blend-tuning set.
        Grid search alpha ∈ {0.0, 0.1, …, 1.0} for best F1.
        """
        split = int(len(df) * 0.9)
        val   = df.iloc[split:]
        y_val = val["flood"].values

        # LSTM needs seq_len rows of context — use full df for prediction
        try:
            p_lstm = self.lstm.predict_proba(df)
            # Align: LSTM output is shorter than df by seq_len rows
            seq_len = self.lstm.seq_len
            p_lstm_val = p_lstm[max(0, len(p_lstm) - len(val)):]
            y_val_aligned = y_val[-len(p_lstm_val):]
        except Exception as e:
            log.warning("LSTM predict failed during alpha tuning: %s — defaulting alpha=0.4", e)
            self.alpha = 0.4
            return

        p_xgb_val = self.xgb.predict_proba(val)
        # Match lengths
        min_len = min(len(p_lstm_val), len(p_xgb_val))
        p_lstm_val = p_lstm_val[-min_len:]
        p_xgb_val  = p_xgb_val[-min_len:]
        y_aligned  = y_val_aligned[-min_len:]

        best_f1, best_alpha = -1, 0.5
        for a in np.arange(0.0, 1.01, 0.05):
            p_ens = a * p_lstm_val + (1 - a) * p_xgb_val
            preds = (p_ens >= self._threshold).astype(int)
            f1    = f1_score(y_aligned, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_alpha = f1, a

        self.alpha = round(best_alpha, 2)
        log.info("Optimal ensemble alpha=%.2f  (val F1=%.4f)", self.alpha, best_f1)

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba_aligned(self, df: pd.DataFrame):
        """
        Returns (timestamps, probabilities) with LSTM burn-in rows removed.
        """
        seq_len   = getattr(self.lstm, "seq_len", 24)
        p_lstm    = self.lstm.predict_proba(df)
        p_xgb_all = self.xgb.predict_proba(df)
        p_xgb     = p_xgb_all[-len(p_lstm):]

        p_ens  = self.alpha * p_lstm + (1 - self.alpha) * p_xgb
        index  = df.index[-len(p_ens):]
        return index, p_ens

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        _, proba = self.predict_proba_aligned(df)
        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self._threshold).astype(int)

    def set_threshold(self, threshold: float):
        """Adjust classification threshold (useful for precision/recall trade-off)."""
        self._threshold      = threshold
        self.lstm._threshold = threshold
        self.xgb._threshold  = threshold
        log.info("Classification threshold set to %.2f", threshold)
