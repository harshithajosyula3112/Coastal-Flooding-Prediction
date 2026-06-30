"""
src/model_xgb.py
----------------
XGBoost flood classifier with SHAP-based feature importance.

Handles class imbalance via scale_pos_weight and uses Optuna for
lightweight hyperparameter tuning on the validation set.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

# All tabular features available to XGBoost
XGB_FEATURES = [
    # Tide
    "water_level_m", "wl_roc_1h", "wl_roc_3h", "wl_roc_6h",
    "tidal_anomaly", "surge_event",
    "wl_lag_1h", "wl_lag_3h", "wl_lag_6h", "wl_lag_12h", "wl_lag_24h",
    "water_level_m_roll3h_mean", "water_level_m_roll6h_max",
    "water_level_m_roll12h_mean", "water_level_m_roll24h_max",
    # Wind
    "wind_speed_ms", "wind_u", "wind_v", "wind_sustained_6h",
    "wind_speed_lag_3h", "wind_speed_lag_6h", "wind_onshore",
    # Pressure
    "pressure_mb", "pressure_tend_1h", "pressure_tend_3h", "pressure_tend_6h",
    "inv_barometer_effect", "pressure_lag_6h", "pressure_lag_12h",
    # Temporal
    "hour", "month", "season", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Met
    "air_temp_c", "water_temp_c",
]

DEFAULT_PARAMS = {
    "n_estimators":     800,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "eval_metric":      "auc",
    "random_state":     42,
    "n_jobs":           -1,
}


class XGBForecaster:
    """
    XGBoost binary classifier for flood prediction.

    Optionally runs Optuna hyperparameter search before final training.
    SHAP values are computed after training for interpretability.
    """

    def __init__(self, tune: bool = False, n_trials: int = 30):
        self.tune      = tune
        self.n_trials  = n_trials
        self.model     = None
        self.feature_cols: list[str] = []
        self.shap_values = None
        self._threshold  = 0.5

    # ── Data prep ──────────────────────────────────────────────────────────────
    def _get_xy(self, df: pd.DataFrame):
        self.feature_cols = [c for c in XGB_FEATURES if c in df.columns]
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df["flood"]
        return X, y

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    def _tune(self, X_tr, y_tr, X_va, y_va, scale_pos_weight: float):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("Optuna not installed — skipping hyperparameter tuning.")
            return DEFAULT_PARAMS.copy()

        from xgboost import XGBClassifier

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
                "max_depth":        trial.suggest_int("max_depth", 4, 9),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
                "tree_method":      "hist",
                "eval_metric":      "auc",
                "random_state":     42,
                "n_jobs":           -1,
            }
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False)
            return roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"scale_pos_weight": scale_pos_weight,
                     "tree_method": "hist", "eval_metric": "auc",
                     "random_state": 42, "n_jobs": -1})
        log.info("Best Optuna params: %s  (AUC=%.4f)", best, study.best_value)
        return best

    # ── Fit ────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        from xgboost import XGBClassifier

        X, y = self._get_xy(df)

        # Chronological 80/10/10 split
        n = len(X)
        i1, i2 = int(n * 0.8), int(n * 0.9)
        X_tr, y_tr = X.iloc[:i1],   y.iloc[:i1]
        X_va, y_va = X.iloc[i1:i2], y.iloc[i1:i2]

        pos_rate        = float(y_tr.mean())
        scale_pos_weight= (1 - pos_rate) / max(pos_rate, 1e-6)
        log.info("XGB scale_pos_weight=%.2f  (flood rate=%.3f%%)",
                 scale_pos_weight, pos_rate * 100)

        params = (
            self._tune(X_tr, y_tr, X_va, y_va, scale_pos_weight)
            if self.tune
            else {**DEFAULT_PARAMS, "scale_pos_weight": scale_pos_weight}
        )

        self.model = XGBClassifier(**params)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100,
        )

        val_auc = roc_auc_score(y_va, self.model.predict_proba(X_va)[:, 1])
        log.info("XGB validation AUC: %.4f", val_auc)

        # SHAP feature importance
        self._compute_shap(X_tr.sample(min(2000, len(X_tr)), random_state=42))

        return self

    def _compute_shap(self, X_sample: pd.DataFrame):
        try:
            import shap
            explainer        = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_sample)
            log.info("SHAP values computed on %d samples.", len(X_sample))
        except ImportError:
            log.warning("shap not installed — skipping feature importance.")

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._get_xy(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self._threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importance as a sorted Series."""
        imp = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False)

    # ── Persist ────────────────────────────────────────────────────────────────
    def save(self, path: str = "models/xgb_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("XGB model saved to %s", path)

    @classmethod
    def load(cls, path: str = "models/xgb_model.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("XGB model loaded from %s", path)
        return obj
