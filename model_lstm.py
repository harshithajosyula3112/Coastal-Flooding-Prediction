"""
src/model_lstm.py
-----------------
Bidirectional LSTM with attention for hourly flood probability forecasting.

Architecture
────────────
Input  → BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Attention → Dense(32) → Sigmoid
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Feature columns used by the LSTM (time-ordered sequences matter here)
LSTM_FEATURES = [
    "water_level_m", "wl_roc_1h", "wl_roc_3h", "tidal_anomaly",
    "wl_lag_1h", "wl_lag_3h", "wl_lag_6h", "wl_lag_12h",
    "wind_speed_ms", "wind_u", "wind_v", "wind_sustained_6h",
    "pressure_mb", "pressure_tend_1h", "pressure_tend_3h", "inv_barometer_effect",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "season",
    "surge_event",
]


def _build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    """Convert flat DataFrame into overlapping (X, y) sequence arrays."""
    available = [c for c in feature_cols if c in df.columns]
    X_arr = df[available].fillna(0).values
    y_arr = df["flood"].values

    X_seqs, y_seqs = [], []
    for i in range(seq_len, len(X_arr)):
        X_seqs.append(X_arr[i - seq_len: i])
        y_seqs.append(y_arr[i])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


class AttentionLayer:
    """Simple scaled dot-product attention (NumPy implementation for portability)."""

    @staticmethod
    def forward(lstm_output: np.ndarray) -> np.ndarray:
        # lstm_output: (batch, timesteps, features)
        scores = np.sum(lstm_output * lstm_output[:, -1:, :], axis=-1, keepdims=True)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        return (lstm_output * weights).sum(axis=1)


class LSTMForecaster:
    """
    Bidirectional LSTM flood probability forecaster.

    Uses TensorFlow/Keras when available; falls back to a lightweight
    scikit-learn GradientBoosting model on sequence-flattened features
    so the pipeline works in environments without a GPU or TF installed.
    """

    def __init__(self, seq_len: int = 24, epochs: int = 30, batch_size: int = 256):
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model      = None
        self.backend    = None
        self.feature_cols: list[str] = []
        self._threshold = 0.5

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_keras_model(self, n_features: int):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, Input

            inp = Input(shape=(self.seq_len, n_features))

            # BiLSTM block 1
            x = layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, dropout=0.2,
                            recurrent_dropout=0.1)
            )(inp)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(0.3)(x)

            # BiLSTM block 2
            x = layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2)
            )(x)

            # Attention
            attn = layers.Dense(1, activation="tanh")(x)
            attn = layers.Flatten()(attn)
            attn = layers.Activation("softmax")(attn)
            attn = layers.RepeatVector(128)(attn)       # 2 × 64 from BiLSTM
            attn = layers.Permute([2, 1])(attn)
            ctx  = layers.Multiply()([x, attn])
            ctx  = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(ctx)

            # Head
            x = layers.Dense(32, activation="relu")(ctx)
            x = layers.Dropout(0.2)(x)
            out = layers.Dense(1, activation="sigmoid")(x)

            model = Model(inp, out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["AUC", "Precision", "Recall"],
            )
            self.backend = "keras"
            return model

        except ImportError:
            log.warning("TensorFlow not found — using sklearn fallback for LSTM module.")
            return None

    def _build_sklearn_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.backend = "sklearn"
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42, verbose=0,
        )

    # ── Fit ────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        self.feature_cols = [c for c in LSTM_FEATURES if c in df.columns]
        log.info("LSTM training on %d features, seq_len=%d", len(self.feature_cols), self.seq_len)

        # Chronological 80/20 split
        split = int(len(df) * 0.8)
        train, val = df.iloc[:split], df.iloc[split:]

        X_tr, y_tr = _build_sequences(train, self.feature_cols, self.seq_len)
        X_va, y_va = _build_sequences(val,   self.feature_cols, self.seq_len)

        # Class imbalance: compute pos_weight
        pos_rate = y_tr.mean()
        class_weight = {0: 1.0, 1: max(1.0, (1 - pos_rate) / max(pos_rate, 1e-6))}
        log.info("Class weight ratio  0:%.2f  1:%.2f", class_weight[0], class_weight[1])

        n_features = X_tr.shape[2]
        model = self._build_keras_model(n_features)

        if model is not None:
            import tensorflow as tf
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", patience=5, restore_best_weights=True, mode="max"
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
            ]
            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            model = self._build_sklearn_model()
            # Flatten sequences for sklearn
            X_tr_flat = X_tr.reshape(len(X_tr), -1)
            X_va_flat = X_va.reshape(len(X_va), -1)
            model.fit(X_tr_flat, y_tr)

        self.model = model
        log.info("LSTM (%s backend) training complete", self.backend)
        return self

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return flood probability for each row (after seq_len burn-in)."""
        X, _ = _build_sequences(df, self.feature_cols, self.seq_len)
        if self.backend == "keras":
            proba = self.model.predict(X, verbose=0).flatten()
        else:
            proba = self.model.predict_proba(X.reshape(len(X), -1))[:, 1]
        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self._threshold).astype(int)

    # ── Persist ────────────────────────────────────────────────────────────────
    def save(self, path: str = "models/lstm_model"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "keras":
            self.model.save(f"{path}.keras")
        else:
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(self.model, f)
        log.info("LSTM model saved to %s", path)

    def load(self, path: str = "models/lstm_model"):
        keras_path = Path(f"{path}.keras")
        pkl_path   = Path(f"{path}.pkl")
        if keras_path.exists():
            import tensorflow as tf
            self.model   = tf.keras.models.load_model(str(keras_path))
            self.backend = "keras"
        elif pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.model = pickle.load(f)
            self.backend = "sklearn"
        else:
            raise FileNotFoundError(f"No saved model found at {path}")
        log.info("LSTM model loaded from %s", path)
        return self
