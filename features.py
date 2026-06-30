"""
src/features.py
---------------
Domain-driven feature engineering for coastal flood prediction.

Features created
────────────────
Temporal         : hour, month, season, is_weekend, hour_sin/cos, month_sin/cos
Tide             : rolling means/maxes, rate-of-change, tidal anomaly
Wind             : u/v components, sustained wind proxy, directional bins
Pressure         : pressure tendency (1h, 3h, 6h drops), inverse barometer effect
Surge            : storm surge proxy = water_level − predicted tidal signal
Lag features     : 1h, 3h, 6h, 12h, 24h lags of water level and pressure
Lead label       : next-1h, next-3h, next-6h flood labels for multi-horizon output
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sin_cos(series: pd.Series, period: float):
    """Encode a cyclic variable as sine/cosine pair."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _rolling(series: pd.Series, windows: list[int], funcs: dict):
    """Return dict of rolling statistics for given windows."""
    result = {}
    for w in windows:
        for name, fn in funcs.items():
            result[f"{series.name}_roll{w}h_{name}"] = (
                series.rolling(w, min_periods=max(1, w // 2)).agg(fn)
            )
    return result


# ── Feature groups ─────────────────────────────────────────────────────────────

def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"]       = idx.hour
    df["month"]      = idx.month
    df["dayofweek"]  = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["season"]     = ((idx.month % 12) // 3).astype(int)   # 0=winter … 3=fall

    h_sin, h_cos = _sin_cos(pd.Series(idx.hour,  index=idx), 24)
    m_sin, m_cos = _sin_cos(pd.Series(idx.month, index=idx), 12)
    df["hour_sin"]  = h_sin.values
    df["hour_cos"]  = h_cos.values
    df["month_sin"] = m_sin.values
    df["month_cos"] = m_cos.values
    return df


def tide_features(df: pd.DataFrame) -> pd.DataFrame:
    wl = df["water_level_m"]

    # Rolling stats
    for col, val in _rolling(wl, [3, 6, 12, 24], {"mean": "mean", "max": "max", "std": "std"}).items():
        df[col] = val

    # Rate of change
    df["wl_roc_1h"]  = wl.diff(1)
    df["wl_roc_3h"]  = wl.diff(3)
    df["wl_roc_6h"]  = wl.diff(6)

    # Tidal anomaly: deviation from 25-hour running mean (removes tidal signal)
    df["tidal_anomaly"] = wl - wl.rolling(25, min_periods=12, center=True).mean()

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        df[f"wl_lag_{lag}h"] = wl.shift(lag)

    return df


def wind_features(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_speed_ms" not in df.columns:
        return df

    speed = df["wind_speed_ms"]
    direc = df.get("wind_dir_deg", pd.Series(0, index=df.index))

    # Decompose into meteorological u/v components
    rad = np.deg2rad(direc)
    df["wind_u"] = -speed * np.sin(rad)   # east–west
    df["wind_v"] = -speed * np.cos(rad)   # north–south

    # Sustained wind (6h mean) — more predictive than instantaneous
    df["wind_sustained_6h"] = speed.rolling(6, min_periods=3).mean()

    # Onshore wind proxy for Hampton Roads / Mid-Atlantic (dominant from NE–SE)
    df["wind_onshore"] = (
        ((direc >= 0) & (direc <= 135)) | ((direc >= 315) & (direc <= 360))
    ).astype(int)

    # Lag
    df["wind_speed_lag_3h"] = speed.shift(3)
    df["wind_speed_lag_6h"] = speed.shift(6)

    return df


def pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    if "pressure_mb" not in df.columns:
        return df

    p = df["pressure_mb"]

    # Pressure tendency — falling pressure precedes storm surge
    df["pressure_tend_1h"] = p.diff(1)
    df["pressure_tend_3h"] = p.diff(3)
    df["pressure_tend_6h"] = p.diff(6)

    # Inverse barometer effect: 1 hPa drop ≈ +1 cm water level rise
    # Use anomaly from 24h mean
    df["inv_barometer_effect"] = -(p - p.rolling(24, min_periods=12).mean()) / 100

    df["pressure_lag_6h"]  = p.shift(6)
    df["pressure_lag_12h"] = p.shift(12)

    return df


def surge_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Storm surge proxy = water level anomaly from 25h tidal average.
    Already computed in tide_features; this adds a binary 'surge_event' flag.
    """
    if "tidal_anomaly" not in df.columns:
        return df

    surge_threshold = df["tidal_anomaly"].quantile(0.90)
    df["surge_event"] = (df["tidal_anomaly"] >= surge_threshold).astype(int)
    return df


def lead_labels(df: pd.DataFrame, horizons: list[int] = [1, 3, 6]) -> pd.DataFrame:
    """Create shifted flood labels for multi-horizon prediction targets."""
    for h in horizons:
        df[f"flood_next_{h}h"] = df["flood"].shift(-h)
    return df


# ── Main entry ─────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Parameters
    ----------
    df : Preprocessed DataFrame from preprocessor.preprocess()

    Returns
    -------
    DataFrame with all engineered features appended
    """
    log.info("Engineering features …")
    n_before = df.shape[1]

    df = temporal_features(df)
    df = tide_features(df)
    df = wind_features(df)
    df = pressure_features(df)
    df = surge_proxy(df)
    df = lead_labels(df, horizons=[1, 3, 6])

    # Drop rows where lag/lead features are still NaN (first/last 24 rows)
    df = df.iloc[24:-6].copy()

    n_after = df.shape[1]
    log.info("Feature engineering complete: %d → %d columns, %d rows",
             n_before, n_after, len(df))
    return df
