"""
src/preprocessor.py
-------------------
Cleans raw NOAA data, handles missing values, creates the flood label,
and produces a analysis-ready DataFrame.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

# Columns expected from the data loader
REQUIRED_COLS = ["water_level_m"]
OPTIONAL_COLS = ["wind_speed_ms", "wind_dir_deg", "pressure_mb",
                 "air_temp_c", "water_temp_c"]

# Physically plausible bounds for outlier capping
BOUNDS = {
    "water_level_m":  (-3.0,  5.0),
    "wind_speed_ms":  (  0.0, 90.0),
    "wind_dir_deg":   (  0.0,360.0),
    "pressure_mb":    (870.0,1084.0),
    "air_temp_c":     (-40.0, 55.0),
    "water_temp_c":   ( -2.0, 35.0),
}


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values to physically plausible ranges."""
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].clip(lo, hi)
            after  = df[col].isna().sum()
            if after > before:
                log.debug("Capped %d outliers in %s", after - before, col)
    return df


def _fill_missing(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """
    Fill short gaps (≤ max_gap_hours) with linear interpolation.
    Longer gaps are left as NaN; rows with NaN water_level are dropped later.
    """
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].interpolate(
                method="time", limit=max_gap_hours, limit_direction="both"
            )
            filled = n_missing - df[col].isna().sum()
            log.debug("Filled %d / %d missing values in %s", filled, n_missing, col)
    return df


def _create_flood_label(water_level: pd.Series, threshold: float) -> pd.Series:
    """
    Binary flood label: 1 when water level exceeds threshold above MHHW.
    Also create a 'near-flood' label at 80 % of threshold for early warning.
    """
    return (water_level >= threshold).astype(int)


def preprocess(
    df: pd.DataFrame,
    flood_threshold: float = 0.5,
    max_gap_hours: int = 3,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df              : Raw merged DataFrame from data_loader
    flood_threshold : Metres above MHHW to define a flood event
    max_gap_hours   : Maximum consecutive missing hours to interpolate

    Returns
    -------
    Clean DataFrame with flood label and scaled features appended
    """
    log.info("Raw shape: %s", df.shape)

    # ── 1. Validate required columns ──────────────────────────────────────────
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from data.")

    # ── 2. Enforce hourly frequency (reindex with hourly periods) ─────────────
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "datetime"

    # ── 3. Cap physical outliers ──────────────────────────────────────────────
    df = _cap_outliers(df)

    # ── 4. Fill short gaps ────────────────────────────────────────────────────
    df = _fill_missing(df, max_gap_hours=max_gap_hours)

    # ── 5. Drop rows where water level is still missing ───────────────────────
    before = len(df)
    df = df.dropna(subset=["water_level_m"])
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with missing water_level_m", dropped)

    # ── 6. Create flood label ─────────────────────────────────────────────────
    df["flood"]      = _create_flood_label(df["water_level_m"], flood_threshold)
    df["near_flood"] = _create_flood_label(df["water_level_m"], 0.8 * flood_threshold)

    flood_rate = df["flood"].mean() * 100
    log.info("Flood events: %.2f%% of hourly records (threshold=%.2f m)",
             flood_rate, flood_threshold)

    # ── 7. Scale numeric features (preserve originals with _raw suffix) ────────
    feature_cols = [c for c in OPTIONAL_COLS if c in df.columns]
    if feature_cols:
        scaler = RobustScaler()
        df[[f"{c}_scaled" for c in feature_cols]] = scaler.fit_transform(
            df[feature_cols].fillna(df[feature_cols].median())
        )

    log.info("Preprocessed shape: %s", df.shape)
    return df
