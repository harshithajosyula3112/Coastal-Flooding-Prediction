"""
src/data_loader.py
------------------
Downloads tide-gauge water levels and meteorological data from the
NOAA CO-OPS API, caches results locally, and returns a merged DataFrame.

NOAA API docs: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Products we pull for each station
PRODUCTS = {
    "water_level":    "Observed water level (metres, MHHW datum)",
    "wind":           "Wind speed & direction",
    "air_pressure":   "Barometric pressure",
    "air_temperature":"Air temperature",
    "water_temperature": "Water temperature",
}


def _fetch_product(
    station_id: str,
    product: str,
    start_date: str,
    end_date: str,
    units: str = "metric",
    time_zone: str = "GMT",
    interval: str = "h",
) -> pd.DataFrame:
    """
    Pull one NOAA product for the given date range.
    NOAA limits single requests to 1 year; this function chunks automatically.
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    chunks: list[pd.DataFrame] = []

    # Split into annual chunks to respect NOAA's 1-year limit
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        params = {
            "product":   product,
            "station":   station_id,
            "begin_date": current.strftime("%Y%m%d"),
            "end_date":   chunk_end.strftime("%Y%m%d"),
            "datum":     "MHHW",
            "time_zone": time_zone,
            "interval":  interval,
            "units":     units,
            "application": "coastal_flood_ml",
            "format":    "json",
        }
        log.debug("Fetching %s  %s → %s", product, current.date(), chunk_end.date())
        resp = requests.get(NOAA_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            log.warning("NOAA error for %s [%s–%s]: %s",
                        product, current.date(), chunk_end.date(), data["error"])
            current = chunk_end + pd.Timedelta(days=1)
            time.sleep(1)
            continue

        obs = data.get("data", [])
        if obs:
            df = pd.DataFrame(obs)
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t")
            chunks.append(df)

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)   # be polite to the API

    return pd.concat(chunks) if chunks else pd.DataFrame()


def _parse_water_level(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["v"], errors="coerce")
    s.name = "water_level_m"
    return s


def _parse_wind(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["wind_speed_ms"]  = pd.to_numeric(df["s"], errors="coerce")
    out["wind_dir_deg"]   = pd.to_numeric(df["d"], errors="coerce")
    return out


def _parse_scalar(df: pd.DataFrame, col_name: str) -> pd.Series:
    s = pd.to_numeric(df["v"], errors="coerce")
    s.name = col_name
    return s


def load_noaa_data(
    station_id: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
    skip_download: bool = False,
) -> pd.DataFrame:
    """
    Load and merge all NOAA products for a given station and date range.

    Parameters
    ----------
    station_id    : NOAA station ID (e.g. '8638610' for Sewells Point, VA)
    start_date    : ISO date string, e.g. '2010-01-01'
    end_date      : ISO date string, e.g. '2023-12-31'
    cache_dir     : Local directory for Parquet cache files
    skip_download : If True, return cached data without hitting the API

    Returns
    -------
    pd.DataFrame with hourly index and columns for each meteorological product
    """
    cache_path = Path(cache_dir) / f"{station_id}_{start_date}_{end_date}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_download and cache_path.exists():
        log.info("Loading cached data from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Downloading NOAA data for station %s  (%s → %s)",
             station_id, start_date, end_date)

    frames: dict[str, pd.Series | pd.DataFrame] = {}

    # Water level
    raw_wl = _fetch_product(station_id, "water_level", start_date, end_date)
    if not raw_wl.empty:
        frames["water_level"] = _parse_water_level(raw_wl)

    # Wind
    raw_wind = _fetch_product(station_id, "wind", start_date, end_date)
    if not raw_wind.empty:
        wind_df = _parse_wind(raw_wind)
        frames["wind_speed_ms"] = wind_df["wind_speed_ms"]
        frames["wind_dir_deg"]  = wind_df["wind_dir_deg"]

    # Scalar products
    scalar_map = {
        "air_pressure":      "pressure_mb",
        "air_temperature":   "air_temp_c",
        "water_temperature": "water_temp_c",
    }
    for product, col in scalar_map.items():
        raw = _fetch_product(station_id, product, start_date, end_date)
        if not raw.empty:
            frames[col] = _parse_scalar(raw, col)

    # Merge all into one hourly DataFrame
    merged = pd.concat(frames.values(), axis=1)
    merged.index.name = "datetime"
    merged = merged.sort_index()

    log.info("Downloaded %d hourly records with %d features",
             len(merged), merged.shape[1])

    merged.to_parquet(cache_path)
    log.info("Cached to %s", cache_path)

    return merged
