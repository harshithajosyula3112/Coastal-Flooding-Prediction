"""
Coastal Flooding Prediction Pipeline
=====================================
Author : Harshitha Josyula
Project: iHARP ML Challenge
Stack  : Python · LSTM · XGBoost · NOAA tide-gauge data
"""

import argparse
import logging
from pathlib import Path

from src.data_loader   import load_noaa_data
from src.preprocessor  import preprocess
from src.features      import engineer_features
from src.model_lstm    import LSTMForecaster
from src.model_xgb     import XGBForecaster
from src.ensemble      import EnsembleModel
from src.evaluate      import evaluate_all
from src.visualize     import plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Coastal Flooding Prediction Pipeline")
    p.add_argument("--station",   default="8638610", help="NOAA station ID")
    p.add_argument("--start",     default="2010-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default="2023-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Flood threshold in metres above MHHW")
    p.add_argument("--seq-len",   type=int, default=24, help="LSTM lookback window (hours)")
    p.add_argument("--output",    default="outputs", help="Output directory")
    p.add_argument("--skip-download", action="store_true",
                   help="Use cached data if available")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log.info("── Step 1 / 6  Load NOAA data ──────────────────────────")
    raw = load_noaa_data(
        station_id   = args.station,
        start_date   = args.start,
        end_date     = args.end,
        cache_dir    = "data",
        skip_download= args.skip_download,
    )

    log.info("── Step 2 / 6  Preprocess ───────────────────────────────")
    df = preprocess(raw, flood_threshold=args.threshold)

    log.info("── Step 3 / 6  Feature engineering ─────────────────────")
    df = engineer_features(df)

    log.info("── Step 4 / 6  Train models ─────────────────────────────")
    lstm = LSTMForecaster(seq_len=args.seq_len)
    xgb  = XGBForecaster()
    ens  = EnsembleModel(lstm, xgb)
    ens.fit(df)

    log.info("── Step 5 / 6  Evaluate ─────────────────────────────────")
    metrics = evaluate_all(ens, df, out)

    log.info("── Step 6 / 6  Visualise ────────────────────────────────")
    plot_results(df, ens, metrics, out)

    log.info("Done.  Results saved to %s", out)
    return metrics


if __name__ == "__main__":
    main()
