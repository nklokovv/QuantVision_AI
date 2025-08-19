# models/predict_catboost_simple_main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def _resolve_model_path(model_path_or_dir: str) -> Path:
    p = Path(model_path_or_dir).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        # pick first .cbm in the directory
        for cbm in sorted(p.glob("*.cbm")):
            return cbm
    raise FileNotFoundError(f"No model file found at: {p}")

def _build_features(df: pd.DataFrame, ticker: str, target_col: str = "y_5d") -> tuple[pd.DataFrame, List[int], List[str]]:
    """
    Feature rule (no meta needed):
      feat_cols = ['Ticker'] + [all other columns except {target_col, 'Date', 'Return_5d'}]
    Ensures Ticker exists and is str; cat_features_idx = [0].
    """
    if "Ticker" not in df.columns:
        df.insert(0, "Ticker", ticker.upper())
    else:
        # keep Ticker as first feature
        cols = df.columns.tolist()
        cols = ["Ticker"] + [c for c in cols if c != "Ticker"]
        df = df[cols]

    drop = {target_col, "Date", "Return_5d"}
    feat_cols = ["Ticker"] + [c for c in df.columns if c not in drop and c != "Ticker"]

    X = df[feat_cols].copy()
    X["Ticker"] = X["Ticker"].astype(str)
    X = X.replace([np.inf, -np.inf], np.nan)
    cat_idx = [0]
    return X, cat_idx, feat_cols

def predict_for_ticker(
    data_dir: str,
    ticker: str,
    model_path_or_dir: str,
    target_col: str = "y_5d",
    out_dir: Optional[str] = None,
) -> Path:
    data_dir_p = Path(data_dir).expanduser().resolve()
    model_path = _resolve_model_path(model_path_or_dir)
    out_dir_p = Path(out_dir).expanduser().resolve() if out_dir else model_path.parent

    csv_path = data_dir_p / f"{ticker.upper()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    X, cat_idx, feat_cols = _build_features(df, ticker, target_col=target_col)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    y_prob = model.predict_proba(Pool(X, cat_features=cat_idx))[:, 1]

    pred = pd.DataFrame({
        "Ticker": df["Ticker"] if "Ticker" in df.columns else ticker.upper(),
        "Date": df["Date"] if "Date" in df.columns else "",
        "y_prob": y_prob,
    })
    if "Return_5d" in df.columns:
        pred["Return_5d"] = df["Return_5d"]

    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / f"{ticker.upper()}_pred.csv"
    pred.to_csv(out_path, index=False)
    print(f"✅ Saved predictions → {out_path}  (rows={len(pred)})")
    return out_path


# -------- regular main (edit & run) --------
if __name__ == "__main__":
    DATA_DIR  = "dataset/train_data/final"     # folder with per-ticker CSVs
    MODEL     = "dataset/models_cv/catboost_cv_best.cbm"  # or a folder containing *.cbm
    TICKER    = "AAPL"

    predict_for_ticker(DATA_DIR, TICKER, MODEL, target_col="y_5d", out_dir=None)
