from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

NUMERIC_DTYPES = ("int16","int32","int64","float16","float32","float64")

def load_all_csvs(input_folder: str, target_col: str = "y_5d") -> pd.DataFrame:
    """
    Load all per-ticker CSVs. Ensure Date is YYYY-MM-DD, add Ticker from filename if missing.
    Ensure y_5d exists (compute from Adj Close/Close if absent).
    """
    dfs = []
    p = Path(input_folder)
    for f in sorted(p.glob("*.csv")):
        ticker = f.stem.upper()
        df = pd.read_csv(f)
        if "Ticker" not in df.columns:
            df.insert(0, "Ticker", ticker)
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], errors="coerce")
            df["Date"] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), df["Date"].astype(str).str[:10])
        if target_col not in df.columns:
            price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
            if price_col is not None:
                df["fwd_ret_5d"] = df[price_col].shift(-5) / df[price_col] - 1.0
                df[target_col] = (df["fwd_ret_5d"] > 0).astype(int)
                df["Return_5d"] = df[price_col].pct_change(5)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSVs in {input_folder}")
    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=[target_col], how="any")
    full = full.sort_values(["Ticker","Date"]).reset_index(drop=True)
    return full

def split_by_tickers(full_df: pd.DataFrame, test_tickers: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    8+2 ticker split. If test_tickers not provided, pick the last two alphabetically.
    """
    uniq = sorted(full_df["Ticker"].astype(str).unique().tolist())
    if test_tickers is None:
        test_tickers = uniq[-2:]
    else:
        test_tickers = [t.upper() for t in test_tickers]
    tr = full_df[~full_df["Ticker"].isin(test_tickers)].copy()
    te = full_df[ full_df["Ticker"].isin(test_tickers)].copy()
    return tr, te

def select_numeric_features(df: pd.DataFrame, target_col: str="y_5d", extra_drop: Optional[Sequence[str]]=None):
    drop_cols = {"Date", target_col, "Return_5d"}
    if extra_drop: drop_cols.update(extra_drop)
    keep = [c for c in df.columns if c not in drop_cols]
    keep = [c for c in keep if str(df[c].dtype) in NUMERIC_DTYPES]
    X = df[keep].replace([np.inf,-np.inf], np.nan).fillna(0.0).copy()
    y = df[target_col].astype(int).copy()
    return X, y

def build_sequences(df: pd.DataFrame, window: int=30, target_col: str="y_5d", extra_drop: Optional[Sequence[str]]=("Ticker",)):
    """
    Sliding windows per ticker. Label = target at window end.
    Returns X:(N,window,F) and y:(N,)
    """
    Xs, ys = [], []
    for _, g in df.groupby("Ticker"):
        g = g.sort_values("Date")
        Xg, yg = select_numeric_features(g, target_col=target_col, extra_drop=extra_drop)
        Xv, yv = Xg.values, yg.values
        if len(g) < window: continue
        for i in range(window-1, len(g)):
            Xs.append(Xv[i-window+1:i+1]); ys.append(yv[i])
    if not Xs:
        raise ValueError("No sequences built. Reduce window or check data.")
    X = np.stack(Xs).astype("float32")
    y = np.array(ys).astype("float32")
    return X, y

def print_metrics(y_true, y_prob, threshold: float=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = float("nan")
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"\nROC-AUC: {auc:.4f}")
    print("Confusion Matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)
    try:
        print("\n"+classification_report(y_true, y_pred, digits=4))
    except Exception:
        pass