# backtests/backtest_long5d_main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# import the predictor (no meta/json required)
from research_files_and_data.models.predict import predict_for_ticker


def _resolve_path_smart(p: str, anchor_file: str = __file__) -> Path:
    """Resolve p as absolute/CWD, or relative to the project root."""
    cand = Path(p).expanduser()
    if cand.is_absolute():
        return cand
    cwd_cand = (Path.cwd() / cand).resolve()
    if cwd_cand.exists():
        return cwd_cand
    proj_root = Path(anchor_file).resolve().parents[2]
    return (proj_root / cand).resolve()


def _load_or_make_predictions(
    data_dir: str,
    ticker: str,
    model_path_or_dir: str,
    preds_dir: Optional[str] = None,
) -> pd.DataFrame:
    data_dir_p = _resolve_path_smart(data_dir)
    model_p = _resolve_path_smart(model_path_or_dir)
    preds_dir_p = _resolve_path_smart(preds_dir) if preds_dir else (model_p.parent if model_p.is_file() else model_p)

    pred_path = preds_dir_p / f"{ticker.upper()}_pred.csv"
    if not pred_path.exists():
        pred_path = predict_for_ticker(str(data_dir_p), ticker, str(model_p), out_dir=str(preds_dir_p))

    df = pd.read_csv(pred_path)
    # ensure Return_5d present (needed to realize PnL)
    if "Return_5d" not in df.columns:
        orig = pd.read_csv(data_dir_p / f"{ticker.upper()}.csv")
        if "Date" in orig.columns:
            orig["Date"] = pd.to_datetime(orig["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.merge(orig[["Date", "Return_5d"]], on="Date", how="left")
    return df


def backtest_long_step5(
    preds_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float = 10_000.0,
    thr: float = 0.5,
    cost_bp_per_side: float = 0.0,
    horizon_days: int = 5,
    verbose: bool = True,   # << prints per BUY trade
) -> dict:
    """
    Long-only, step-by-5-days:
      - On each entry date, if y_prob >= thr, invest FULL cash for 5 days, earn Return_5d, pay roundtrip costs.
      - Else, stay in cash for that 5-day block.
      - Non-overlapping windows (advance by `horizon_days` each step).
    Prints a line for each BUY trade when verbose=True.
    """
    needed = {"Date", "y_prob", "Return_5d"}
    missing = needed - set(preds_df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    df = preds_df.copy()
    df['Return_5d'] = df['Return_5d']
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    m = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df = df.loc[m].sort_values("Date").reset_index(drop=True)

    if df.empty:
        return {"final_cash": initial_cash, "n_trades": 0, "trades": [], "note": "No rows in the selected period."}

    roundtrip_cost = 2.0 * (cost_bp_per_side / 10_000.0)

    cash = float(initial_cash)
    trades = []
    i, n = 0, len(df)

    while i < n:
        row  = df.iloc[i]
        prob = float(row["y_prob"])
        ret5 = float(row["Return_5d"]) if not pd.isna(row["Return_5d"]) else np.nan
        date = row["Date"].strftime("%Y-%m-%d")

        if np.isnan(ret5):
            trades.append({"EntryDate": date, "Action": "SKIP_NAN_Return_5d", "y_prob": prob, "Cash": cash})
            i += horizon_days
            continue

        if prob >= thr:
            cash_before = cash
            net_ret = ret5 - roundtrip_cost
            cash *= (1.0 + net_ret)

            if verbose:
                print(
                    f"[BUY] {date}  prob={prob:.3f}  ret5={ret5:+.4f}  net={net_ret:+.4f}  "
                    f"cash_before={cash_before:,.2f}  cash_after={cash:,.2f}"
                )

            trades.append({
                "EntryDate": date, "Action": "BUY&SELL_5D", "y_prob": prob,
                "Return_5d": ret5, "NetReturn": net_ret,
                "Cash_before": cash_before, "Cash_after": cash
            })
        else:
            trades.append({"EntryDate": date, "Action": "NO_TRADE", "y_prob": prob, "Cash": cash})

        i += horizon_days

    n_trades = sum(1 for t in trades if t["Action"].startswith("BUY"))
    return {"final_cash": cash, "n_trades": n_trades, "trades": trades}


# -------- regular main (edit & run) --------
if __name__ == "__main__":
    # Paths
    DATA_DIR   = "/Users/nikita/Documents/stock_project_train_data/final"            # per-ticker CSVs (e.g., AAPL.csv)
    MODEL      = "/Users/nikita/Downloads/catboost_model.cbm"  # or a folder containing *.cbm
    PREDS_DIR  = None                                      # None → alongside model; or set a folder path

    # Backtest config
    TICKER       = "AAPL"
    START_DATE   = "2023-01-01"
    END_DATE     = "2024-01-01"
    INITIAL_CASH = 10_000.0
    THRESHOLD    = 0.5
    COST_BP_SIDE = 0.0        # set e.g. 5.0 for 5 bps/side
    HORIZON_DAYS = 5

    preds = _load_or_make_predictions(DATA_DIR, TICKER, MODEL, preds_dir=PREDS_DIR)

    res = backtest_long_step5(
        preds, START_DATE, END_DATE,
        initial_cash=INITIAL_CASH,
        thr=THRESHOLD,
        cost_bp_per_side=COST_BP_SIDE,
        horizon_days=HORIZON_DAYS,
        verbose=True,   # << will print cash before/after each BUY
    )

    print(f"\n== Backtest: {TICKER} {START_DATE}→{END_DATE} ==")
    print(f"Initial cash: {INITIAL_CASH:,.2f}")
    print(f"Final cash:   {res['final_cash']:,.2f}")
    print(f"Trades taken: {res['n_trades']}")

    # Optional: save a trade log
    out_dir = _resolve_path_smart(PREDS_DIR or MODEL)
    if out_dir.is_file():
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{TICKER.upper()}_long5d_trades.csv"
    pd.DataFrame(res["trades"]).to_csv(log_path, index=False)
    print(f"🧾 Trade log → {log_path}")

