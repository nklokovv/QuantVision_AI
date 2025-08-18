
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_csv_smart(path: Path, ticker_from_name: Optional[str]) -> pd.DataFrame:
    """Read a CSV and normalize basic columns."""
    df = pd.read_csv(path, low_memory=False)
    if "Ticker" not in df.columns and ticker_from_name:
        df.insert(0, "Ticker", ticker_from_name.upper())

    # Normalize Date
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), df["Date"].astype(str).str[:10])

    # Standardize Ticker casing
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    return df


def analyze_final_data(data_dir: str, out_dir: Optional[str] = None, save_csv: bool = True) -> dict:
    """
    Load all per‑ticker CSVs from a folder, print useful diagnostics,
    and optionally save summary CSVs. Returns a small dict of headline stats.
    """
    p = Path(data_dir).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {p}")

    files = sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {p}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        ticker = f.stem.split(".")[0].upper()
        try:
            df = _read_csv_smart(f, ticker)
            df["__sourcefile"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Failed to read {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No readable CSVs.")

    full = pd.concat(dfs, ignore_index=True)
    print(f"\n📦 Loaded {len(files)} files → {len(full):,} rows, {len(full.columns)} columns")
    if "Ticker" not in full.columns:
        raise RuntimeError("No 'Ticker' column found after loading.")

    # Basic expectations
    target_col = "y_5d" if "y_5d" in full.columns else None
    price_cols = [c for c in ["Adj Close", "Close", "Open", "High", "Low"] if c in full.columns]

    # Per-ticker stats
    grp = full.groupby("Ticker", dropna=False, sort=True)
    per_ticker_rows = grp.size().rename("rows")
    per_ticker_start = grp["Date"].min() if "Date" in full.columns else pd.Series(dtype=object)
    per_ticker_end = grp["Date"].max() if "Date" in full.columns else pd.Series(dtype=object)

    per_ticker = pd.concat([per_ticker_rows, per_ticker_start.rename("start"), per_ticker_end.rename("end")], axis=1)
    if "Date" in full.columns:
        per_ticker["days_span"] = (
            pd.to_datetime(per_ticker["end"]) - pd.to_datetime(per_ticker["start"])
        ).dt.days

    print("\n🧾 Tickers loaded:", ", ".join(per_ticker.index.tolist()[:20]),
          ("..." if len(per_ticker) > 20 else ""))
    print(per_ticker.sort_values("rows", ascending=False).head(10).to_string())

    # Duplicates by (Ticker, Date)
    if "Date" in full.columns:
        dup_mask = full.duplicated(subset=["Ticker", "Date"], keep=False)
        dup_cnt = int(dup_mask.sum())
        print(f"\n🔎 Duplicates (Ticker+Date): {dup_cnt}")
        if dup_cnt > 0:
            print(full.loc[dup_mask, ["Ticker", "Date", "__sourcefile"]].head(10).to_string())

    # Missingness
    miss = full.isna().mean().sort_values(ascending=False)
    print("\n🕳️  Missingness by column (top 15):")
    print((miss.head(15) * 100).round(2).astype(str) + "%")

    # Optional sentiment presence
    cand_sent = [c for c in full.columns if "sent" in c.lower()]
    if cand_sent:
        print("\n📰 Detected sentiment-related columns:", ", ".join(cand_sent[:30]))

    # Target distribution
    if target_col is not None:
        vc = full[target_col].value_counts(dropna=False)
        vcp = (vc / vc.sum() * 100).round(2)
        print(f"\n🎯 Target '{target_col}' distribution (overall):")
        for k, v in vc.items():
            print(f"  class {k}: {v} rows ({vcp[k]}%)")

        per_t_bal = (
            full.groupby("Ticker")[target_col]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .rename(columns=lambda c: f"p_class_{c}")
        )
        per_t_cnt = full.groupby("Ticker")[target_col].size().rename("count")
        bal = per_t_bal.join(per_t_cnt)
        print("\n🎯 Target balance per ticker (head):")
        print(bal.head(10).to_string())

        # Flag severe imbalance
        if {"p_class_0", "p_class_1"}.issubset(bal.columns):
            severe = bal[(bal["p_class_0"] < 0.1) | (bal["p_class_1"] < 0.1)]
            if len(severe) > 0:
                print("\n⚠️  Severely imbalanced tickers (one class < 10%):")
                print(severe.sort_values("count", ascending=False).head(20).to_string())

    # Feature space
    numeric_dtypes = ("int16","int32","int64","float16","float32","float64")
    drop_cols = {"Date", "__sourcefile"}
    if target_col: drop_cols.add(target_col)
    drop_cols.update({"Return_5d"})  # no 'fwd_ret_5d' here per user's data
    feature_cols = [c for c in full.columns if c not in drop_cols and str(full[c].dtype) in numeric_dtypes]
    print(f"\n🧮 Numeric feature columns: {len(feature_cols)}")
    print(", ".join(feature_cols[:25]) + (" ..." if len(feature_cols) > 25 else ""))

    # Correlation with target (point-biserial via Pearson on 0/1)
    if target_col and feature_cols:
        y = full[target_col].astype(float)
        corr = {}
        for c in feature_cols:
            try:
                corr[c] = float(pd.Series(full[c]).astype(float).corr(y))
            except Exception:
                corr[c] = np.nan
        corr_s = pd.Series(corr).dropna().sort_values(key=lambda x: x.abs(), ascending=False)
        print("\n📈 Top 10 |corr(feature, y_5d)|:")
        print(corr_s.head(10).round(4).to_string())

    # Suspicious / leakage-prone names (generic)
    sus_substrings = ["fwd", "future", "lead", "ahead", "t+1", "t+2", "target"]
    sus = [c for c in full.columns if any(s in c.lower() for s in sus_substrings)]
    if sus:
        print("\n🛑 Potential leakage columns (by name):", ", ".join(sus))

    # Monotonic date check per ticker
    if "Date" in full.columns:
        issues = []
        for t, g in full.groupby("Ticker"):
            dt = pd.to_datetime(g["Date"], errors="coerce")
            if dt.is_monotonic_increasing is False:
                issues.append(t)
        if issues:
            print("\n⚠️  Non‑monotonic Date order in tickers:", ", ".join(issues[:20]),
                  ("..." if len(issues) > 20 else ""))

        # Common overlap window
        span = grp["Date"].agg(["min", "max"])
        common_start = pd.to_datetime(span["min"]).max()
        common_end = pd.to_datetime(span["max"]).min()
        if pd.notna(common_start) and pd.notna(common_end) and common_start <= common_end:
            days = (common_end - common_start).days
            print(f"\n🪄 Common overlap window: {common_start.date()} → {common_end.date()} ({days} days)")
        else:
            print("\nℹ️  No common overlap window across all tickers.")

    # Save CSV outputs
    odir = None
    if save_csv:
        odir = _ensure_dir(out_dir or (p / "_analysis"))
        per_ticker.to_csv(odir / "per_ticker_stats.csv")
        miss.rename("missing_rate").to_frame().to_csv(odir / "missing_by_column.csv")
        if target_col is not None:
            bal.to_csv(odir / "class_balance_per_ticker.csv")
        if target_col and feature_cols:
            corr_s.rename("corr_with_y").to_frame().to_csv(odir / "feature_correlations.csv")
        print(f"\n💾 Saved CSVs to: {odir}")

    # Return a compact stats dict
    return {
        "n_files": len(files),
        "n_rows": int(len(full)),
        "n_cols": int(len(full.columns)),
        "tickers": sorted(full["Ticker"].unique().tolist()),
        "out_dir": str(odir) if odir else None,
        "target_present": bool(target_col is not None),
        "n_features_numeric": len(feature_cols),
    }


if __name__ == "__main__":
    # <<< EDIT THESE >>>
    DATA_DIR = "/Users/nikita/Documents/final_project_data/sentiment_features_cleaned"
    OUT_DIR = "/Users/nikita/Documents/final_project_data"       # or set to a path, e.g. "/Users/nikita/Documents/stock_project_train_data/report"
    SAVE_CSV = True      # set False to only print and not save

    stats = analyze_final_data(DATA_DIR, out_dir=OUT_DIR, save_csv=SAVE_CSV)
    print("\nSummary:", stats)
