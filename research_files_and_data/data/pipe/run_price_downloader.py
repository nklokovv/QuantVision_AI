from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd  # ← needed for post-download normalization
from research_files_and_data.data.download_data import YFDownloader


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_downloaded_csv(path: Path) -> None:
    """
    Fix yfinance 'multi-index' CSVs where first column header == 'Price'
    and first two rows are 'Ticker' and 'Date'. Keep Date as 'YYYY-MM-DD' string.
    Also handles BOM and 'Unnamed: 0'.
    """
    if not path.exists():
        return

    # If file is empty, bail gracefully
    if path.stat().st_size == 0:
        print(f"⚠️  Empty CSV: {path.name}")
        return

    df = pd.read_csv(path)

    # If DataFrame has no columns, bail
    if df.empty and len(df.columns) == 0:
        print(f"⚠️  No columns in CSV: {path.name}")
        return

    # Strip/clean column names + possible BOM on first column
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) > 0:
        first_col_clean = df.columns[0].lstrip("\ufeff")
        if first_col_clean != df.columns[0]:
            df = df.rename(columns={df.columns[0]: first_col_clean})

    # yfinance multi-index case
    if len(df) >= 2 and df.columns[0] == "Price":
        c0 = str(df.iloc[0, 0]).strip().lower()
        c1 = str(df.iloc[1, 0]).strip().lower()
        if c0 == "ticker" and c1 == "date":
            df = df.iloc[2:].copy()               # drop the 2 header rows
            df = df.rename(columns={"Price": "Date"})
        else:
            df = df.rename(columns={"Price": "Date"})  # fallback: still call it 'Date'

    # Some exports keep index as 'Unnamed: 0'
    if "Unnamed: 0" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})

    # Ensure a Date column exists
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    # Normalize Date to 'YYYY-MM-DD' strings (no datetime dtype persists)
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), df["Date"].astype(str).str.slice(0, 10))

    # Drop any leftover header rows that slipped through
    df = df[~df["Date"].str.lower().isin(["date", "ticker"])].reset_index(drop=True)

    # Save back
    df.to_csv(path, index=False)


def run_price_downloader(
    tickers: List[str],
    start_date: str,            # "YYYY-MM-DD"
    end_date: Optional[str],    # "YYYY-MM-DD" or None
    base_dir: str,              # e.g., "/Users/nikita/Documents/stock_project"
    out_subdir: str = "tickers",
    interval: str = "1d",
    include_xlk: bool = False,
    xlk_symbol: str = "XLK",
) -> None:
    """
    Downloads OHLCV price data for the provided tickers (and optionally XLK) from yfinance
    over the specified date range, and writes one CSV per ticker into:
        {base_dir}/{out_subdir}/TICKER.csv

    Notes:
    - Dates are left as 'YYYY-MM-DD' strings after normalization.
    - If end_date is None, downloader will fetch up to the latest available date.
    """
    out_dir = Path(base_dir).expanduser().resolve() / out_subdir
    ensure_dir(out_dir)

    syms = sorted({t.upper() for t in tickers})
    if include_xlk:
        syms.add(xlk_symbol.upper())

    syms = sorted(syms)
    print(f"⬇️  Downloading: {syms}")
    yd = YFDownloader(
        tickers=syms,
        start=start_date,
        end=end_date,
        interval=interval,
        out_dir=str(out_dir),
    )
    yd.save_all()

    # Post-process each downloaded CSV to enforce clean 'Date' column & remove multi-index headers
    for s in syms:
        csv_path = out_dir / f"{s}.csv"
        if csv_path.exists():
            _normalize_downloaded_csv(csv_path)
        else:
            print(f"⚠️  Expected file not found: {csv_path}")

    print(f"✅ Done. Files saved in: {out_dir}")


# ---------------- Example regular run (edit values below) ----------------
if __name__ == "__main__":
    TICKERS     = ["AAPL"]
    START       = "2020-01-01"
    END         = "2021-01-01"        # or None for "up to latest"
    BASE_DIR    = "/Users/nikita/Documents/stock_project11"  # ← change to your path
    SUBDIR      = "tickers"           # output subfolder inside BASE_DIR
    INTERVAL    = "1d"                # "1d" | "1wk" | "1mo"
    INCLUDE_XLK = False               # set True if you also want XLK.csv downloaded
    XLK         = "XLK"

    run_price_downloader(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        base_dir=BASE_DIR,
        out_subdir=SUBDIR,
        interval=INTERVAL,
        include_xlk=INCLUDE_XLK,
        xlk_symbol=XLK,
    )
