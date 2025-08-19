# data/pipe/run_full_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import glob as _glob
import time
import pandas as pd

from research_files_and_data.data.download_data import YFDownloader
from research_files_and_data.data.preprocess_data import FeaturePreprocessor
from research_files_and_data.data.stock_news_downloader import StockNewsFetcher
from research_files_and_data.data.finbert_processor import FinbertSentimentProcessor

def add_sent_5d_avg(df: pd.DataFrame, sentiment_col: str = "DailySent") -> pd.DataFrame:
    if sentiment_col in df.columns:
        df["Sent_5d_Avg"] = df[sentiment_col].rolling(window=5, min_periods=1).mean()
    return df

def add_sentiment_signal(df: pd.DataFrame, sentiment_col: str = "DailySent") -> pd.DataFrame:
    if sentiment_col in df.columns:
        df["SentimentSignal"] = df[sentiment_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = (avg_gain / (avg_loss.replace(0, pd.NA))).fillna(0)
    rsi = 100 - (100 / (1 + rs.replace(0, pd.NA)))
    return rsi

def add_bullish_bearish_rsi(
    df: pd.DataFrame,
    sentiment_col: str = "DailySent",
    period: int = 14,
    bullish_thresh: int = 70,
    bearish_thresh: int = 30,
) -> pd.DataFrame:
    if sentiment_col in df.columns:
        bullish_rsi = compute_rsi(df[sentiment_col].clip(lower=0), period=period)
        bearish_rsi = compute_rsi(df[sentiment_col].clip(upper=0).abs(), period=period)
        df["bullish_sentiment_rsi"] = (bullish_rsi > bullish_thresh).astype(int)
        df["bearish_sentiment_rsi"] = (bearish_rsi < bearish_thresh).astype(int)
    return df

def add_has_sent(df: pd.DataFrame) -> pd.DataFrame:
    """
    HasSent = 1 if there is sentiment this date (DailySent not-NaN OR NewsCount>0 if present), else 0.
    NOTE: We drop 'NewsCount' from the saved CSV afterwards.
    """
    if "DailySent" in df.columns:
        ds = df["DailySent"].notna()
    else:
        ds = pd.Series(False, index=df.index)

    if "NewsCount" in df.columns:
        nc = pd.to_numeric(df["NewsCount"], errors="coerce").fillna(0) > 0
    else:
        nc = pd.Series(False, index=df.index)

    df["HasSent"] = ((ds) | (nc)).astype(int)
    return df


def _yf_dates(start_date: str, end_date: Optional[str]) -> tuple[str, Optional[str]]:
    """
    yfinance's 'end' is EXCLUSIVE. Return (start, end_plus_1_day) so your input window is inclusive.
    """
    s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    e = None if end_date is None else (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return s, e

def _fix_empty_downloads(out_dir: Path, tickers: list[str], start: str, end: Optional[str]) -> None:
    """
    If any downloaded CSV is empty/malformed, re-download that ticker via yfinance (single-thread, with retries)
    and write a clean CSV with a normalized 'Date' column.
    """
    try:
        import yfinance as yf
    except Exception:
        print("ℹ️ yfinance not available for fallback; skipping empty-file fix.")
        return

    for t in sorted(set(tickers)):
        csv_path = out_dir / f"{t}.csv"
        needs_fix = True
        if csv_path.exists():
            try:
                probe = pd.read_csv(csv_path, nrows=2)
                needs_fix = probe.empty or ("Date" not in probe.columns)
            except Exception:
                needs_fix = True

        if not needs_fix:
            continue

        for attempt in range(1, 4):
            try:
                df = yf.download(
                    t, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False, threads=False
                )
                if not df.empty:
                    df = df.reset_index()
                    if "Adj Close" in df.columns:
                        df = df.rename(columns={"Adj Close": "AdjClose"})
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    df.to_csv(csv_path, index=False)
                    print(f"   ⚙️ fixed empty download: {t} ({len(df)} rows)")
                    break
            except Exception:
                pass
            time.sleep(1.2 * attempt)  # backoff


def run_full_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    base_dir: str,
    api_token: str,
    prices_dir: str = "tickers",
    processed_dir: str = "processed",
    news_raw_dir: str = "news_raw",
    news_scored_dir: str = "news_scored",
    final_dir: str = "final",
    xlk_symbol: str = "XLK",
) -> None:
    """
    Writes per-ticker final CSVs WITHOUT 'NewsCount'. Columns may include:
    Date, <price/tech features...>, DailySent, HasSent, Sent_5d_Avg, SentimentSignal, bullish_sentiment_rsi, bearish_sentiment_rsi
    """
    base = Path(base_dir).expanduser().resolve()
    d_prices    = (base / prices_dir).resolve()
    d_processed = (base / processed_dir).resolve()
    d_news_raw  = (base / news_raw_dir).resolve()
    d_news_sc   = (base / news_scored_dir).resolve()
    d_final     = (base / final_dir).resolve()
    for d in [d_prices, d_processed, d_news_raw, d_news_sc, d_final]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) Prices (tickers + XLK) ----------------
    all_syms = sorted({*(s.upper() for s in tickers), xlk_symbol.upper()})
    s, e = _yf_dates(start_date, end_date)  # make end inclusive for yfinance
    print(f"⬇️  Downloading prices for: {all_syms}  range=[{s} → {e or 'latest'})")

    YFDownloader(tickers=all_syms, start=s, end=e, interval="1d", out_dir=str(d_prices)).save_all()

    # Repair any empty/malformed downloads (common with very recent windows)
    _fix_empty_downloads(d_prices, all_syms, start=s, end=e)

    xlk_csv = d_prices / f"{xlk_symbol.upper()}.csv"

    # ---------------- 2) Preprocess + merge with XLK ----------------
    print("🧮 Preprocessing and merging with XLK features…")
    FeaturePreprocessor(tickers_folder=str(d_prices), xlk_csv_path=str(xlk_csv), out_dir=str(d_processed)).process_all()

    # ---------------- 3) News download ----------------
    print("📰 Fetching news per ticker…")
    StockNewsFetcher(api_token, d_news_raw).fetch_all(tickers, start_date=start_date, end_date=end_date or start_date)

    # ---------------- 4) FinBERT scoring + daily aggregation ----------------
    print("🧠 Applying FinBERT on news and aggregating daily sentiment…")
    fp = FinbertSentimentProcessor()
    fp.process_dir(d_news_raw, out_dir=d_news_sc)  # per-article scores
    for f in sorted(d_news_sc.glob("*.csv")):
        if f.name.lower().endswith("_daily.csv"):
            continue
        try:
            daily_path = fp.aggregate_daily(f)  # writes alongside
            print(f"   Daily aggregated → {daily_path.name}")
        except Exception as e:
            print(f"   ⚠️ Skipped daily aggregation for {f.name}: {e}")

    def _latest_daily_for_ticker(t: str) -> Optional[Path]:
        pats = [
            str(d_news_sc / f"{t.upper()}_*_daily.csv"),
            str(d_news_sc / f"*{t.upper()}*_daily.csv"),
        ]
        matches: List[str] = []
        for p in pats:
            matches.extend(_glob.glob(p))
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(matches[0])

    # ---------------- 5) Merge per ticker & add features (drop NewsCount before saving) ----------------
    for t in sorted({s.upper() for s in tickers}):
        proc_csv = d_processed / f"{t}.csv"
        if not proc_csv.exists():
            print(f"   ⚠️ Missing processed CSV for {t}: {proc_csv}")
            continue

        price_df = pd.read_csv(proc_csv)
        if "Date" not in price_df.columns:
            print(f"   ⚠️ 'Date' missing in {proc_csv.name}, skipping {t}")
            continue
        price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        daily_path = _latest_daily_for_ticker(t)
        if daily_path is None:
            final_df = price_df.copy()
            final_df["HasSent"] = 0
        else:
            sent_df = pd.read_csv(daily_path)
            if "Date" in sent_df.columns:
                sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            keep_cols = [c for c in ["Date", "DailySent", "NewsCount"] if c in sent_df.columns]
            sent_df = sent_df[keep_cols].copy() if keep_cols else pd.DataFrame(columns=["Date"])
            final_df = price_df.merge(sent_df, on="Date", how="left")

            # HasSent first
            final_df = add_has_sent(final_df)

            # Sentiment-derived features (no NewsCount needed)
            if "DailySent" in final_df.columns:
                final_df = add_sent_5d_avg(final_df, "DailySent")
                final_df = add_sentiment_signal(final_df, "DailySent")
                final_df = add_bullish_bearish_rsi(final_df, "DailySent", period=14)

            # drop NewsCount from output if it came through
            if "NewsCount" in final_df.columns:
                final_df = final_df.drop(columns=["NewsCount"])

        out_path = d_final / f"{t}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved final per-ticker CSV → {out_path}")

    print("\n🏁 Pipeline complete.")


# ------------------------------ Regular main (edit & run) ------------------------------
if __name__ == "__main__":
    TICKERS   = ["AAPL"]
    START     = "2025-07-01"
    END       = "2025-08-18"      # or None
    BASE_DIR  = "/Users/nikita/Documents"  # project-relative
    API_TOKEN = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"

    run_full_pipeline(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        base_dir=BASE_DIR,
        api_token=API_TOKEN,
        prices_dir="tickers",
        processed_dir="processed",
        news_raw_dir="news_raw",
        news_scored_dir="news_scored",
        final_dir="final",
        xlk_symbol="XLK",
    )
