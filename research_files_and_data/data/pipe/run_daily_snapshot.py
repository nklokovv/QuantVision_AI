
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd

from research_files_and_data.data.download_data import YFDownloader
from research_files_and_data.data.preprocess_data import FeaturePreprocessor
from research_files_and_data.data.stock_news_downloader import StockNewsFetcher
from research_files_and_data.data.finbert_processor import FinbertSentimentProcessor


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def add_sent_5d_avg(df: pd.DataFrame, sentiment_col: str = "DailySent") -> pd.DataFrame:
    df["Sent_5d_Avg"] = df[sentiment_col].rolling(window=5, min_periods=1).mean()
    return df


def add_sentiment_signal(df: pd.DataFrame, sentiment_col: str = "DailySent") -> pd.DataFrame:
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


def add_bullish_bearish_rsi_binary(
    df: pd.DataFrame,
    sentiment_col: str = "DailySent",
    period: int = 14,
    bullish_thresh: int = 70,
    bearish_thresh: int = 30,
) -> pd.DataFrame:
    pos = compute_rsi(df[sentiment_col].clip(lower=0), period=period)
    neg = compute_rsi(df[sentiment_col].clip(upper=0).abs(), period=period)
    df["bullish_sentiment_signal"] = (pos > bullish_thresh).astype(int)
    df["bearish_sentiment_signal"] = (neg < bearish_thresh).astype(int)
    return df


def run_daily_snapshot(
    tickers: List[str],
    date: str,  # YYYY-MM-DD target trading day
    base_dir: str,
    api_token: str,
    xlk_symbol: str = "XLK",
    lookback_days: int = 60,      # for price features (RSI_14, SMA_5, etc.)
    news_lookback_days: int = 7,  # for 5d sentiment features
) -> Path:
    """
    Produces one CSV with a single row per ticker for the target date.

    Output path: {base_dir}/daily_snapshot/daily_snapshot_{date}.csv
    """
    target_dt = _parse_date(date)
    start_prices_dt = target_dt - timedelta(days=lookback_days)
    start_news_dt = target_dt - timedelta(days=news_lookback_days)

    base = Path(base_dir)
    d_prices    = (base / "tickers").resolve()
    d_processed = (base / "processed").resolve()
    d_news_raw  = (base / "news_raw").resolve()
    d_news_sc   = (base / "news_scored").resolve()
    d_tmp_final = (base / "final_tmp").resolve()   # temp per-ticker merge before snapshot
    d_out       = (base / "daily_snapshot").resolve()

    _ensure_dirs(d_prices, d_processed, d_news_raw, d_news_sc, d_tmp_final, d_out)

    # 1) Download prices for tickers + XLK with a lookback window
    all_syms = sorted(set([s.upper() for s in tickers] + [xlk_symbol.upper()]))
    yd = YFDownloader(
        tickers=all_syms,
        start=_date_str(start_prices_dt),
        end=_date_str(target_dt + timedelta(days=1)),  # inclusive window for most providers
        interval="1d",
        out_dir=str(d_prices),
    )
    yd.save_all()
    xlk_csv = d_prices / f"{xlk_symbol.upper()}.csv"

    # 2) Preprocess per ticker with XLK merge
    pre = FeaturePreprocessor(
        tickers_folder=str(d_prices),
        xlk_csv_path=str(xlk_csv),
        out_dir=str(d_processed),
    )
    pre.process_all()

    # 3) Download news for a short lookback window up to target date (inclusive)
    sn = StockNewsFetcher(api_token, d_news_raw)
    sn.fetch_all(
        tickers,
        start_date=_date_str(start_news_dt),
        end_date=_date_str(target_dt),
    )

    # 4) Run FinBERT scoring and daily aggregation
    fp = FinbertSentimentProcessor()
    fp.process_dir(d_news_raw, out_dir=d_news_sc)
    for f in sorted(d_news_sc.glob("*.csv")):
        name = f.name.lower()
        if name.endswith("_daily.csv"):
            continue
        try:
            fp.aggregate_daily(f)
        except Exception as e:
            print(f"⚠️  Daily aggregation skipped for {f.name}: {e}")

    # helper to find most recently modified *_daily.csv for a ticker
    def _find_latest_daily(t: str) -> Optional[Path]:
        import glob as _glob
        pats = [
            str(d_news_sc / f"{t.upper()}_*_daily.csv"),
            str(d_news_sc / f"*{t.upper()}*_daily.csv"),
        ]
        matches = []
        for p in pats:
            matches.extend(_glob.glob(p))
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(matches[0])

    # 5) Merge per-ticker processed features with daily sentiment, then slice the target date
    rows = []
    for t in tickers:
        tU = t.upper()
        proc_csv = d_processed / f"{tU}.csv"
        if not proc_csv.exists():
            print(f"⚠️ Processed CSV missing for {tU}")
            continue

        price_df = pd.read_csv(proc_csv)
        if "Date" not in price_df.columns:
            print(f"⚠️ 'Date' missing in {proc_csv.name}")
            continue

        # attach sentiment if available
        daily_path = _find_latest_daily(tU)
        if daily_path is not None:
            sent_df = pd.read_csv(daily_path)
            keep_cols = [c for c in ["Date", "DailySent", "NewsCount"] if c in sent_df.columns]
            sent_df = sent_df[keep_cols].copy()
            merged = price_df.merge(sent_df, on="Date", how="left")
            if "DailySent" in merged.columns:
                merged = add_sent_5d_avg(merged, "DailySent")
                merged = add_sentiment_signal(merged, "DailySent")
                merged = add_bullish_bearish_rsi_binary(merged, "DailySent", period=14)
                nc = merged["NewsCount"] if "NewsCount" in merged.columns else pd.Series(0, index=merged.index)
                merged["hassent"] = ((merged["DailySent"].notna()) | (nc.fillna(0) > 0)).astype(int)
            else:
                merged["hassent"] = 0
        else:
            merged = price_df.copy()
            merged["hassent"] = 0

        # select the target date; if missing (holiday), choose the last available before target
        merged = merged.sort_values("Date")
        snap = merged[merged["Date"] == _date_str(target_dt)]
        if snap.empty:
            # pick last available row before (or equal) target date
            snap = merged[merged["Date"] <= _date_str(target_dt)].tail(1)
        if snap.empty:
            print(f"⚠️ No data for {tU} up to {date}")
            continue

        snap = snap.copy()
        snap.insert(0, "Ticker", tU)
        rows.append(snap.iloc[0])

    if not rows:
        raise RuntimeError("No snapshot rows produced. Check tickers/date inputs.")

    out_df = pd.DataFrame(rows)
    out_path = d_out / f"daily_snapshot_{date}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved daily snapshot → {out_path}")
    return out_path


if __name__ == "__main__":
    # Example usage. Edit inline or call run_daily_snapshot() from another script.
    TICKERS   = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AMD", "INTC"]
    DATE      = "2025-08-15"
    BASE_DIR  = "/Users/nikita/Documents/stock_project_for_model1"
    API_TOKEN = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"

    run_daily_snapshot(
        tickers=TICKERS,
        date=DATE,
        base_dir=BASE_DIR,
        api_token=API_TOKEN,
        xlk_symbol="XLK",
        lookback_days=60,
        news_lookback_days=7,
    )
