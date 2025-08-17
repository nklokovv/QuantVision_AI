from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import glob as _glob
from data.download_data import YFDownloader
from data.preprocess_data import FeaturePreprocessor
from data.stock_news_downloader import StockNewsFetcher
from data.finbert_processor import FinbertSentimentProcessor

# ---------- Sentiment feature helpers ----------
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

def add_bullish_bearish_rsi(
    df: pd.DataFrame,
    sentiment_col: str = "DailySent",
    period: int = 14,
    bullish_thresh: int = 70,
    bearish_thresh: int = 30,
) -> pd.DataFrame:
    # Compute RSI separately for positive and negative sentiment
    bullish_rsi = compute_rsi(df[sentiment_col].clip(lower=0), period=period)
    bearish_rsi = compute_rsi(df[sentiment_col].clip(upper=0).abs(), period=period)

    # Turn into binary signals
    df["bullish_sentiment_signal"] = (bullish_rsi > bullish_thresh).astype(int)
    df["bearish_sentiment_signal"] = (bearish_rsi < bearish_thresh).astype(int)

    return df

# ---------- Pipeline ----------
def run_full_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    base_dir: str,
    api_token: str,
    # subfolders under base_dir (can be absolute paths too)
    prices_dir: str = "tickers",
    processed_dir: str = "processed",
    news_raw_dir: str = "news_raw",
    news_scored_dir: str = "news_scored",
    final_dir: str = "final",
    xlk_symbol: str = "XLK",
):
    """
    Orchestrates the entire workflow end-to-end.
    Paths:
      base_dir/prices_dir      : raw yfinance csvs for tickers + XLK
      base_dir/processed_dir   : per-ticker engineered price features merged with XLK features
      base_dir/news_raw_dir    : per-ticker raw StockNewsAPI csvs
      base_dir/news_scored_dir : per-ticker FinBERT-scored csvs (+ *_daily.csv created alongside)
      base_dir/final_dir       : final per-ticker merged csvs
    """
    base = Path(base_dir)
    d_prices    = (base / prices_dir).resolve()
    d_processed = (base / processed_dir).resolve()
    d_news_raw  = (base / news_raw_dir).resolve()
    d_news_sc   = (base / news_scored_dir).resolve()
    d_final     = (base / final_dir).resolve()

    for d in [d_prices, d_processed, d_news_raw, d_news_sc, d_final]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Download prices (tickers + XLK)
    all_syms = sorted(set([s.upper() for s in tickers] + [xlk_symbol.upper()]))
    print(f"⬇️  Downloading prices for: {all_syms}")
    yd = YFDownloader(tickers=all_syms, start=start_date, end=end_date, interval="1d", out_dir=str(d_prices))
    yd.save_all()
    xlk_csv = d_prices / f"{xlk_symbol.upper()}.csv"

    # 2) Preprocess per ticker & merge with XLK engineered features
    print("🧮 Preprocessing and merging with XLK features…")
    pre = FeaturePreprocessor(
        tickers_folder=str(d_prices),
        xlk_csv_path=str(xlk_csv),
        out_dir=str(d_processed)
    )
    pre.process_all()

    # 3) Download news per ticker
    print("📰 Fetching news per ticker…")
    sn = StockNewsFetcher(api_token, d_news_raw)
    sn.fetch_all(tickers, start_date=start_date, end_date=end_date or start_date)

    # 4) Apply FinBERT & aggregate daily
    print("🧠 Applying FinBERT on news and aggregating daily sentiment…")
    fp = FinbertSentimentProcessor()
    fp.process_dir(d_news_raw, out_dir=d_news_sc)  # saves per-article scored CSVs

    for f in sorted(d_news_sc.glob("*.csv")):
        name = f.name.lower()
        if name.endswith("_daily.csv"):
            # already aggregated → skip
            continue
        try:
            daily_path = fp.aggregate_daily(f)  # writes next to the scored file
            print(f"   Daily aggregated → {daily_path.name}")
        except Exception as e:
            print(f"   ⚠️ Skipped daily aggregation for {f.name}: {e}")
    # 5) For each ticker, merge processed price features with daily sentiment
    def _find_latest_daily_for_ticker(t: str) -> Optional[Path]:
        # We look for files like {TICKER}_*_daily.csv or scored names turned into *_daily.csv
        pats = [
            str(d_news_sc / f"{t.upper()}_*_daily.csv"),
            str(d_news_sc / f"*{t.upper()}*_daily.csv"),
        ]
        matches = []

        for p in pats:
            matches.extend(_glob.glob(p))
        if not matches:
            return None
        # choose the most recently modified
        matches = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(matches[0])

    for t in tickers:
        t_upper = t.upper()
        proc_csv = d_processed / f"{t_upper}.csv"
        if not proc_csv.exists():
            print(f"   ⚠️ Processed CSV missing for {t_upper}: {proc_csv}")
            continue

        # Load processed price features
        price_df = pd.read_csv(proc_csv)
        # Ensure Date format remains 'YYYY-MM-DD'
        if "Date" not in price_df.columns:
            print(f"   ⚠️ 'Date' column missing in {proc_csv.name}, skipping")
            continue
        # 6) Attach sentiment
        daily_path = _find_latest_daily_for_ticker(t_upper)
        if daily_path is None:
            print(f"   ⚠️ No daily sentiment file for {t_upper}; writing price-only final CSV")
            final_df = price_df.copy()
        else:
            sent_df = pd.read_csv(daily_path)
            # Expect columns: Date, DailySent (+maybe NewsCount)
            # Keep Date as string; merge on Date
            keep_cols = [c for c in ["Date", "DailySent", "NewsCount"] if c in sent_df.columns]
            sent_df = sent_df[keep_cols].copy()
            final_df = price_df.merge(sent_df, on="Date", how="left")

            # 7) Add sentiment-based features
            if "DailySent" in final_df.columns:
                final_df = add_sent_5d_avg(final_df, "DailySent")
                final_df = add_sentiment_signal(final_df, "DailySent")
                final_df = add_bullish_bearish_rsi(final_df, "DailySent", period=14)
            else:
                print(f"   ⚠️ 'DailySent' not found for {t_upper}; skipping sentiment features")

        out_path = (base / final_dir) / f"{t_upper}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved final per-ticker CSV → {out_path}")

    print("\n🏁 Pipeline complete.")

if __name__ == "__main__":
    # Example usage (edit as needed)
    TICKERS   = ["AAPL"]
    START     = "2020-01-01"
    END       = "2021-01-01"
    BASE_DIR  = "/Users/nikita/Documents/stock_project"
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
