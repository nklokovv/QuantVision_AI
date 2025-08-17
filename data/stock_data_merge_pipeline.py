import os
import time
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from download_data import YFDownloader
from stock_news_downloader import StockNewsFetcher


# --- Yahoo Finance Download ---
def run_yahoo_finance_download(tickers, start_date, end_date, out_dir):
    """
    Download stock data using YFDownloader.
    Each ticker's data will be saved as a CSV in the specified output directory.
    """
    print("▶ Starting Yahoo Finance download …")
    downloader = YFDownloader(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        out_dir=out_dir
    )
    data = downloader.save_all()
    print("✅ Yahoo Finance download completed.\n")
    return data


# --- Stock News Fetching ---
def run_stock_news_fetch(api_token, tickers, start_date, end_date, news_out_dir):
    """
    Fetch stock news using StockNewsFetcher and save one CSV per ticker.
    """
    print("▶ Fetching stock news …")
    news_fetcher = StockNewsFetcher(api_token, Path(news_out_dir))
    files = news_fetcher.fetch_all(tickers, start_date, end_date)
    print("✅ Stock news fetch completed.\n")
    return files


# --- Additional Sentiment Functions ---
def add_sent_5d_avg(df, sentiment_col="DailySent"):
    df["Sent_5d_Avg"] = df[sentiment_col].rolling(window=5, min_periods=1).mean()
    return df


def add_sentiment_signal(df, sentiment_col="DailySent"):
    df["SentimentSignal"] = df[sentiment_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_bullish_bearish_rsi(df, sentiment_col="DailySent", period=14):
    df["bullish_sentiment_rsi"] = compute_rsi(df[sentiment_col].clip(lower=0), period=period)
    df["bearish_sentiment_rsi"] = compute_rsi(df[sentiment_col].clip(upper=0).abs(), period=period)
    return df


# --- XLK Data Download ---
def download_xlk_data(xlk_csv_path, start_date, end_date):
    """
    Check for the XLK CSV file; if not present, download XLK data.
    """
    xlk_path = Path(xlk_csv_path)
    if not xlk_path.exists():
        print(f"▶ XLK CSV not found at {xlk_csv_path}. Downloading XLK data …")
        downloader = YFDownloader(
            tickers=["XLK"],
            start=start_date,
            end=end_date,
            interval="1d",
            out_dir=xlk_path.parent
        )
        downloader.save_all()  # XLK.csv will be saved in the specified folder
        print(f"✅ XLK data downloaded to {xlk_csv_path}\n")
    else:
        print(f"✅ XLK CSV found at {xlk_csv_path}.\n")


# --- Updated Merge Pipeline ---
def run_merge_pipeline(finbert_input_path, santiment_csv_path, xlk_csv_path, output_dir):
    """
    Merge the per-ticker FinBERT CSV files with the Santiment CSV based on both 'ticker' and 'Date'.
    A new binary column 'has_sent' is added:
      - 1 if there is a corresponding Santiment record on that date
      - 0 if there is no record for that date.

    Additional sentiment features are computed afterward, XLK data is merged on Date,
    and finally each ticker's merged data is saved into its own CSV file.
    """
    print("▶ Starting merge pipeline …")
    try:
        # Check if finbert_input_path is a directory or a file
        if os.path.isdir(finbert_input_path):
            csv_files = glob.glob(os.path.join(finbert_input_path, "*.csv"))
            if not csv_files:
                print(f"❌ No CSV files found in directory: {finbert_input_path}")
                return
            finbert_df_list = [pd.read_csv(f) for f in csv_files]
            finbert_df = pd.concat(finbert_df_list, ignore_index=True)
        else:
            finbert_df = pd.read_csv(finbert_input_path)
        santiment_df = pd.read_csv(santiment_csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return

    # Set flag for sentiment events in Santiment data.
    santiment_df["has_sent"] = 1

    # Merge FinBERT with Santiment data based on 'ticker' and 'Date'
    merged_df = pd.merge(
        finbert_df,
        santiment_df[["ticker", "Date", "santiment_grade", "has_sent", "DailySent"]],
        on=["ticker", "Date"],
        how="left"
    )
    merged_df["has_sent"] = merged_df["has_sent"].fillna(0).astype(int)

    if "DailySent" in merged_df.columns:
        merged_df = add_sent_5d_avg(merged_df, sentiment_col="DailySent")
        merged_df = add_sentiment_signal(merged_df, sentiment_col="DailySent")
        merged_df = add_bullish_bearish_rsi(merged_df, sentiment_col="DailySent", period=14)
    else:
        print("⚠ 'DailySent' column not found in merged data. Skipping additional sentiment features.")

    # Load XLK data and merge on Date
    try:
        xlk_df = pd.read_csv(xlk_csv_path)
        merged_df = pd.merge(merged_df, xlk_df, on="Date", how="left")
    except Exception as e:
        print(f"⚠ Issue reading or merging XLK data: {e}")

    cols = [
        "Date", "ticker", "santiment_grade", "has_sent", "DailySent", "Sent_5d_Avg", "SentimentSignal",
        "bullish_sentiment_rsi", "bearish_sentiment_rsi",
        "Volume", "SMA_5", "EMA_12", "EMA_26", "RSI_14", "MACD",
        "BB_upper", "BB_lower", "Momentum_10", "Volatility_10d", "Return_5d",
        "SMA_5_XLK", "EMA_12_XLK", "EMA_26_XLK", "RSI_14_XLK", "MACD_XLK",
        "BB_upper_XLK", "BB_lower_XLK", "Momentum_10_XLK", "Volatility_10d_XLK"
    ]
    for c in cols:
        if c not in merged_df.columns:
            merged_df[c] = np.nan

    out = merged_df[cols].copy()
    out = out.dropna().reset_index(drop=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tickers_list = out["ticker"].unique()
    for ticker in tickers_list:
        ticker_df = out[out["ticker"] == ticker]
        output_file = os.path.join(output_dir, f"{ticker}_final.csv")
        try:
            ticker_df.to_csv(output_file, index=False)
            print(f"✅ Saved merged data for {ticker} to {output_file}")
        except Exception as e:
            print(f"❌ Error saving file for {ticker}: {e}")
    print("✅ Merge pipeline completed.\n")


def main():
    # --- Configuration ---
    tickers = ["AAPL"]
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    # Stock data (each ticker CSV and XLK) will be saved under this folder.
    stock_data_dir = "/Users/nikita/Documents/data/tickers"
    # News data folder.
    news_data_dir = "/Users/nikita/Documents/data/news"
    # FinBERT-processed CSVs are in the after_finbert folder.
    finbert_dir = "/Users/nikita/Documents/data/after_finbert"
    # Final merged per-ticker outputs will be saved here.
    merge_output_dir = "/Users/nikita/Documents/data/final_data"

    # Previously, we assumed an aggregated file named 'daily_finbert.csv'.
    # Now, we pass the directory with the per-ticker CSVs.
    finbert_input = finbert_dir  # updated to pass directory instead of a file

    # Santiment CSV (aggregated from FinBERT news or other sources)
    santiment_csv_path = "/Users/nikita/Documents/data/santiment.csv"
    # XLK CSV under the stock data directory.
    xlk_csv_path = os.path.join(stock_data_dir, "XLK.csv")

    # API token for StockNewsFetcher.
    api_token = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"

    # --- Pipeline Execution Steps ---
    # 1. Download stock data (per ticker) using Yahoo Finance.
    run_yahoo_finance_download(tickers, start_date, end_date, stock_data_dir)

    # 2. Fetch stock news.
    run_stock_news_fetch(api_token, tickers, start_date, end_date, news_data_dir)

    # 3. Download XLK data if not already present (in the tickers folder).
    download_xlk_data(xlk_csv_path, start_date, end_date)

    # 4. Merge FinBERT (per-ticker CSVs), Santiment, and XLK features,
    #    and save a merged CSV per ticker into the final_data folder.
    run_merge_pipeline(finbert_input, santiment_csv_path, xlk_csv_path, merge_output_dir)

if __name__ == "__main__":
    main()