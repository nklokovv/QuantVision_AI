import yfinance as yf
import pandas as pd
from pathlib import Path

class YFDownloader:
    def __init__(self, tickers, start, end=None, interval="1d", out_dir="data"):
        """
        tickers : list of str, tickers to download
        start   : str (YYYY-MM-DD), start date
        end     : str (YYYY-MM-DD), optional end date
        interval: str, data interval ("1d","1wk","1mo", etc.)
        out_dir : folder to save CSV files
        """
        self.tickers = [t.upper() for t in tickers]
        self.start = start
        self.end = end
        self.interval = interval
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def download_one(self, ticker):
        """Download data for a single ticker and return DataFrame."""
        df = yf.download(
            ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )
        return df

    def save_one(self, ticker):
        """Download and save one ticker to CSV."""
        df = self.download_one(ticker)
        file_path = self.out_dir / f"{ticker}.csv"
        df.to_csv(file_path)
        print(f"✅ Saved {ticker} to {file_path}")
        return df

    def save_all(self):
        """Download and save all tickers to CSVs. Returns dict of DataFrames."""
        data = {}
        for t in self.tickers:
            data[t] = self.save_one(t)
        return data

# if __name__ == '__main__':
#     tickers = ["TSLA", "XLK"]
#
#     client = YFDownloader(
#         tickers=tickers,
#         start="2020-01-01",
#         end="2025-01-01",
#         interval="1d",
#         out_dir="/Users/nikita/Documents/data"
#     )
#
#     # Save all tickers as CSV
#     data = client.save_all()