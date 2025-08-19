import os
from pathlib import Path
import pandas as pd
import numpy as np

class FeaturePreprocessor:
    """
    Preprocess per-ticker CSVs + XLK index CSV, engineer features, and save per-ticker outputs.
    Output columns (per ticker):
    Date, Volume, SMA_5, EMA_12, EMA_26, RSI_14, MACD, BB_upper, BB_lower,
    Momentum_10, Volatility_10d,
    SMA_5_XLK, EMA_12_XLK, EMA_26_XLK, RSI_14_XLK, MACD_XLK, BB_upper_XLK,
    BB_lower_XLK, Momentum_10_XLK, Volatility_10d_XLK,
    Return_5d
    """

    def __init__(
        self,
        tickers_folder: str,
        xlk_csv_path: str,
        out_dir: str = "processed",
        price_col_preference=("Adj Close", "Close"),
        bb_window: int = 20,
        bb_std: float = 2.0,
    ):
        self.tickers_folder = Path(tickers_folder)
        self.xlk_csv_path = Path(xlk_csv_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.price_col_pref = price_col_preference
        self.bb_window = bb_window
        self.bb_std = bb_std

        # Preload and feature-ize XLK
        self.xlk_df = self._load_csv(self.xlk_csv_path)
        self.xlk_df = self._engineer_features(self.xlk_df, suffix="_XLK")
        self.xlk_df = self.xlk_df[
            ["Date",
             "SMA_5_XLK","EMA_12_XLK","EMA_26_XLK","RSI_14_XLK","MACD_XLK",
             "BB_upper_XLK","BB_lower_XLK","Momentum_10_XLK","Volatility_10d_XLK"]
        ]

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """
        Load yfinance CSVs (including the 'Price/Ticker/Date' header style),
        drop first 2 rows, and keep Date as plain 'YYYY-MM-DD' strings.
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)

        # Handle yfinance multi-index CSVs where first col = 'Price', row0='Ticker', row1='Date'
        if df.columns[0] == "Price" and len(df) >= 2:
            if str(df.iloc[0, 0]).strip().lower() == "ticker" and str(df.iloc[1, 0]).strip().lower() == "date":
                df = df.iloc[2:].copy()  # drop first 2 rows
                df = df.rename(columns={"Price": "Date"})  # rename first col to Date
            else:
                df = df.rename(columns={"Price": "Date"})  # fallback rename

        # Ensure Date column exists
        if "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Date"})

        # Drop fully empty rows
        df = df.dropna(how="all")

        # Sort by Date as strings (they are already YYYY-MM-DD so lexicographic = chronological)
        df = df.sort_values("Date").reset_index(drop=True)

        # Ensure correct dtypes for numeric columns
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def _pick_price_col(self, df: pd.DataFrame) -> str:
        for c in self.price_col_pref:
            if c in df.columns:
                return c
        # fallback
        for c in ["Close", "AdjClose", "Adj_Close", "Adj close"]:
            if c in df.columns:
                return c
        raise ValueError("No suitable price column found (looked for Adj Close/Close variants).")

    def _rsi(self, s: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI."""
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Inside the FeaturePreprocessor class

    def _engineer_features(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Create features on a price series, add Return_5d and y_5d only for main (no suffix)."""
        price_col = self._pick_price_col(df)
        px = df[price_col].astype(float)

        # Moving averages
        df[f"SMA_5{suffix}"] = px.rolling(5).mean()
        df[f"EMA_12{suffix}"] = px.ewm(span=12, adjust=False).mean()
        df[f"EMA_26{suffix}"] = px.ewm(span=26, adjust=False).mean()

        # RSI(14)
        df[f"RSI_14{suffix}"] = self._rsi(px, period=14)

        # MACD (line only): EMA12 - EMA26
        ema12 = df[f"EMA_12{suffix}"]
        ema26 = df[f"EMA_26{suffix}"]
        df[f"MACD{suffix}"] = ema12 - ema26

        # Bollinger Bands (20, 2σ by default) on price
        ma = px.rolling(self.bb_window).mean()
        sd = px.rolling(self.bb_window).std()
        df[f"BB_upper{suffix}"] = ma + self.bb_std * sd
        df[f"BB_lower{suffix}"] = ma - self.bb_std * sd

        # Momentum 10 (percentage change over 10 periods)
        df[f"Momentum_10{suffix}"] = px.pct_change(10)

        # Volatility 10d: rolling std of daily returns over 10 days
        daily_ret = px.pct_change()
        df[f"Volatility_10d{suffix}"] = daily_ret.rolling(10).std()

        # Only for the main (ticker) dataset
        if suffix == "":
            # Calculate the 5-day forward return
            df["Return_5d"] = px.shift(-5) / px - 1
            # Generate binary target: 1 if Return_5d > 0, else 0
            df["y_5d"] = df["Return_5d"].apply(lambda x: 1 if x > 0 else 0)

        return df

    # ---------- public API ----------

    def process_all(self) -> None:
        """Process every CSV in the tickers_folder and save a cleaned CSV per ticker."""
        for fname in os.listdir(self.tickers_folder):
            if not fname.lower().endswith(".csv"):
                continue
            in_path = self.tickers_folder / fname
            try:
                out_df, out_path = self._process_one(in_path)
                print(f"✅ Saved: {out_path} ({len(out_df)} rows)")
            except Exception as e:
                print(f"❌ Failed on {fname}: {e}")

    def _process_one(self, in_path: Path):
        df = self._load_csv(in_path)

        # Basic sanity: keep only needed raw cols if present
        # (We will always keep Date + Volume in output)
        if "Volume" not in df.columns:
            df["Volume"] = np.nan  # ensure column exists

        # Build ticker features
        df = self._engineer_features(df, suffix="")  # adds Return_5d too

        # Merge XLK features on Date
        merged = pd.merge(df, self.xlk_df, on="Date", how="left")

        # Select + order exact output columns
        cols = [
            "Date", "Volume",
            "SMA_5", "EMA_12", "EMA_26", "RSI_14", "MACD",
            "BB_upper", "BB_lower", "Momentum_10", "Volatility_10d",
            "SMA_5_XLK", "EMA_12_XLK", "EMA_26_XLK", "RSI_14_XLK", "MACD_XLK",
            "BB_upper_XLK", "BB_lower_XLK", "Momentum_10_XLK", "Volatility_10d_XLK",
            "Return_5d", "y_5d"
        ]
        # Make sure all exist
        for c in cols:
            if c not in merged.columns:
                merged[c] = np.nan
        out = merged[cols].copy()

        # Drop any NaNs (including lookback-induced NaNs + initial 2-row drop leftovers)
        out = out.dropna().reset_index(drop=True)

        # Save
        out_path = self.out_dir / in_path.name  # keep ticker name
        out.to_csv(out_path, index=False)
        return out, out_path


# ---------------- example usage ----------------
# if __name__ == "__main__":
#     pre = FeaturePreprocessor(
#         tickers_folder="/Users/nikita/Documents/data",
#         xlk_csv_path="/Users/nikita/Documents/data/XLK.csv",
#         out_dir="/Users/nikita/Documents/data/proccesed"
#     )
#     pre.process_all()
