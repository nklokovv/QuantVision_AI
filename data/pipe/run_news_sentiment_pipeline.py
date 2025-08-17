
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd

# Use your package-style imports as requested
from data.stock_news_downloader import StockNewsFetcher
from data.finbert_processor import FinbertSentimentProcessor


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_concat(paths: List[Path]) -> pd.DataFrame:
    """Read multiple CSVs and concat; returns empty DF if no paths."""
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"⚠️  Failed to read {p.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out


def _all_for(prefix_dir: Path, ticker: str) -> List[Path]:
    """Return all CSVs that likely belong to ticker (sorted by mtime descending)."""
    import glob
    pats = [
        str(prefix_dir / f"{ticker.upper()}_*.csv"),
        str(prefix_dir / f"*{ticker.upper()}*.csv"),
    ]
    matches: List[str] = []
    for p in pats:
        matches.extend(glob.glob(p))
    matches = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return [Path(m) for m in matches]


def _safe_merge_news_and_scores(raw_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame that contains BOTH original news columns and FinBERT score columns.
    If scored_df already includes all raw columns → return scored_df.
    Otherwise, try key-based merges; fallback to positional concat if row counts match.
    """
    if raw_df.empty and scored_df.empty:
        return pd.DataFrame()

    if raw_df.empty:
        return scored_df
    if scored_df.empty:
        return raw_df

    raw_cols = set(raw_df.columns)
    scored_cols = set(scored_df.columns)

    # If scored already contains original columns → assume fully merged
    if raw_cols.issubset(scored_cols):
        return scored_df

    # Identify sentiment columns (common names)
    sentiment_cols = [c for c in scored_df.columns if c.lower() in
                      {"sent_pos", "sent_neut", "sent_neg", "sent_score", "sentiment", "label"}]
    if not sentiment_cols:
        # nothing to merge; return raw as-is
        return raw_df

    # Normalize potential date column for stability (optional but helps)
    if "publishedAt" in raw_df.columns:
        try:
            raw_df["publishedAt"] = pd.to_datetime(raw_df["publishedAt"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if "publishedAt" in scored_df.columns:
        try:
            scored_df["publishedAt"] = pd.to_datetime(scored_df["publishedAt"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    # Try merges on stable keys in order
    merge_key_sets = [
        ["uuid"],
        ["url"],
        ["publishedAt", "title"],
        ["Date", "title"],
        ["title"],
    ]
    for keys in merge_key_sets:
        if all(k in raw_df.columns for k in keys) and all(k in scored_df.columns for k in keys):
            try:
                merged = raw_df.merge(scored_df[keys + sentiment_cols], on=keys, how="left")
                return merged
            except Exception:
                pass

    # Fallback: if same length, concat by position
    if len(raw_df) == len(scored_df):
        scored_only = scored_df[[c for c in scored_df.columns if c not in raw_df.columns]].reset_index(drop=True)
        return pd.concat([raw_df.reset_index(drop=True), scored_only], axis=1)

    # Last resort: best-effort index-wise combine on intersection length
    scored_df = scored_df.reset_index(drop=True)
    raw_df = raw_df.reset_index(drop=True)
    common = min(len(raw_df), len(scored_df))
    combined = pd.concat([raw_df.iloc[:common].copy(),
                          scored_df[[c for c in scored_df.columns if c not in raw_df.columns]].iloc[:common].copy()],
                         axis=1)
    return combined


def run_news_sentiment_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: str,
    base_dir: str,
    api_token: str,
    raw_dir: str = "news_raw",
    scored_dir: str = "news_scored",
    per_ticker_out_dir: str = "news_scored_per_ticker",
) -> None:
    """
    For each ticker:
      1) Download raw news CSVs into {base_dir}/{raw_dir}
      2) Run FinBERT scoring into {base_dir}/{scored_dir}
      3) Merge news + scores across ALL files found per ticker and save ONE CSV per ticker
         in {base_dir}/{per_ticker_out_dir}
    """
    base = Path(base_dir)
    d_raw   = (base / raw_dir).resolve()
    d_score = (base / scored_dir).resolve()
    d_out   = (base / per_ticker_out_dir).resolve()

    _ensure_dir(d_raw)
    _ensure_dir(d_score)
    _ensure_dir(d_out)

    # 1) Download news per ticker
    fetcher = StockNewsFetcher(api_token, d_raw)
    fetcher.fetch_all(tickers, start_date=start_date, end_date=end_date)

    # 2) Score with FinBERT
    finbert = FinbertSentimentProcessor()
    finbert.process_dir(d_raw, out_dir=d_score)

    # 3) Merge per ticker (concat possible multiple raw/scored files)
    for t in tickers:
        tU = t.upper()

        raw_paths   = _all_for(d_raw, tU)
        scored_paths = _all_for(d_score, tU)

        raw_df    = _read_concat(raw_paths)
        scored_df = _read_concat(scored_paths)

        if raw_df.empty and scored_df.empty:
            print(f"⚠️  No news files found for {tU}")
            continue

        # (optional) deduplicate raw by URL or title+publishedAt if available
        if "url" in raw_df.columns:
            raw_df = raw_df.drop_duplicates(subset=["url"], keep="first")
        elif {"title", "publishedAt"}.issubset(raw_df.columns):
            raw_df = raw_df.drop_duplicates(subset=["title", "publishedAt"], keep="first")

        merged = _safe_merge_news_and_scores(raw_df, scored_df)

        # Add Ticker column if missing
        if "Ticker" not in merged.columns:
            merged.insert(0, "Ticker", tU)

        out_path = d_out / f"{tU}.csv"
        merged.to_csv(out_path, index=False)
        print(f"✅ {tU}: news + scores saved → {out_path}")


# ---------------- Example regular run (edit values below) ----------------
if __name__ == "__main__":
    TICKERS   = ["AAPL", "MSFT", "TSLA"]
    START     = "2024-01-01"
    END       = "2024-12-31"
    BASE_DIR  = "/Users/nikita/Documents/stock_project11"  # ← change to your path
    API_TOKEN = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"

    run_news_sentiment_pipeline(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        base_dir=BASE_DIR,
        api_token=API_TOKEN,
        raw_dir="news_raw",
        scored_dir="news_scored",
        per_ticker_out_dir="news_scored_per_ticker",
    )
