# data/pipe/news_sent_minimal.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json
import pandas as pd

# Your implementations
from data.stock_news_downloader import StockNewsFetcher
from data.finbert_processor import FinbertSentimentProcessor


# ---------- filesystem ----------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- column normalization ----------
def _to_date_col(df: pd.DataFrame) -> pd.Series:
    """
    Return a normalized YYYY-MM-DD date series from common candidates:
    'Date', 'publishedAt', 'date', etc.
    """
    for c in ["Date", "publishedAt", "date", "published_at"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            return s.dt.strftime("%Y-%m-%d")
    # nothing found → empty string
    return pd.Series([""] * len(df))


def _to_source_col(df: pd.DataFrame) -> pd.Series:
    """
    Return a normalized source-name series from candidates:
    'source_name', 'Source', 'source', 'news_site', 'provider'.
    If 'source' looks like a JSON/dict, try to parse `.name`.
    """
    candidates = ["source_name", "Source", "source", "news_site", "provider", "site"]
    for c in candidates:
        if c in df.columns:
            s = df[c]
            # try to parse dict/JSON with "name"
            def _norm_one(v):
                if isinstance(v, str):
                    v_str = v.strip()
                    # simple JSON/dict heuristics
                    if (v_str.startswith("{") and v_str.endswith("}")) or (v_str.startswith("[") and v_str.endswith("]")):
                        try:
                            obj = json.loads(v_str)
                            if isinstance(obj, dict) and "name" in obj:
                                return str(obj.get("name", "")).strip()
                        except Exception:
                            pass
                    return v_str
                if isinstance(v, dict):
                    return str(v.get("name", "")).strip()
                return "" if pd.isna(v) else str(v)
            return s.map(_norm_one)
    # nothing found
    return pd.Series([""] * len(df))


def _to_sent_score_col(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric sentiment score.
    Prefers existing Sent_Score/sent_score; else compute pos - neg when available;
    else map label {'positive': +1, 'negative': -1, 'neutral': 0}.
    """
    for c in ["Sent_Score", "sent_score", "sentiment_score", "score"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    pos_names = [c for c in df.columns if c.lower() in {"sent_pos", "pos", "positive"}]
    neg_names = [c for c in df.columns if c.lower() in {"sent_neg", "neg", "negative"}]
    if pos_names and neg_names:
        pos = pd.to_numeric(df[pos_names[0]], errors="coerce")
        neg = pd.to_numeric(df[neg_names[0]], errors="coerce")
        return pos.sub(neg)

    # last resort: label mapping
    for c in ["label", "sentiment", "pred_label"]:
        if c in df.columns:
            m = df[c].astype(str).str.lower().map(
                {"positive": 1.0, "pos": 1.0, "negative": -1.0, "neg": -1.0, "neutral": 0.0, "neu": 0.0}
            )
            return pd.to_numeric(m, errors="coerce")

    # give up → NaNs
    return pd.Series([float("nan")] * len(df))


def _extract_minimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Date": _to_date_col(df),
        "Source": _to_source_col(df),
        "Sent_Score": _to_sent_score_col(df),
    })
    # drop rows with no date and no score
    mask = out["Date"].ne("") & out["Sent_Score"].notna()
    return out.loc[mask].reset_index(drop=True)


# ---------- merge helper (raw + scored) ----------
def _safe_merge_news_and_scores(raw_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame that contains BOTH original news columns and FinBERT scores.
    If scored_df already has raw columns, return scored_df directly.
    Else try merging on keys; fallback to positional concat.
    """
    raw_cols = set(raw_df.columns)
    scored_cols = set(scored_df.columns)
    if raw_cols.issubset(scored_cols):
        return scored_df

    sentiment_cols = [c for c in scored_df.columns if c.lower() in
                      {"sent_pos", "sent_neut", "sent_neg", "sent_score", "sentiment", "label", "sent_sco", "sentiment_score"}]
    if not sentiment_cols:
        return raw_df

    key_candidates = [
        ["uuid"],
        ["url"],
        ["publishedAt", "title"],
        ["Date", "title"],
        ["title"],
    ]
    for keys in key_candidates:
        if all(k in raw_df.columns for k in keys) and all(k in scored_df.columns for k in keys):
            try:
                return raw_df.merge(scored_df[keys + sentiment_cols], on=keys, how="left")
            except Exception:
                pass

    if len(raw_df) == len(scored_df):
        scored_only = scored_df[[c for c in scored_df.columns if c not in raw_df.columns]].reset_index(drop=True)
        return pd.concat([raw_df.reset_index(drop=True), scored_only], axis=1)

    # best-effort partial alignment
    scored_df = scored_df.reset_index(drop=True)
    raw_df = raw_df.reset_index(drop=True)
    common = min(len(raw_df), len(scored_df))
    return pd.concat(
        [raw_df.iloc[:common].copy(),
         scored_df[[c for c in scored_df.columns if c not in raw_df.columns]].iloc[:common].copy()],
        axis=1
    )


# ---------- main pipeline ----------
def run_news_sent_minimal(
    tickers: List[str],
    start_date: str,
    end_date: str,
    base_dir: str,
    api_token: str,
    raw_dir: str = "news_raw",
    scored_dir: str = "news_scored",
    out_dir_name: str = "news_minimal",
) -> None:
    """
    For each ticker:
      1) Download news → {base}/{raw_dir}
      2) FinBERT score → {base}/{scored_dir}
      3) Merge raw + scores
      4) Extract {Date, Source, Sent_Score}
      5) Save one CSV per ticker to {base}/{out_dir_name}/{TICKER}.csv
    """
    base = Path(base_dir).expanduser().resolve()
    d_raw   = (base / raw_dir).resolve()
    d_score = (base / scored_dir).resolve()
    d_out   = (base / out_dir_name).resolve()
    _ensure_dir(d_raw); _ensure_dir(d_score); _ensure_dir(d_out)

    # 1) Download
    fetcher = StockNewsFetcher(api_token, d_raw)
    fetcher.fetch_all(tickers, start_date=start_date, end_date=end_date)

    # 2) Score
    finbert = FinbertSentimentProcessor()
    finbert.process_dir(d_raw, out_dir=d_score)

    # 3) For each ticker, find latest raw + scored file and build minimal CSV
    def _latest_for(prefix_dir: Path, ticker: str) -> Optional[Path]:
        import glob
        pats = [str(prefix_dir / f"{ticker.upper()}_*.csv"), str(prefix_dir / f"*{ticker.upper()}*.csv")]
        matches: List[str] = []
        for p in pats:
            matches.extend(glob.glob(p))
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(matches[0])

    for t in sorted({tt.upper() for tt in tickers}):
        raw_path   = _latest_for(d_raw, t)
        scored_path = _latest_for(d_score, t)

        if raw_path is None and scored_path is None:
            print(f"⚠️  No news files found for {t}")
            continue

        if raw_path is None:
            # scored-only fallback
            scored_df = pd.read_csv(scored_path)
            merged = scored_df
        elif scored_path is None:
            # raw-only (unlikely if scoring succeeded)
            raw_df = pd.read_csv(raw_path)
            merged = raw_df
        else:
            raw_df = pd.read_csv(raw_path)
            scored_df = pd.read_csv(scored_path)
            merged = _safe_merge_news_and_scores(raw_df, scored_df)

        minimal = _extract_minimal_columns(merged)
        # Add sanity: sort by Date
        if "Date" in minimal.columns:
            minimal = minimal.sort_values("Date").reset_index(drop=True)

        out_path = d_out / f"{t}.csv"
        minimal.to_csv(out_path, index=False)
        print(f"✅ {t}: wrote {len(minimal)} rows → {out_path}")


# -------------- Regular main (edit & run) --------------
if __name__ == "__main__":
    # ✏️ Edit these and run the file
    TICKERS   = ["AAPL"]
    START     = "2024-01-01"
    END       = "2024-02-25"
    BASE_DIR  = "/Users/nikita/Documents/final_project_data_news"  # Mac-friendly
    API_TOKEN = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"

    run_news_sent_minimal(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        base_dir=BASE_DIR,
        api_token=API_TOKEN,
        raw_dir="news_raw",
        scored_dir="news_scored",
        out_dir_name="news_minimal",
    )
