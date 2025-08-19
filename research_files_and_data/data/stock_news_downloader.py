import time
import math
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from email.utils import parsedate_to_datetime


class StockNewsFetcher:
    BASE_URL = "https://stocknewsapi.com/api/v1"

    def __init__(
        self,
        api_token: str,
        out_dir: Path,
        *,
        items_per_page: int = 100,     # API max = 100
        max_pages: int = 100,          # API hard page cap
        rate_sleep: float = 0.35,      # ~3 req/sec to be polite
        timeout: int = 20,             # seconds
        session: Optional[requests.Session] = None
    ):
        self.token = api_token
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.items = min(items_per_page, 100)
        self.max_pages = max_pages
        self.sleep = rate_sleep
        self.timeout = timeout
        self.sess = session or requests.Session()

    # ---------- public API ----------
    def fetch_all(
        self,
        tickers: List[str],
        start_date: str,  # "YYYY-MM-DD"
        end_date: str     # "YYYY-MM-DD"
    ) -> Dict[str, Path]:
        """Fetch & save one CSV per ticker. Returns {ticker: path}."""
        saved = {}
        for t in tickers:
            print(f"▶ {t}: fetching…")
            path = self.fetch_ticker(t, start_date, end_date)
            print(f"   saved → {path.name}")
            saved[t] = path
        return saved

    def fetch_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Path:
        """Fetch full history for one ticker and save CSV."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

        rows = self._fetch_recursive(ticker, start_dt, end_dt)
        df = self._normalize(rows, ticker)
        fname = f"{ticker}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        out_path = self.out_dir / fname
        df.to_csv(out_path, index=False)
        return out_path

    # ---------- internals ----------
    def _fetch_recursive(self, ticker: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Fetch pages; if we hit page cap, split range and recurse."""
        rows, hit_cap = self._fetch_pages(ticker, start, end)
        if not hit_cap:
            return rows
        # split the range roughly in half and recurse
        mid = start + (end - start) / 2
        left  = self._fetch_recursive(ticker, start, mid)
        right = self._fetch_recursive(ticker, mid + timedelta(days=1), end)
        return left + right

    def _fetch_pages(self, ticker: str, start: datetime, end: datetime) -> Tuple[List[Dict[str, Any]], bool]:
        rows, page, hit_cap = [], 1, False
        while page <= self.max_pages:
            params = {
                "tickers": ticker,
                "items":   self.items,
                "date":    f"{start.strftime('%m%d%Y')}-{end.strftime('%m%d%Y')}",
                "page":    page,
                "token":   self.token
            }
            r = self._request_with_retry(params)
            data = r.json().get("data", [])
            if not data:
                break
            rows.extend(data)
            page += 1
            time.sleep(self.sleep)
        if page > self.max_pages and len(rows) > 0:
            hit_cap = True
        return rows, hit_cap

    def _request_with_retry(self, params: Dict[str, Any], retries: int = 3) -> requests.Response:
        """GET with simple exponential backoff on 429/5xx."""
        backoff = 1.0
        for attempt in range(retries + 1):
            resp = self.sess.get(self.BASE_URL, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp
            # Handle common throttling / transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            # For anything else (403, 4xx…), raise immediately for visibility
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text
            raise requests.HTTPError(f"HTTP {resp.status_code}: {msg}")
        # Shouldn’t reach here
        raise RuntimeError("Unreachable: retry loop exhausted without return")

    @staticmethod
    def _parse_rfc822(dt_str: str) -> Tuple[str, str]:
        """
        'Wed, 29 May 2019 15:10:00 -0400' -> ('5/29/2019', '15:10:00')
        """
        dt = parsedate_to_datetime(dt_str)  # robust RFC-822 parser
        return f"{dt.month}/{dt.day}/{dt.year}", dt.strftime("%H:%M:%S")

    def _normalize(self, rows: List[Dict[str, Any]], ticker: str) -> pd.DataFrame:
        """Deduplicate & split date/time; keep the useful columns."""
        if not rows:
            return pd.DataFrame(columns=["ticker","news_url","source_name","title","text","image_url","Date","Time"])

        df = pd.DataFrame(rows)
        # Drop duplicates by URL (common across overlapping ranges)
        if "news_url" in df.columns:
            df = df.drop_duplicates(subset=["news_url"], keep="first").copy()

        # Split RFC-822 'date' field into Date / Time if present
        if "date" in df.columns:
            parsed = df["date"].apply(lambda s: self._parse_rfc822(s) if isinstance(s, str) else (None, None))
            df["Date"] = parsed.apply(lambda x: x[0])
            df["Time"] = parsed.apply(lambda x: x[1])
        else:
            df["Date"] = None
            df["Time"] = None

        df["ticker"] = ticker

        # Keep a tidy set of columns if available
        keep_order = [c for c in [
            "ticker", "news_url", "source_name", "title", "text", "image_url", "Date", "Time"
        ] if c in df.columns]
        return df[keep_order] if keep_order else df


# ================= USAGE EXAMPLE =================
# # 1) Configure
# API_TOKEN = "rkaf940tbhfpb2l9covo7ldphg2xgjdkd7vuohy8"  # <-- put your StockNewsAPI token
# OUT_DIR   = Path("/Users/nikita/Documents/data/news")     # change to your path
# # tickers   = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","AMD","CRM","INTC"]
# tickers   = ["AAPL","MSFT"]
# # 2) Run
# fetcher = StockNewsFetcher(API_TOKEN, OUT_DIR)
# files = fetcher.fetch_all(tickers, start_date="2019-03-01", end_date="2020-03-01")
#
# # 3) Check result
# for t, p in files.items():
#     print(t, "→", p)
