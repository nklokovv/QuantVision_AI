from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class FinbertSentimentProcessor:
    """
    Apply ProsusAI/finbert to StockNewsAPI CSVs (title/text),
    append per-row sentiment columns, and save results.
    """
    def __init__(self,
                 model_name: str = "ProsusAI/finbert",
                 batch_size: int = 16,
                 device: Optional[int] = None):
        if device is None:
            device = 0 if torch.backends.mps.is_available() else -1
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # return_all_scores=True gives P(pos), P(neu), P(neg) via softmax
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            truncation=True,
            max_length=512,
            function_to_apply="softmax",
            return_all_scores=True
        )

    def process_file(self,
                     csv_path: Path,
                     out_dir: Optional[Path] = None,
                     title_col: str = "title",
                     text_col: str = "text",
                     overwrite: bool = False) -> Path:
        """
        Read one raw news CSV, append sentiment columns, save scored CSV.
        """
        csv_path = Path(csv_path)
        out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / csv_path.name

        if out_path.exists() and not overwrite:
            print(f"Skipping (exists): {out_path.name}")
            return out_path

        df = pd.read_csv(csv_path)
        # Ensure columns exist
        if title_col not in df.columns: df[title_col] = ""
        if text_col  not in df.columns: df[text_col]  = ""

        texts = (df[title_col].fillna("") + ". " + df[text_col].fillna("")).tolist()

        # Batch inference
        pos, neu, neg = [], [], []
        for i in range(0, len(texts), 1000):  # chunk calls to keep memory steady
            batch_texts = texts[i:i+1000]
            outputs = self.pipe(batch_texts, batch_size=self.batch_size)
            # outputs: list of [ {label:..., score:...} * 3 ] per row
            for row in outputs:
                # map labels -> scores
                d = {dct["label"].lower(): float(dct["score"]) for dct in row}
                pos.append(d.get("positive", 0.0))
                neu.append(d.get("neutral",  0.0))
                neg.append(d.get("negative", 0.0))

        df["Sent_Pos"]   = pos
        df["Sent_Neut"]  = neu
        df["Sent_Neg"]   = neg
        df["Sent_Score"] = df["Sent_Pos"] - df["Sent_Neg"]

        df.to_csv(out_path, index=False)
        return out_path

    def process_dir(self,
                    in_dir: Path,
                    out_dir: Optional[Path] = None,
                    pattern: str = "*.csv",
                    **kwargs) -> None:
        """
        Apply FinBERT to every CSV in a folder.
        """
        in_dir = Path(in_dir)
        files = list(sorted(in_dir.glob(pattern)))
        if not files:
            print(f"No files matched {pattern} in {in_dir}")
            return
        for f in files:
            print(f"▶ Scoring {f.name} …")
            path = self.process_file(f, out_dir=out_dir, **kwargs)
            print(f"   saved → {path}")

    def aggregate_daily(self,
                        scored_csv: Path,
                        out_dir: Optional[Path] = None,
                        date_col: str = "Date",
                        score_col: str = "Sent_Score",
                        add_counts: bool = False) -> Path:
        scored_csv = Path(scored_csv)
        out_dir = Path(out_dir) if out_dir is not None else scored_csv.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(scored_csv)
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in {scored_csv.name}")

        # parse + standardize to YYYY-MM-DD
        df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
        df = df.dropna(subset=[date_col])
        iso = df[date_col].dt.strftime("%Y-%m-%d")

        grp = df.groupby(iso)
        daily = grp[score_col].mean().rename("DailySent").to_frame()
        if add_counts:
            daily["NewsCount"] = grp.size().astype(int)

        out_path = out_dir / (scored_csv.stem + "_daily.csv")
        daily.reset_index().rename(columns={"index": "Date"}).to_csv(out_path, index=False)
        return out_path

# # ======================= USAGE EXAMPLE =======================
# # 1) Set your folders
# RAW_DIR   = Path("/Users/nikita/Documents/data/news")         # raw CSVs from StockNewsAPI
# SCored_DIR = Path("/Users/nikita/Documents/data/news/after_finbert")   # where to save FinBERT-scored CSVs
#
# # 2) Run scoring on all files in RAW_DIR
# processor = FinbertSentimentProcessor(batch_size=32)  # auto-uses GPU if available
# processor.process_dir(RAW_DIR, out_dir=SCored_DIR)
#
# # 3) (Optional) Make daily-aggregated sentiment CSVs for each ticker
# for f in sorted(SCored_DIR.glob("*.csv")):
#     daily_path = processor.aggregate_daily(f)  # writes *_daily.csv next to scored file
#     print("Daily file:", daily_path.name)
