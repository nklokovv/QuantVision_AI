from __future__ import annotations
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from data_utils import load_all_csvs, split_by_tickers, select_numeric_features, print_metrics

def main(data_dir: str, test_tickers=None, target_col: str="y_5d"):
    full = load_all_csvs(data_dir, target_col=target_col)
    tr, te = split_by_tickers(full, test_tickers=test_tickers)
    Xtr, ytr = select_numeric_features(tr, target_col=target_col, extra_drop=["Ticker"])
    Xte, yte = select_numeric_features(te, target_col=target_col, extra_drop=["Ticker"])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(Xtr, ytr)
    y_prob = pipe.predict_proba(Xte)[:,1]
    print("== Logistic Regression ==")
    print_metrics(yte, y_prob)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    args = ap.parse_args()
    main(args.data_dir, test_tickers=args.test_tickers, target_col=args.target_col)