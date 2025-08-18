from __future__ import annotations
import argparse, lightgbm as lgb
from data_utils import load_all_csvs, split_by_tickers, select_numeric_features, print_metrics

def main(data_dir: str, test_tickers=None, target_col: str="y_5d"):
    full = load_all_csvs(data_dir, target_col=target_col)
    tr, te = split_by_tickers(full, test_tickers=test_tickers)
    Xtr, ytr = select_numeric_features(tr, target_col=target_col, extra_drop=["Ticker"])
    Xte, yte = select_numeric_features(te, target_col=target_col, extra_drop=["Ticker"])

    model = lgb.LGBMClassifier(
        max_depth=7, num_leaves=96, learning_rate=0.02,
        n_estimators=1500, subsample=0.8, colsample_bytree=0.8,
        objective="binary", random_state=42
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    y_prob = model.predict_proba(Xte)[:,1]
    print("== LightGBM ==")
    print_metrics(yte, y_prob)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    args = ap.parse_args()
    main(args.data_dir, test_tickers=args.test_tickers, target_col=args.target_col)