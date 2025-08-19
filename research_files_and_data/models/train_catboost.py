from __future__ import annotations
import argparse
from catboost import CatBoostClassifier, Pool
from data_utils import load_all_csvs, split_by_tickers, print_metrics

def main(data_dir: str, test_tickers=None, target_col: str="y_5d"):
    full = load_all_csvs(data_dir, target_col=target_col)
    tr, te = split_by_tickers(full, test_tickers=test_tickers)

    # keep Ticker as categorical first column, other numeric features follow
    drop_cols = {target_col, "Date", "Return_5d"}
    feat_cols = [c for c in tr.columns if c not in drop_cols]
    if "Ticker" not in feat_cols: feat_cols = ["Ticker"] + feat_cols
    else: feat_cols = ["Ticker"] + [c for c in feat_cols if c != "Ticker"]

    def Xy(df):
        X = df[feat_cols].copy()
        y = df[target_col].astype(int).values
        return X, y

    Xtr, ytr = Xy(tr)
    Xte, yte = Xy(te)
    cat_idx = [0]
    pool_tr, pool_te = Pool(Xtr, ytr, cat_features=cat_idx), Pool(Xte, yte, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.1, l2_leaf_reg=9,
        loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False
    )
    model.fit(pool_tr, eval_set=pool_te, use_best_model=True)
    y_prob = model.predict_proba(pool_te)[:,1]
    print("== CatBoost ==")
    print_metrics(yte, y_prob)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    args = ap.parse_args()
    main(args.data_dir, test_tickers=args.test_tickers, target_col=args.target_col)