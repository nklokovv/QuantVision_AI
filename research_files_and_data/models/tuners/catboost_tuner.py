# models/tuners/tune_catboost_cv_main.py
from __future__ import annotations
import json, math, itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# your utilities
from research_files_and_data.models.data_utils import load_all_csvs, split_by_tickers, print_metrics


# ---------- helpers ----------
def build_feature_view(df: pd.DataFrame, target_col: str):
    drop_cols = {target_col, "Date", "Return_5d"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    if "Ticker" not in feat_cols:
        feat_cols = ["Ticker"] + feat_cols
    else:
        feat_cols = ["Ticker"] + [c for c in feat_cols if c != "Ticker"]
    X = df[feat_cols].copy()
    y = df[target_col].astype(int).values
    cat_idx = [0]
    return X, y, cat_idx, feat_cols


def make_time_folds(train_df: pd.DataFrame, n_folds: int = 3, min_train_frac: float = 0.6) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    For each ticker, create n_folds chronological validation windows after an initial min_train_frac.
    Fold k uses: train = [start : fold_start), valid = [fold_start : fold_end)
    Then concat across tickers for each fold.
    """
    folds: List[Tuple[List[pd.DataFrame], List[pd.DataFrame]]] = [([], []) for _ in range(n_folds)]
    for _, g in train_df.groupby("Ticker"):
        g = g.sort_values("Date")
        n = len(g)
        if n < 50:
            # with short series, just use last 20% as a single fold
            cut = max(1, int(math.floor(n * 0.8)))
            folds[0][0].append(g.iloc[:cut]); folds[0][1].append(g.iloc[cut:])
            continue

        start = int(math.floor(n * min_train_frac))
        remain = n - start
        if remain < n_folds:  # not enough points → reduce folds
            local_folds = max(1, min(remain, n_folds))
        else:
            local_folds = n_folds

        # equal windows over the remainder
        edges = np.linspace(start, n, local_folds + 1, dtype=int)
        for k in range(local_folds):
            st, en = edges[k], edges[k + 1]
            if st >= en:  # skip empty
                continue
            folds[k][0].append(g.iloc[:st])  # train up to st
            folds[k][1].append(g.iloc[st:en])  # val window

    # stitch across tickers & drop empty folds
    out: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for k in range(n_folds):
        tr_parts, va_parts = folds[k]
        if not tr_parts or not va_parts:
            continue
        tr_df = pd.concat(tr_parts, ignore_index=True)
        va_df = pd.concat(va_parts, ignore_index=True)
        out.append((tr_df, va_df))
    return out


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden") -> float:
    from sklearn.metrics import roc_curve, precision_recall_curve
    y_true = y_true.astype(int); y_prob = y_prob.astype(float)
    if method == "f1":
        p, r, thr = precision_recall_curve(y_true, y_prob)
        f1 = 2 * p * r / (p + r + 1e-9)
        return float(thr[int(np.nanargmax(f1[:-1]))])
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = np.isfinite(thr)
    j = tpr[mask] - fpr[mask]
    return float(thr[mask][int(np.argmax(j))])


def class_weights_from_y(y: np.ndarray) -> List[float]:
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    if pos == 0 or neg == 0:  # degenerate
        return [1.0, 1.0]
    return [1.0, neg / pos]  # [w0, w1]


def save_artifacts(model: CatBoostClassifier, out_dir: Optional[str], best_params: Dict, meta: Dict) -> Optional[Path]:
    if not out_dir:
        return None
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model_path = out / "catboost_cv_best.cbm"
    model.save_model(str(model_path))
    (out / "catboost_cv_best_params.json").write_text(json.dumps(best_params, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return model_path


# ---------- grid (narrow, high-signal) ----------
def build_param_grid() -> List[Dict]:
    grid = {
        "depth": [6, 8],
        "learning_rate": [0.02, 0.03, 0.05],
        "l2_leaf_reg": [3, 5, 7, 9],
        "min_data_in_leaf": [32, 64, 128],
        "bootstrap_type": ["Bayesian"],          # stable for small data
        "bagging_temperature": [0.25, 0.5, 0.75],
        "border_count": [254],
        "random_strength": [1.0, 2.0],
        "leaf_estimation_iterations": [3],
    }
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, values))
        # enforce bootstrap constraint (already Bayesian only)
        d.pop("subsample", None)
        combos.append(d)
    return combos


# ---------- main ----------
def main(
    data_dir: str,
    test_tickers=None,
    target_col: str = "y_5d",
    n_folds: int = 3,
    min_train_frac: float = 0.6,
    seed: int = 42,
    threshold_method: str = "youden",
    save_dir: Optional[str] = None,
):
    # Load & split
    full = load_all_csvs(data_dir, target_col=target_col)
    train_df, test_df = split_by_tickers(full, test_tickers=test_tickers)

    # CV folds inside training tickers
    folds = make_time_folds(train_df, n_folds=n_folds, min_train_frac=min_train_frac)
    if not folds:
        raise RuntimeError("Could not build CV folds (not enough data).")

    # Feature views once per fold
    fold_views = []
    for tr_df, va_df in folds:
        Xtr, ytr, cat_idx, feat_cols = build_feature_view(tr_df, target_col)
        Xva, yva, _, _               = build_feature_view(va_df, target_col)
        cw = class_weights_from_y(ytr)
        fold_views.append((Xtr, ytr, Xva, yva, cat_idx, cw))
    # Test view (held-out tickers)
    Xte, yte, _, _ = build_feature_view(test_df, target_col)

    # Parameter grid
    param_grid = build_param_grid()

    best_params = None
    best_auc = -1.0
    log_rows = []

    for i, params in enumerate(param_grid, 1):
        aucs = []
        for (Xtr, ytr, Xva, yva, cat_idx, class_weights) in fold_views:
            pool_tr = Pool(Xtr, ytr, cat_features=cat_idx)
            pool_va = Pool(Xva, yva, cat_features=cat_idx)

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=4000,
                od_type="Iter",
                od_wait=300,
                random_seed=seed,
                verbose=False,
                class_weights=class_weights,
                **params,
            )
            model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
            # use model's best AUC if available, otherwise compute
            cur_auc = model.get_best_score().get("validation", {}).get("AUC")
            if cur_auc is None:
                y_prob = model.predict_proba(pool_va)[:, 1]
                cur_auc = roc_auc_score(yva, y_prob)
            aucs.append(float(cur_auc))

        mean_auc = float(np.mean(aucs))
        log_rows.append({"iter": i, "mean_val_auc": mean_auc, **params})
        print(f"[{i:03d}/{len(param_grid)}] mean AUC={mean_auc:.5f} "
              f"depth={params['depth']} lr={params['learning_rate']:.3f} "
              f"l2={params['l2_leaf_reg']} min_leaf={params['min_data_in_leaf']} "
              f"bag_temp={params['bagging_temperature']} rand={params['random_strength']}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    assert best_params is not None
    print("\n🏆 CV-best mean AUC:", round(best_auc, 6))
    print("🏁 Best params:", json.dumps(best_params, indent=2))

    # Refit on all training data with a final validation window (last fold) for early stopping
    Xtr_all, ytr_all, cat_idx, feat_cols = build_feature_view(train_df, target_col)
    # reuse class weights from full train
    class_weights = class_weights_from_y(ytr_all)
    # Use last fold's validation as eval_set to control early stopping
    _, va_last = folds[-1]
    Xva_last, yva_last, _, _ = build_feature_view(va_last, target_col)

    pool_tr_all = Pool(Xtr_all, ytr_all, cat_features=cat_idx)
    pool_va_last = Pool(Xva_last, yva_last, cat_features=cat_idx)

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=4000,
        od_type="Iter",
        od_wait=300,
        random_seed=seed,
        verbose=False,
        class_weights=class_weights,
        **best_params,
    )
    final_model.fit(pool_tr_all, eval_set=pool_va_last, use_best_model=True)

    # Choose threshold on aggregated validation predictions (use all folds)
    val_probs_all, val_true_all = [], []
    for (Xtr, ytr, Xva, yva, cat_idx, class_weights) in fold_views:
        m = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=final_model.tree_count_,  # trained length
            random_seed=seed,
            verbose=False,
            class_weights=class_weights,
            **best_params,
        )
        m.fit(Pool(Xtr, ytr, cat_features=cat_idx), eval_set=Pool(Xva, yva, cat_features=cat_idx), use_best_model=False)
        val_probs_all.append(m.predict_proba(Pool(Xva, yva, cat_features=cat_idx))[:, 1])
        val_true_all.append(yva)
    val_probs_all = np.concatenate(val_probs_all)
    val_true_all = np.concatenate(val_true_all)
    thr = find_best_threshold(val_true_all, val_probs_all, method=threshold_method)
    print(f"\n🔎 Chosen threshold ({threshold_method} on CV validation): {thr:.4f}")

    # Test evaluation on held-out tickers
    y_prob_te = final_model.predict_proba(Pool(Xte, yte, cat_features=cat_idx))[:, 1]
    print("\n== CatBoost (CV-tuned) ==")
    print_metrics(yte, y_prob_te, threshold=thr)

    # Save artifacts (optional)
    path = save_artifacts(final_model, save_dir, best_params, {
        "target_col": target_col,
        "test_tickers": list(test_tickers or []),
        "n_folds": n_folds,
        "min_train_frac": min_train_frac,
        "cv_best_mean_auc": best_auc,
        "threshold_method": threshold_method,
        "threshold": float(thr),
        "features": feat_cols,
        "cat_features_idx": [0],
        "class_weights": class_weights,
    })
    if path:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(Path(save_dir) / "catboost_cv_search_log.csv", index=False)
        print(f"\n💾 Saved model → {path}")
        print(f"💾 Saved search log → {Path(save_dir) / 'catboost_cv_search_log.csv'}")


# ----- EDIT & RUN (regular main) -----
if __name__ == "__main__":
    DATA_DIR = "/Users/nikita/Documents/final_project_data/sentiment_features_cleaned"
    TEST_TICKERS = ["TSLA", "NVDA"]           # or None
    SAVE_DIR = "/Users/nikita/Documents/final_project_data/models_cv"  # or None
    N_FOLDS = 3
    MIN_TRAIN_FRAC = 0.6
    SEED = 42
    THRESHOLD_METHOD = "youden"  # or "f1"

    main(
        data_dir=DATA_DIR,
        test_tickers=TEST_TICKERS,
        target_col="y_5d",
        n_folds=N_FOLDS,
        min_train_frac=MIN_TRAIN_FRAC,
        seed=SEED,
        threshold_method=THRESHOLD_METHOD,
        save_dir=SAVE_DIR,
    )
