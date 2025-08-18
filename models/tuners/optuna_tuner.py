from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, CatBoostError
import optuna
from optuna.samplers import TPESampler

from models.data_utils import load_all_csvs, split_by_tickers, print_metrics


def time_val_split_per_ticker(df: pd.DataFrame, val_frac: float = 0.2):
    tr_parts, va_parts = [], []
    for _, g in df.groupby("Ticker"):
        g = g.sort_values("Date")
        n = len(g)
        cut = max(1, int(math.floor(n * (1 - val_frac))))
        tr_parts.append(g.iloc[:cut]); va_parts.append(g.iloc[cut:])
    return pd.concat(tr_parts, ignore_index=True), pd.concat(va_parts, ignore_index=True)


def build_feature_view(df: pd.DataFrame, target_col: str):
    drop_cols = {target_col, "Date", "Return_5d"}  # (no fwd_ret_5d)
    feat_cols = [c for c in df.columns if c not in drop_cols]
    feat_cols = ["Ticker"] + [c for c in feat_cols if c != "Ticker"] if "Ticker" in feat_cols else ["Ticker"] + feat_cols
    X = df[feat_cols].copy()
    y = df[target_col].astype(int).values
    cat_idx = [0]
    return X, y, cat_idx, feat_cols


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden") -> float:
    from sklearn.metrics import roc_curve, precision_recall_curve
    y_true = y_true.astype(int); y_prob = y_prob.astype(float)
    if method == "f1":
        p, r, thr = precision_recall_curve(y_true, y_prob)
        f1 = 2*p*r/(p+r+1e-9); return float(thr[int(np.nanargmax(f1[:-1]))])
    fpr, tpr, thr = roc_curve(y_true, y_prob); mask = np.isfinite(thr)
    j = tpr[mask] - fpr[mask]; return float(thr[mask][int(np.argmax(j))])


def save_artifacts(model: CatBoostClassifier, out_dir: Optional[str], best_params: Dict, meta: Dict) -> Optional[Path]:
    if not out_dir: return None
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p_model = out / "catboost_optuna_best.cbm"
    model.save_model(str(p_model))
    (out / "catboost_optuna_params.json").write_text(json.dumps(best_params, indent=2))
    (out / "catboost_optuna_meta.json").write_text(json.dumps(meta, indent=2))
    return p_model


def make_objective(Xtr, ytr, cat_idx, Xva, yva, seed: int):
    """Return a callable Optuna objective (must RETURN this function!)."""
    pool_tr = Pool(Xtr, ytr, cat_features=cat_idx)
    pool_va = Pool(Xva, yva, cat_features=cat_idx)

    def objective(trial: optuna.Trial) -> float:
        # Sample a VALID CatBoost config
        params = {
            "depth": trial.suggest_int("depth", 4, 10, step=2),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 12),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "random_strength": trial.suggest_float("random_strength", 1.0, 5.0),
            "leaf_estimation_iterations": trial.suggest_categorical("leaf_estimation_iterations", [1, 3, 5]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        }
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        else:  # Bernoulli
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

        try:
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="AUC",
                iterations=3000, od_type="Iter", od_wait=150,
                random_seed=seed, verbose=False, **params
            )
            model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
            best_auc = model.get_best_score().get("validation", {}).get("AUC")
            if best_auc is None:
                from sklearn.metrics import roc_auc_score
                y_prob = model.predict_proba(pool_va)[:, 1]
                best_auc = roc_auc_score(yva, y_prob)
            return float(best_auc)
        except CatBoostError:
            # Invalid combo or edge case → prune the trial instead of crashing
            raise optuna.TrialPruned()

    return objective  # <<< THIS LINE IS CRITICAL


def main(
    data_dir: str,
    test_tickers=None,
    target_col: str = "y_5d",
    val_frac: float = 0.2,
    n_trials: int = 30,
    seed: int = 42,
    threshold_method: str = "youden",
    save_dir: Optional[str] = None,
):
    full = load_all_csvs(data_dir, target_col=target_col)
    train_df, test_df = split_by_tickers(full, test_tickers=test_tickers)

    tr_df, va_df = time_val_split_per_ticker(train_df, val_frac=val_frac)
    Xtr, ytr, cat_idx, feat_cols = build_feature_view(tr_df, target_col)
    Xva, yva, _, _               = build_feature_view(va_df, target_col)
    Xte, yte, _, _               = build_feature_view(test_df, target_col)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    objective = make_objective(Xtr, ytr, cat_idx, Xva, yva, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_value  = study.best_value
    print("\n🏆 Best validation AUC:", round(best_value, 6))
    print("🏁 Best params:", json.dumps(best_params, indent=2))

    pool_tr = Pool(Xtr, ytr, cat_features=cat_idx)
    pool_va = Pool(Xva, yva, cat_features=cat_idx)
    best_model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC",
        iterations=3000, od_type="Iter", od_wait=150,
        random_seed=seed, verbose=False, **best_params
    )
    best_model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

    y_prob_val = best_model.predict_proba(pool_va)[:, 1]
    thr = find_best_threshold(yva, y_prob_val, method=threshold_method)
    print(f"\n🔎 Chosen threshold ({threshold_method} on validation): {thr:.4f}")

    pool_te = Pool(Xte, yte, cat_features=cat_idx)
    y_prob_te = best_model.predict_proba(pool_te)[:, 1]
    print("\n== CatBoost (Optuna-tuned) ==")
    print_metrics(yte, y_prob_te, threshold=thr)

    meta = {
        "target_col": target_col, "test_tickers": list(test_tickers or []),
        "val_frac": val_frac, "threshold_method": threshold_method, "threshold": float(thr),
        "best_auc_val": float(best_value), "features": feat_cols, "cat_features_idx": cat_idx,
        "optuna_best_trial": dict(number=study.best_trial.number, value=study.best_value),
    }
    path = save_artifacts(best_model, save_dir, best_params, meta)
    if path:
        print(f"\n💾 Saved best model to: {path}")
        print(f"💾 Saved params/meta to: {Path(save_dir).resolve()}")


# -------- regular main (edit & run) --------
if __name__ == "__main__":
    DATA_DIR = "/Users/nikita/Documents/final_project_data/sentiment_features_cleaned"
    TEST_TICKERS = ["TSLA", "NVDA"]           # or None
    SAVE_DIR ="/Users/nikita/Documents/final_project_data"  # or None
    VAL_FRAC = 0.2
    N_TRIALS = 30
    SEED = 42
    THRESHOLD_METHOD = "youden"  # or "f1"

    main(
        data_dir=DATA_DIR,
        test_tickers=TEST_TICKERS,
        target_col="y_5d",
        val_frac=VAL_FRAC,
        n_trials=N_TRIALS,
        seed=SEED,
        threshold_method=THRESHOLD_METHOD,
        save_dir=SAVE_DIR,
    )
