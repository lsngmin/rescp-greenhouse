"""
LightGBM forecaster.
- input: window (6, 9) flattened to 54 features (no standardisation; tree model)
- one model per target (T_in_max1h, RH_in_max1h)
- tuning: Optuna TPE, 50 trials, objective = validation RMSE_T + RMSE_RH; number of trees by early stopping (50) on validation
- final: selected parameters, 66 training days + early stopping on validation, 5 seeds -> calibration/test predictions

Usage: python run_lgbm.py --data data --out results --name kangwon [--trials 50]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
import optuna
from common import Samples, SEEDS, rmse, save_pred

MODEL = "lgbm"
EARLY_STOP = 50
MAX_TREES = 3000


def flat(X):
    return X.reshape(len(X), -1)


def fit_one(params, Xtr, ytr, Xva, yva, seed):
    p = dict(params, objective="regression", metric="rmse", verbosity=-1, seed=seed,
             bagging_seed=seed, feature_fraction_seed=seed, n_jobs=8)
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=MAX_TREES, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def fit_two(params, S, Xtr, Xva, seed):
    """Fit one model per target. Returns (models, val_rmse_list, best_iters)."""
    models, vr, its = [], [], []
    for j in range(2):
        m = fit_one(params, Xtr, S.y_train[:, j], Xva, S.y_val[:, j], seed)
        models.append(m)
        vr.append(rmse(S.y_val[:, j], m.predict(Xva, num_iteration=m.best_iteration)))
        its.append(int(m.best_iteration))
    return models, vr, its


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", required=True)
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    S = Samples(a.data, a.name)
    print(json.dumps(S.summary(), ensure_ascii=False))
    Xtr, Xva, Xca, Xte = flat(S.X_train), flat(S.X_val), flat(S.X_calib), flat(S.X_test)
    out = Path(a.out).expanduser() / a.name / MODEL
    out.mkdir(parents=True, exist_ok=True)

    # ---- tuning (seed 0) ----
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        }
        _, vr, its = fit_two(params, S, Xtr, Xva, seed=0)
        trial.set_user_attr("val_rmse", vr)
        trial.set_user_attr("best_iters", its)
        return sum(vr)

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params, bagging_freq=1)
    tune_info = {
        "n_trials": a.trials, "best_value": best.value, "best_params": best_params,
        "best_val_rmse": best.user_attrs["val_rmse"], "best_iters": best.user_attrs["best_iters"],
        "tune_sec": round(time.time() - t0, 1),
        "trials": [{"n": t.number, "value": t.value, "params": t.params} for t in study.trials],
    }
    (out / "tuning.json").write_text(json.dumps(tune_info, indent=2))
    print(f"[tune] best val RMSE T={best.user_attrs['val_rmse'][0]:.3f} RH={best.user_attrs['val_rmse'][1]:.3f} "
          f"iters={best.user_attrs['best_iters']} ({tune_info['tune_sec']}s)")

    # ---- final models: 5 seeds ----
    for seed in SEEDS:
        models, vr, its = fit_two(best_params, S, Xtr, Xva, seed)
        p_ca = np.column_stack([m.predict(Xca, num_iteration=m.best_iteration) for m in models])
        p_te = np.column_stack([m.predict(Xte, num_iteration=m.best_iteration) for m in models])
        met = save_pred(out / f"seed{seed}", S, p_ca, p_te,
                        extra={"val_rmse_T": vr[0], "val_rmse_RH": vr[1], "n_trees": its,
                               "n_params_leaves_total": int(sum(m.num_trees() for m in models))})
        for j, m in enumerate(models):
            m.save_model(str(out / f"seed{seed}" / f"model_{j}.txt"))
        print(f"[seed {seed}] val T={vr[0]:.3f} RH={vr[1]:.3f} | calib T={met['rmse_calib_T']:.3f} RH={met['rmse_calib_RH']:.3f} "
              f"| test T={met['rmse_test_T']:.3f} RH={met['rmse_test_RH']:.3f} | trees={its}")


if __name__ == "__main__":
    main()
