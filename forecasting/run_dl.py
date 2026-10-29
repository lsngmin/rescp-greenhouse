"""
Common training script for the six deep-learning forecasters: gru / xlstm / moderntcn / timemixer / patchtst / itransformer
- input (B, 6, 9), standardised with 66-day training statistics -> model -> (B, 2) standardised targets
- MSE loss, AdamW, early stopping with patience 10, at most 100 epochs
- tuning: Optuna TPE, 50 trials (seed 0), objective = validation RMSE_T + RMSE_RH in original units
- final: selected parameters trained with 5 seeds (no retraining on train+val) -> calibration/test predictions

Usage: python run_dl.py --model patchtst --name kangwon [--trials 50] [--gpu 0]
"""
import argparse, json, sys, time, copy, importlib, types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import optuna
from common import Samples, SEEDS, rmse, save_pred

CODE = Path(__file__).resolve().parent
THIRD = CODE.parent / "third_party"
MAX_EPOCH, PATIENCE = 100, 10
SEQ_LEN, N_VAR, N_OUT = 6, 9, 2


# ----------------------------------------------------------------------------- third-party loaders
def _purge(names):
    for k in list(sys.modules):
        if k.split(".")[0] in names:
            del sys.modules[k]


def load_tslib(model_file):
    """Model class from Time-Series-Library models/<model_file>.py."""
    _purge({"models", "layers", "utils"})
    sys.path.insert(0, str(THIRD / "Time-Series-Library"))
    try:
        mod = importlib.import_module(f"models.{model_file}")
        return mod.Model
    finally:
        sys.path.pop(0)
        _purge({"models", "layers", "utils"})


def load_moderntcn():
    """Inner ModernTCN class from the official repository (classification variant).
    The Model wrapper expects a full argparse namespace, so the inner class is used directly."""
    _purge({"models", "layers", "utils"})
    sys.path.insert(0, str(THIRD / "ModernTCN" / "ModernTCN-classification"))
    try:
        mod = importlib.import_module("models.ModernTCN")
        return mod.ModernTCN
    finally:
        sys.path.pop(0)
        _purge({"models", "layers", "utils"})


# ----------------------------------------------------------------------------- model definitions
class PatchTSTNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        Model = load_tslib("PatchTST")
        cfg = types.SimpleNamespace(task_name="classification", seq_len=SEQ_LEN, pred_len=1, enc_in=N_VAR,
                                    num_class=N_OUT, d_model=hp["d_model"], n_heads=hp["n_heads"],
                                    e_layers=hp["e_layers"], d_ff=hp["d_model"] * hp["ff_mult"],
                                    dropout=hp["dropout"], factor=1, activation="gelu", embed="timeF", freq="h")
        self.m = Model(cfg, patch_len=hp["patch_len"], stride=hp["stride"])

    def forward(self, x):
        # same as TSLib classification() but without per-sample instance normalisation.
        # instance normalisation removes the window mean (temperature level), which is needed for absolute-value targets; inputs are already globally standardised.
        m = self.m
        enc_out, n_vars = m.patch_embedding(x.permute(0, 2, 1))
        enc_out, _ = m.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])).permute(0, 1, 3, 2)
        out = m.dropout(m.flatten(enc_out))
        return m.projection(out.reshape(out.shape[0], -1))


class ITransformerNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        Model = load_tslib("iTransformer")
        cfg = types.SimpleNamespace(task_name="classification", seq_len=SEQ_LEN, pred_len=1, enc_in=N_VAR,
                                    num_class=N_OUT, d_model=hp["d_model"], n_heads=hp["n_heads"],
                                    e_layers=hp["e_layers"], d_ff=hp["d_model"] * hp["ff_mult"],
                                    dropout=hp["dropout"], factor=1, activation="gelu", embed="timeF", freq="h",
                                    use_norm=True)
        self.m = Model(cfg)

    def forward(self, x):
        return self.m(x, None, None, None)


class ModernTCNNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        ModernTCN = load_moderntcn()
        n_stage = hp["n_stage"]
        self.m = ModernTCN(task_name="classification", patch_size=hp["patch_size"], patch_stride=hp["patch_stride"],
                           stem_ratio=6, downsample_ratio=2, ffn_ratio=hp["ffn_ratio"],
                           num_blocks=[hp["num_blocks"]] * n_stage,
                           large_size=[hp["large_size"]] * n_stage, small_size=[3] * n_stage,
                           dims=[hp["dims"]] * n_stage, dw_dims=[hp["dims"]] * n_stage,
                           nvars=N_VAR, small_kernel_merged=False, backbone_dropout=hp["dropout"],
                           head_dropout=hp["dropout"], use_multi_scale=False, revin=False, affine=False,
                           subtract_last=False, freq="h", seq_len=SEQ_LEN, c_in=(N_VAR,), individual=False,
                           target_window=1, class_drop=hp["dropout"], class_num=N_OUT)
        # the official code expects c_in as a tuple (self.n_vars[0]); the classification forward does not use RevIN, hence revin=False

    def forward(self, x):            # (B, L, M) -> ModernTCN expects (B, M, L)
        return self.m(x.permute(0, 2, 1), None)


class XLSTMNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig
        d = hp["d_model"]
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=hp["conv_kernel"], qkv_proj_blocksize=4, num_heads=hp["n_heads"])),
            context_length=SEQ_LEN, num_blocks=hp["e_layers"], embedding_dim=d, dropout=hp["dropout"], slstm_at=[])
        self.inp = nn.Linear(N_VAR, d)
        self.stack = xLSTMBlockStack(cfg)
        self.head = nn.Linear(d, N_OUT)

    def forward(self, x):
        h = self.stack(self.inp(x))     # (B, L, d)
        return self.head(h[:, -1])


class GRUNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.rnn = nn.GRU(N_VAR, hp["d_model"], num_layers=hp["e_layers"], batch_first=True,
                          dropout=hp["dropout"] if hp["e_layers"] > 1 else 0.0)
        self.drop = nn.Dropout(hp["dropout"])
        self.head = nn.Linear(hp["d_model"], N_OUT)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1]))


class TimeMixerNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        Model = load_tslib("TimeMixer")
        cfg = types.SimpleNamespace(task_name="classification", seq_len=SEQ_LEN, pred_len=1, label_len=0,
                                    enc_in=N_VAR, c_out=N_VAR, num_class=N_OUT, d_model=hp["d_model"],
                                    d_ff=hp["d_model"] * hp["ff_mult"], e_layers=hp["e_layers"], dropout=hp["dropout"],
                                    channel_independence=0, decomp_method="moving_avg", moving_avg=hp["moving_avg"],
                                    top_k=5, down_sampling_layers=hp["down_layers"], down_sampling_window=2,
                                    down_sampling_method="avg", use_norm=1, embed="timeF", freq="h")
        self.m = Model(cfg)

    def forward(self, x):
        mask = torch.ones(x.shape[0], x.shape[1], device=x.device)   # no padding
        return self.m(x, mask, None, None)


MODELS = {"patchtst": PatchTSTNet, "itransformer": ITransformerNet, "moderntcn": ModernTCNNet, "xlstm": XLSTMNet,
          "gru": GRUNet, "timemixer": TimeMixerNet}


# ----------------------------------------------------------------------------- search space
def suggest(trial, model):
    hp = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }
    if model in ("patchtst", "itransformer"):
        hp["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
        hp["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
        hp["e_layers"] = trial.suggest_int("e_layers", 1, 3)
        hp["ff_mult"] = trial.suggest_categorical("ff_mult", [1, 2, 4])
        if model == "patchtst":
            hp["patch_len"] = trial.suggest_int("patch_len", 2, 3)
            hp["stride"] = trial.suggest_int("stride", 1, 2)
    elif model == "moderntcn":
        hp["patch_size"] = trial.suggest_int("patch_size", 2, 3)
        hp["patch_stride"] = trial.suggest_int("patch_stride", 1, 2)
        hp["n_stage"] = trial.suggest_int("n_stage", 1, 2)
        hp["num_blocks"] = trial.suggest_int("num_blocks", 1, 2)
        hp["large_size"] = trial.suggest_categorical("large_size", [5, 7, 9])
        hp["ffn_ratio"] = trial.suggest_categorical("ffn_ratio", [1, 2, 4])
        hp["dims"] = trial.suggest_categorical("dims", [32, 64, 128])
    elif model == "xlstm":
        hp["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
        hp["n_heads"] = trial.suggest_categorical("n_heads", [2, 4])
        hp["e_layers"] = trial.suggest_int("e_layers", 1, 3)
        hp["conv_kernel"] = trial.suggest_categorical("conv_kernel", [2, 4])
    elif model == "gru":
        hp["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128, 256])
        hp["e_layers"] = trial.suggest_int("e_layers", 1, 3)
    elif model == "timemixer":
        hp["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
        hp["ff_mult"] = trial.suggest_categorical("ff_mult", [1, 2, 4])
        hp["e_layers"] = trial.suggest_int("e_layers", 1, 3)
        hp["moving_avg"] = trial.suggest_categorical("moving_avg", [3, 5])
        hp["down_layers"] = 1   # two scales (6 -> 3); 0 is not supported by the TimeMixer code and 2 degenerates to length 1
    return hp


# ----------------------------------------------------------------------------- training loop
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_t(a, dev):
    return torch.as_tensor(a, dtype=torch.float32, device=dev)


@torch.no_grad()
def predict(net, X, dev, bs=1024):
    net.eval()
    out = [net(to_t(X[i:i + bs], dev)).float().cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def train_one(model, hp, S, seed, dev, log=None):
    """Train on 66 days with early stopping on validation. Returns (net, best_val_rmse[T,RH], epochs_run, best_epoch, n_params)."""
    set_seed(seed)
    net = MODELS[model](hp).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    Xtr, ytr = to_t(S.norm_x(S.X_train), dev), to_t(S.norm_y(S.y_train), dev)
    Xva = S.norm_x(S.X_val)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(1, MAX_EPOCH + 1):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            loss = nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        pv = S.denorm_y(predict(net, Xva, dev))
        vr = [rmse(S.y_val[:, j], pv[:, j]) for j in range(2)]
        score = sum(vr)
        if log: log(f"  ep{ep:3d} loss={loss.item():.4f} val T={vr[0]:.3f} RH={vr[1]:.3f}")
        if score < best - 1e-6:
            best, best_vr, best_ep, bad = score, vr, ep, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best_vr, ep, best_ep, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 trial, 2 epochs, 1 seed")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    global MAX_EPOCH
    if a.smoke:
        MAX_EPOCH = 2
    S = Samples(a.data, a.name)
    print(f"[{a.model}/{a.name}] dev={dev}", json.dumps(S.summary(), ensure_ascii=False))
    out = Path(a.out).expanduser() / a.name / a.model
    out.mkdir(parents=True, exist_ok=True)

    # ---- tuning ----
    def objective(trial):
        hp = suggest(trial, a.model)
        t0 = time.time()
        try:
            _, vr, ep, best_ep, n_params = train_one(a.model, hp, S, seed=0, dev=dev)
        except (RuntimeError, ValueError, AssertionError) as e:   # incompatible combination (e.g. head count mismatch)
            print(f"  trial {trial.number} failed: {str(e)[:120]}")
            raise optuna.TrialPruned()
        trial.set_user_attr("val_rmse", vr); trial.set_user_attr("epochs", ep)
        trial.set_user_attr("best_epoch", best_ep); trial.set_user_attr("n_params", n_params)
        print(f"  trial {trial.number:2d} val T={vr[0]:.3f} RH={vr[1]:.3f} ep={best_ep}/{ep} "
              f"params={n_params} ({time.time() - t0:.0f}s)", flush=True)
        return sum(vr)

    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=1 if a.smoke else a.trials)
    best = study.best_trial
    best_hp = suggest(optuna.trial.FixedTrial(best.params), a.model)
    tune_info = {
        "n_trials": len(study.trials), "n_failed": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_value": best.value, "best_params": best.params, "best_val_rmse": best.user_attrs["val_rmse"],
        "best_epoch": best.user_attrs["best_epoch"], "n_params": best.user_attrs["n_params"],
        "tune_sec": round(time.time() - t0, 1),
        "trials": [{"n": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials],
    }
    (out / "tuning.json").write_text(json.dumps(tune_info, indent=2))
    print(f"[tune] best val T={best.user_attrs['val_rmse'][0]:.3f} RH={best.user_attrs['val_rmse'][1]:.3f} "
          f"params={best.user_attrs['n_params']} ({tune_info['tune_sec']:.0f}s) {best.params}", flush=True)

    # ---- final models: 5 seeds ----
    for seed in (SEEDS[:1] if a.smoke else SEEDS):
        net, vr, ep, best_ep, n_params = train_one(a.model, best_hp, S, seed, dev)
        p_ca = S.denorm_y(predict(net, S.norm_x(S.X_calib), dev))
        p_te = S.denorm_y(predict(net, S.norm_x(S.X_test), dev))
        met = save_pred(out / f"seed{seed}", S, p_ca, p_te,
                        extra={"val_rmse_T": vr[0], "val_rmse_RH": vr[1], "epochs_run": ep,
                               "best_epoch": best_ep, "n_params": n_params})
        torch.save(net.state_dict(), out / f"seed{seed}" / "model.pt")
        print(f"[seed {seed}] val T={vr[0]:.3f} RH={vr[1]:.3f} | calib T={met['rmse_calib_T']:.3f} RH={met['rmse_calib_RH']:.3f} "
              f"| test T={met['rmse_test_T']:.3f} RH={met['rmse_test_RH']:.3f} | ep={best_ep}/{ep}", flush=True)


if __name__ == "__main__":
    main()
