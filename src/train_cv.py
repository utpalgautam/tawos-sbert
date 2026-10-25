"""Steps 6-7: 5-fold CV on development data, then final refit + single test evaluation."""
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .model import build_model
from .evaluate import metrics, baseline_metrics


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, X, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.as_tensor(X[i:i + bs], device=device)).cpu().numpy())
    return np.concatenate(out)


def fit(Xtr, ytr, cfg, device, lr, seed, Xva=None, yva=None, fixed_epochs=None):
    """Train a fresh CNN. With validation data: early stopping on validation MAE.
    With fixed_epochs (final refit): train exactly that many epochs, no validation."""
    set_seed(seed)
    t = cfg["training"]
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(Xtr, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    best_mae, best_ep, best_state, bad = np.inf, 0, None, 0
    n_epochs = fixed_epochs or t["max_epochs"]
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(perm), t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if fixed_epochs is None:
            mae = metrics(yva, predict(model, Xva, device))["mae"]
            if mae < best_mae:
                best_mae, best_ep, bad = mae, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= t["early_stopping_patience"]:
                    break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        return model, best_ep, best_mae
    return model, fixed_epochs, None


def run_cv_and_final(X, y, manifest, cfg, project_id, out_dir, device, max_epochs=None):
    seed = cfg["experiment"]["random_seed"]
    if max_epochs:
        cfg["training"]["max_epochs"] = max_epochs
    lrs = cfg["training"].get("learning_rate_grid", [cfg["training"]["learning_rate"]])
    dev = (manifest["split"] == "development").to_numpy()
    test = ~dev
    folds = manifest["fold"].to_numpy()

    results = {}
    for lr in lrs:
        rows = []
        for f in range(1, cfg["experiment"]["validation_folds"] + 1):
            tr, va = dev & (folds != f), dev & (folds == f)
            model, best_ep, _ = fit(X[tr], y[tr], cfg, device, lr, seed, X[va], y[va])
            m = metrics(y[va], predict(model, X[va], device))
            b = baseline_metrics(y[tr], y[va])
            rows.append({"project_id": project_id, "fold": f, "mae": m["mae"], "mdae": m["mdae"],
                         "best_epoch": best_ep, "train_count": int(tr.sum()),
                         "validation_count": int(va.sum()), "seed": seed, "learning_rate": lr,
                         "baseline_mae": b["mae"], "baseline_mdae": b["mdae"]})
            print(f"  lr={lr} fold {f}: MAE={m['mae']:.3f} MdAE={m['mdae']:.3f} "
                  f"(baseline MAE={b['mae']:.3f}) best_epoch={best_ep}")
        results[lr] = pd.DataFrame(rows)

    best_lr = min(results, key=lambda k: results[k]["mae"].mean())
    cv = results[best_lr]
    cv.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summ = pd.DataFrame([{"project_id": project_id, "learning_rate": best_lr,
                          "mae_mean": cv["mae"].mean(), "mae_std": cv["mae"].std(ddof=1),
                          "mdae_mean": cv["mdae"].mean(), "mdae_std": cv["mdae"].std(ddof=1),
                          "baseline_mae_mean": cv["baseline_mae"].mean(),
                          "baseline_mdae_mean": cv["baseline_mdae"].mean()}])
    summ.to_csv(out_dir / "cv_summary.csv", index=False)
    final_epochs = int(np.median(cv["best_epoch"]))
    best_cfg = {"learning_rate": best_lr, "final_epochs": final_epochs,
                "selection_rule": "lowest mean validation MAE across folds; epochs = median best epoch",
                "mean_cv_mae": float(cv["mae"].mean())}
    with open(out_dir / "best_config.json", "w") as fh:
        json.dump(best_cfg, fh, indent=2)

    # final refit on ALL development data; evaluate the test set ONCE
    model, _, _ = fit(X[dev], y[dev], cfg, device, best_lr, seed, fixed_epochs=final_epochs)
    pred = predict(model, X[test], device)
    tm = metrics(y[test], pred)
    bm = baseline_metrics(y[dev], y[test])
    test_df = pd.DataFrame({"issue_key": manifest.loc[test, "issue_key"].to_numpy(),
                            "actual": y[test], "predicted": pred,
                            "absolute_error": np.abs(y[test] - pred)})
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "cnn": cfg["cnn"],
                "best_config": best_cfg}, out_dir / "final_model.pt")
    test_metrics = {"project_id": int(project_id), "mae": tm["mae"], "mdae": tm["mdae"],
                    "test_count": int(test.sum()), "baseline_mae": bm["mae"],
                    "baseline_mdae": bm["mdae"], "encoder_checkpoint": cfg["sbert"]["checkpoint"],
                    "model_version": "sbert_cnn_v1", "seed": seed}
    with open(out_dir / "test_metrics.json", "w") as fh:
        json.dump(test_metrics, fh, indent=2)
    print(f"TEST: MAE={tm['mae']:.3f} MdAE={tm['mdae']:.3f} | baseline MAE={bm['mae']:.3f} "
          f"MdAE={bm['mdae']:.3f} | n={int(test.sum())}")
    return model, cv, test_df, test_metrics
