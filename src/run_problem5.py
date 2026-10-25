"""One project, end to end.

  python -m src.run_problem5 --project-id 28 --member Utpal
  python -m src.run_problem5 --project-id 28 --member Utpal --max-epochs 3   # smoke test
"""
import argparse
import json
import platform
import sys
import numpy as np
import pandas as pd
from .common import load_config, output_dir
from .preprocess import preprocess_project
from .make_splits import make_splits
from .embed_sbert import embed_texts, write_embedding_metadata


def main(encoder=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--config", default="configs/problem5_sbert_cnn.yaml")
    ap.add_argument("--max-epochs", type=int, default=None, help="override for smoke tests only")
    ap.add_argument("--skip-unlabeled", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    print(f"== Project {a.project_id} ==")
    key, pdir = preprocess_project(cfg, a.project_id)
    odir = output_dir(a.member, key)
    labeled = pd.read_csv(pdir / "labeled_clean.csv")
    unlabeled = pd.read_csv(pdir / "unlabeled.csv")

    manifest = make_splits(labeled, cfg, odir / "split_manifest.csv")

    emb_l, h = embed_texts(labeled["input_text"].tolist(), cfg, pdir / "emb_labeled", encoder)
    write_embedding_metadata(pdir / "embedding_metadata.json", cfg, emb_l.shape[1],
                             len(labeled), len(unlabeled), h)
    X = emb_l.astype("float32")
    y = labeled["story_point"].to_numpy(dtype="float32")

    from .train_cv import run_cv_and_final, get_device
    from .evaluate import make_plots
    device = get_device()
    print("Device:", device)
    model, cv, test_df, tm = run_cv_and_final(X, y, manifest, cfg, a.project_id, odir, device, a.max_epochs)
    make_plots(test_df, cv, y, odir)

    if not a.skip_unlabeled and len(unlabeled):
        from .predict_unlabeled import predict_unlabeled
        emb_u, _ = embed_texts(unlabeled["input_text"].tolist(), cfg, pdir / "emb_unlabeled", encoder)
        predict_unlabeled(model, emb_u.astype("float32"), unlabeled, cfg,
                          odir / "unlabeled_predictions.csv", device)

    import torch, sklearn
    meta = {"project_id": a.project_id, "project_key": key, "member": a.member,
            "seed": cfg["experiment"]["random_seed"], "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "encoder_checkpoint": cfg["sbert"]["checkpoint"], "dataset_path": cfg["data"]["path"],
            "max_epochs_override": a.max_epochs, "config": cfg}
    with open(odir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("Done. Outputs:", odir)


if __name__ == "__main__":
    main()
