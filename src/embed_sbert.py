import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import yaml
from sentence_transformers import SentenceTransformer

CONFIG = Path("configs/problem5_sbert_cnn.yaml")
ALLOC = Path("project_allocation.csv")
MEMBER = "simran"
PROC = Path("data/processed")
OUT = Path("outputs") / MEMBER

cfg = yaml.safe_load(CONFIG.read_text())
checkpoint = cfg["sbert"]["checkpoint"]
expected_dim = cfg["sbert"]["expected_embedding_dim"]
batch_size = cfg["sbert"]["encode_batch_size"]
prep_version = cfg["experiment"]["preprocessing_version"]
do_unlabeled = "--unlabeled" in sys.argv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Checkpoint: {checkpoint}")
model = SentenceTransformer(checkpoint, device=device)
model.eval()  # frozen encoder, no fine-tuning


def encode(texts):
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=True)
    assert emb.ndim == 2 and emb.shape[1] == expected_dim, \
        f"Unexpected embedding shape {emb.shape}, expected dim {expected_dim}"
    assert np.isfinite(emb).all(), "NaN/Inf found in embeddings"
    return emb.astype(np.float32)


alloc = pd.read_csv(ALLOC)
mine = alloc[alloc["assigned_member"].str.lower() == MEMBER]

for _, row in mine.iterrows():
    key, pid = row["project_key"], int(row["project_id"])
    d = PROC / key
    meta_dir = OUT / key
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "embedding_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t0 = time.time()

    sets = [("labeled", "labeled_clean.csv")]
    if do_unlabeled:
        sets.append(("unlabeled", "unlabeled.csv"))

    for name, fname in sets:
        df = pd.read_csv(d / fname)
        texts = df["input_text"].fillna("").tolist()
        n_blank = sum(1 for t in texts if t == "")
        print(f"\n{key} [{name}]: {len(texts)} rows (blank text: {n_blank})")
        emb = encode(texts)
        np.save(d / f"emb_{name}.npy", emb)
        df[["issue_key", "project_id"]].assign(
            row_index=np.arange(len(df)), set=name,
            preprocessing_version=prep_version
        ).to_csv(d / f"emb_{name}_index.csv", index=False)
        meta[f"n_{name}"] = int(len(df))
        meta[f"n_{name}_blank_text"] = int(n_blank)
        print(f"  saved emb_{name}.npy shape={emb.shape}")

    meta.update({
        "project_id": pid,
        "project_key": key,
        "encoder_checkpoint": checkpoint,
        "embedding_dim": int(expected_dim),
        "fine_tuned": False,
        "encode_batch_size": batch_size,
        "preprocessing_version": prep_version,
        "device": device,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "note": "model revision hash not pinned",
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  {key} done in {time.time() - t0:.1f}s")

print("\nAll done.")