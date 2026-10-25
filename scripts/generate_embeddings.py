#!/usr/bin/env python3

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    config_path = BASE_DIR / "configs/problem5.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["sbert"]

    input_path = BASE_DIR / data_cfg["processed_dir"] / "labeled_clean.csv"
    output_dir = BASE_DIR / data_cfg["embeddings_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "sbert_embeddings.npy"
    metadata_path = output_dir / "embedding_metadata.csv"

    df = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        data_cfg["issue_key_column"],
        data_cfg["project_column"],
        data_cfg["target_column"],
        "input_text",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["input_text"].isna().any():
        raise ValueError("Missing input text detected.")

    if df[data_cfg["issue_key_column"]].duplicated().any():
        raise ValueError("Duplicate issue keys detected in labeled data.")

    # Preserve the exact row order in the metadata file.
    df = df.reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Rows to encode:", len(df))
    print("Device:", device)
    print("Checkpoint:", model_cfg["checkpoint"])
    print("Expected embedding dimension:", model_cfg["expected_embedding_dim"])

    model = SentenceTransformer(
        model_cfg["checkpoint"],
        device=device,
    )

    batch_size = int(model_cfg.get("batch_size", 32))

    embeddings = model.encode(
        df["input_text"].astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=bool(
            model_cfg.get("normalize_embeddings", True)
        ),
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_dim = int(model_cfg["expected_embedding_dim"])

    if embeddings.shape != (len(df), expected_dim):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}; "
            f"expected ({len(df)}, {expected_dim})"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    # Hash the exact input text for reproducibility and audit purposes.
    text_hashes = df["input_text"].astype(str).map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )

    metadata = pd.DataFrame({
        "issue_key": df[data_cfg["issue_key_column"]].astype(str),
        "project_id": df[data_cfg["project_column"]],
        "story_point": df[data_cfg["target_column"]],
        "input_text_sha256": text_hashes,
    })

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)

    print("\nEmbedding generation complete.")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print("Embedding norms (first 5):",
          np.linalg.norm(embeddings[:5], axis=1).round(4).tolist())
    print("Saved embeddings:", embeddings_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()