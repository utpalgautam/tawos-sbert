import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.expanduser("~/tawos-sbert")
SPLIT_DIR = f"{BASE_DIR}/data/processed/splits"
EMBED_DIR = f"{BASE_DIR}/data/processed/embeddings"

os.makedirs(EMBED_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16

print("Loading SBERT model...")
print("PyTorch:", torch.__version__)

# CPU is a safe starting point for an 8 GB Mac.
# We can test Apple's MPS accelerator later.
model = SentenceTransformer(MODEL_NAME, device="cpu")

print("Embedding dimension:", model.get_sentence_embedding_dimension())

for split_name in ["train", "validation", "test"]:
    csv_path = f"{SPLIT_DIR}/{split_name}.csv"
    output_path = f"{EMBED_DIR}/{split_name}_embeddings.npz"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    # Reuse previously completed embeddings
    if os.path.exists(output_path):
        print(f"\nSkipping {split_name}: embeddings already exist.")
        print("If you need to regenerate them, remove:", output_path)
        continue

    data = pd.read_csv(csv_path, low_memory=False)

    required = ["text", "story_point", "project_id"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"{split_name} is missing columns: {missing}")

    data = data[
        data["story_point"].notna()
        & data["project_id"].notna()
    ].copy()

    texts = data["text"].fillna("").astype(str).tolist()

    if not texts or any(not text.strip() for text in texts):
        raise ValueError(
            f"{split_name} contains no text or an empty text entry."
        )

    print(f"\nEncoding {split_name}: {len(texts):,} issues")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu"
    ).astype(np.float32)

    np.savez_compressed(
        output_path,
        X=embeddings,
        y=data["story_point"].to_numpy(dtype=np.float32),
        project_id=data["project_id"].to_numpy(dtype=np.int64),
        issue_key=data["issue_key"].fillna("").astype(str).to_numpy(),
        split_group=data["split_group"].astype(str).to_numpy()
    )

    print("Saved:", output_path)
    print("Embedding shape:", embeddings.shape)

print("\nAll split embedding jobs completed.")
