#!/usr/bin/env python3
"""
Embed SBERT Module for TAWOS SBERT + CNN Pipeline.
Implements Step 5 of COMMON_PIPELINE.md:
  - Checkpoint: sentence-transformers/all-MiniLM-L6-v2
  - Frozen encoder in evaluation mode
  - Verify embedding dimension (384) and check for NaN/Inf
  - Device: Apple Silicon MPS if available, else CPU
  - Saves cached embeddings and embedding_metadata.json
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_embeddings(texts: list, model: SentenceTransformer, batch_size: int = 64, show_progress: bool = True):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
        normalize_embeddings=False
    )
    # Check shape and NaN/Inf
    dim = embeddings.shape[1]
    if dim != 384:
        raise ValueError(f"Expected embedding dimension 384, but got {dim}")
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Detected NaN or Inf in embeddings!")
    return embeddings.cpu()

def embed_project_data(
    project_key: str,
    project_id: int,
    processed_dir: Path,
    output_dir: Path,
    checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labeled_clean_path = processed_dir / "labeled_clean.csv"
    if not labeled_clean_path.exists():
        raise FileNotFoundError(f"Missing {labeled_clean_path}")
        
    df_labeled = pd.read_csv(labeled_clean_path)
    
    if device is None:
        device = get_device()
        
    print(f"[{project_key}] Loading SBERT model: {checkpoint} on {device}...")
    model = SentenceTransformer(checkpoint, device=str(device))
    model.eval()
    
    print(f"[{project_key}] Encoding {len(df_labeled)} labeled issues...")
    texts = df_labeled["input_text"].fillna("").tolist()
    emb_tensor = compute_embeddings(texts, model, batch_size=batch_size)
    
    # Save labeled embeddings
    target_tensor = torch.tensor(df_labeled["story_point"].values, dtype=torch.float32)
    labeled_cache = {
        "issue_keys": df_labeled["issue_key"].tolist(),
        "project_id": project_id,
        "embeddings": emb_tensor,
        "targets": target_tensor
    }
    
    emb_path = output_dir / "embeddings_labeled.pt"
    torch.save(labeled_cache, emb_path)
    print(f"[{project_key}] Saved labeled embeddings to {emb_path}")
    
    # Save metadata
    meta = {
        "project_key": project_key,
        "project_id": project_id,
        "encoder_checkpoint": checkpoint,
        "embedding_dim": int(emb_tensor.shape[1]),
        "total_labeled_encoded": int(emb_tensor.shape[0]),
        "device": str(device),
        "preprocessing_version": "v1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(output_dir / "embedding_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    # Also save metadata in processed_dir
    with open(processed_dir / "embedding_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Encode issue texts using SBERT")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_base", default="outputs")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None)
    parser.add_argument("--checkpoint", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    alloc_df = pd.read_csv(args.allocation_path)
    if args.member:
        alloc_df = alloc_df[alloc_df["assigned_member"] == args.member]
    if args.project_key:
        alloc_df = alloc_df[alloc_df["project_key"] == args.project_key]
        
    for _, row in alloc_df.iterrows():
        p_key = str(row["project_key"])
        p_id = int(row["project_id"])
        p_proc_dir = Path(args.processed_dir) / p_key
        p_out_dir = Path(args.output_base) / args.member / p_key
        embed_project_data(p_key, p_id, p_proc_dir, p_out_dir, checkpoint=args.checkpoint, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
