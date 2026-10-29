#!/usr/bin/env python3
"""
Predict Unlabeled Issues Module.
Implements Step 9 of COMMON_PIPELINE.md:
  - Loads unlabeled.csv for the project
  - Encodes issue text using the same frozen SBERT checkpoint
  - Runs final_model.pt
  - Saves unlabeled_predictions.csv:
      issue_key,project_id,title,predicted_story_point,encoder_checkpoint,model_version
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.model import SBERT_CNN

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def predict_unlabeled_project(
    project_key: str,
    project_id: int,
    processed_dir: Path,
    output_dir: Path,
    checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
):
    unlabeled_path = processed_dir / "unlabeled.csv"
    model_path = output_dir / "final_model.pt"
    
    if not unlabeled_path.exists():
        print(f"[{project_key}] No unlabeled.csv found.")
        return
    if not model_path.exists():
        print(f"[{project_key}] No final_model.pt found at {model_path}.")
        return
        
    df_unlabeled = pd.read_csv(unlabeled_path)
    if df_unlabeled.empty:
        print(f"[{project_key}] unlabeled.csv is empty.")
        empty_df = pd.DataFrame(columns=[
            "issue_key", "project_id", "title", "predicted_story_point", "encoder_checkpoint", "model_version"
        ])
        empty_df.to_csv(output_dir / "unlabeled_predictions.csv", index=False)
        return
        
    device = get_device()
    print(f"[{project_key}] Generating predictions for {len(df_unlabeled)} unlabeled issues...")
    
    # Load SBERT model
    sbert_model = SentenceTransformer(checkpoint, device=str(device))
    sbert_model.eval()
    
    # Encode texts
    texts = df_unlabeled["input_text"].fillna("").tolist()
    # Batch encode
    all_preds = []
    
    # Load CNN
    cnn = SBERT_CNN(in_features=384, dropout=0.2).to(device)
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    cnn.eval()
    
    # Encode and predict in chunks to save memory
    chunk_size = 512
    with torch.no_grad():
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i : i + chunk_size]
            emb = sbert_model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=False
            )
            emb = emb.to(device)
            preds = cnn(emb).cpu().numpy().tolist()
            all_preds.extend(preds)
            
    df_preds = pd.DataFrame({
        "issue_key": df_unlabeled["issue_key"],
        "project_id": project_id,
        "title": df_unlabeled["title"],
        "predicted_story_point": all_preds,
        "encoder_checkpoint": checkpoint,
        "model_version": "v1.0_sbert_cnn"
    })
    
    out_file = output_dir / "unlabeled_predictions.csv"
    df_preds.to_csv(out_file, index=False)
    print(f"[{project_key}] Saved {len(df_preds)} unlabeled predictions to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Predict story points for unlabeled issues")
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
        predict_unlabeled_project(
            project_key=p_key,
            project_id=p_id,
            processed_dir=p_proc_dir,
            output_dir=p_out_dir,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
