"""Step 9: predict for missing-label issues of ONE project; stored separately."""
import pandas as pd
from .train_cv import predict


def predict_unlabeled(model, X_unl, unlabeled, cfg, out_path, device):
    pred = predict(model, X_unl, device)
    df = pd.DataFrame({"issue_key": unlabeled["issue_key"].to_numpy(),
                       "project_id": unlabeled["project_id"].to_numpy(),
                       "title": unlabeled["title"].to_numpy(),
                       "predicted_story_point": pred,
                       "encoder_checkpoint": cfg["sbert"]["checkpoint"],
                       "model_version": "sbert_cnn_v1"})
    df.to_csv(out_path, index=False)
    print("Unlabeled predictions saved:", len(df))
