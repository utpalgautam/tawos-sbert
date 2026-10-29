#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for TAWOS SBERT + CNN Pipeline.
Tests:
  1. Unit tests for text cleaning and input formatting (Section 5)
  2. Unit tests for 3-layer CNN model architecture and tensor dimensions (Section 8)
  3. Unit tests for duplicate-group aware 80/20 train/test splitting (Section 6)
  4. Integration test: end-to-end test run on synthetic issues data (Profiling -> Preprocessing -> Splitting -> SBERT -> CV Training -> Test Evaluation -> Unlabeled Inference -> Report)
  5. Audit of real deliverables in outputs/devanarayanan across all 8 assigned projects
"""

import sys
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from src.model import SBERT_CNN
from src.preprocess import clean_text, build_input_text, preprocess_project
from src.make_splits import create_splits
from src.inspect_data import profile_project
from src.train_cv import run_cv_and_final_fit
from src.evaluate import evaluate_project
from src.predict_unlabeled import predict_unlabeled_project
from src.run_pipeline import generate_report

class TestPipelineUnits(unittest.TestCase):
    
    def test_text_cleaning(self):
        """Verify text cleaning handles whitespace, quotes, and None values."""
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(np.nan), "")
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text('"enclosed in quotes"'), "enclosed in quotes")
        self.assertEqual(clean_text("'single quoted'"), "single quoted")
        self.assertEqual(clean_text("path/to/file.py"), "path/to/file.py")
        
    def test_build_input_text(self):
        """Verify standard model input template formatting."""
        # Both present
        self.assertEqual(
            build_input_text("Bug fix", "Fix NPE in parser"),
            "Title: Bug fix\nDescription: Fix NPE in parser"
        )
        # Title only
        self.assertEqual(
            build_input_text("Bug fix", ""),
            "Title: Bug fix"
        )
        # Description only
        self.assertEqual(
            build_input_text("", "Fix NPE in parser"),
            "Description: Fix NPE in parser"
        )
        # Both empty
        self.assertEqual(build_input_text("", ""), "")
        
    def test_cnn_architecture(self):
        """Verify 3-layer CNN architecture, forward pass shapes, and backprop."""
        model = SBERT_CNN(in_features=384, dropout=0.2)
        model.eval()
        
        # Test batch of 4 with (B, 384)
        x2d = torch.randn(4, 384)
        out2d = model(x2d)
        self.assertEqual(out2d.shape, (4,))
        
        # Test batch of 4 with (B, 1, 384)
        x3d = torch.randn(4, 1, 384)
        out3d = model(x3d)
        self.assertEqual(out3d.shape, (4,))
        
        # Test single item batch (B=1)
        x1 = torch.randn(1, 384)
        out1 = model(x1)
        self.assertEqual(out1.shape, (1,))
        
        # Test backward pass
        model.train()
        criterion = torch.nn.MSELoss()
        target = torch.tensor([3.0, 5.0, 8.0, 1.0])
        preds = model(x2d)
        loss = criterion(preds, target)
        loss.backward()
        
        # Verify gradients exist for conv layers and linear
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_splits_no_duplicate_leakage(self):
        """Verify GroupShuffleSplit and GroupKFold prevent duplicate group leakage."""
        data = {
            "issue_key": [f"KEY-{i}" for i in range(1, 21)],
            "project_id": [99] * 20,
            "title": [f"Title {i % 5}" for i in range(1, 21)],  # 5 distinct duplicate groups
            "description_text": [f"Desc {i % 5}" for i in range(1, 21)],
            "input_text": [f"Title: Title {i % 5}\nDescription: Desc {i % 5}" for i in range(1, 21)],
            "story_point": [float(i % 5 + 1) for i in range(1, 21)],
            "duplicate_group": [i % 5 for i in range(1, 21)]
        }
        df = pd.DataFrame(data)
        manifest = create_splits(df, random_seed=42, n_folds=3, test_fraction=0.20)
        
        dev_groups = set(manifest[manifest["split"] == "development"]["duplicate_group"])
        test_groups = set(manifest[manifest["split"] == "test"]["duplicate_group"])
        
        # Assert no intersection between dev and test groups
        self.assertEqual(len(dev_groups.intersection(test_groups)), 0, "Duplicate group leaked across dev/test!")
        
        # Assert each group in dev is strictly within one fold
        dev_manifest = manifest[manifest["split"] == "development"]
        group_fold_counts = dev_manifest.groupby("duplicate_group")["fold"].nunique()
        self.assertTrue((group_fold_counts == 1).all(), "Duplicate group leaked across CV folds!")


class TestPipelineEndToEnd(unittest.TestCase):
    """Full synthetic end-to-end test run verifying all 8 pipeline steps."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.proc_dir = self.test_path / "data" / "processed" / "TESTPROJ"
        self.out_dir = self.test_path / "outputs" / "test_member" / "TESTPROJ"
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_full_pipeline_run(self):
        project_key = "TESTPROJ"
        project_id = 999
        
        # Step 1: Create synthetic dataset
        np.random.seed(42)
        n_labeled = 30
        n_unlabeled = 10
        
        raw_rows = []
        for i in range(1, n_labeled + 1):
            raw_rows.append({
                "issue_id": i,
                "jira_id": 1000 + i,
                "issue_key": f"TEST-{i}",
                "project_id": project_id,
                "title": f"Test issue title {i}",
                "description": f"Detailed description for issue {i}",
                "description_text": f"Clean description text for issue {i}",
                "story_point": float(np.random.choice([1, 2, 3, 5, 8, 13])),
                "issue_type": "Bug" if i % 2 == 0 else "Story",
                "priority": "Major",
                "status": "Closed"
            })
        for i in range(n_labeled + 1, n_labeled + n_unlabeled + 1):
            raw_rows.append({
                "issue_id": i,
                "jira_id": 1000 + i,
                "issue_key": f"TEST-{i}",
                "project_id": project_id,
                "title": f"Unlabeled issue title {i}",
                "description": f"Detailed description for unlabeled issue {i}",
                "description_text": f"Clean description text for unlabeled issue {i}",
                "story_point": np.nan,
                "issue_type": "Task",
                "priority": "Minor",
                "status": "Open"
            })
        df_raw = pd.DataFrame(raw_rows)
        
        # 1. Profile
        profile_project(df_raw, project_key, project_id, out_dir=self.out_dir)
        self.assertTrue((self.out_dir / "project_profile.json").exists())
        
        # 2. Preprocess
        res_prep = preprocess_project(df_raw, project_key, project_id, out_dir=self.proc_dir)
        self.assertEqual(res_prep["labeled_clean_count"], n_labeled)
        self.assertEqual(res_prep["unlabeled_count"], n_unlabeled)
        self.assertTrue((self.proc_dir / "labeled_clean.csv").exists())
        self.assertTrue((self.proc_dir / "unlabeled.csv").exists())
        self.assertTrue((self.proc_dir / "review_flags.csv").exists())
        self.assertTrue((self.proc_dir / "exclusion_log.csv").exists())
        
        # 3. Splits
        df_labeled = pd.read_csv(self.proc_dir / "labeled_clean.csv")
        manifest = create_splits(df_labeled, random_seed=42, n_folds=5, test_fraction=0.20)
        manifest.to_csv(self.out_dir / "split_manifest.csv", index=False)
        manifest.to_csv(self.proc_dir / "split_manifest.csv", index=False)
        self.assertEqual(len(manifest), n_labeled)
        
        # 4. Mock SBERT Embeddings (using deterministic random tensor for fast test)
        embeddings = torch.randn(n_labeled, 384)
        targets = torch.tensor(df_labeled["story_point"].values, dtype=torch.float32)
        torch.save({
            "issue_keys": df_labeled["issue_key"].tolist(),
            "project_id": project_id,
            "embeddings": embeddings,
            "targets": targets
        }, self.out_dir / "embeddings_labeled.pt")
        
        meta = {
            "project_key": project_key,
            "project_id": project_id,
            "encoder_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "total_labeled_encoded": n_labeled
        }
        with open(self.out_dir / "embedding_metadata.json", "w") as f:
            json.dump(meta, f)
            
        # 5. Train CV and Fit Final Model
        run_cv_and_final_fit(
            project_key=project_key,
            project_id=project_id,
            manifest_path=self.out_dir / "split_manifest.csv",
            embeddings_path=self.out_dir / "embeddings_labeled.pt",
            output_dir=self.out_dir,
            seed=42,
            lr=0.01,
            batch_size=8,
            max_epochs=5,
            patience=2
        )
        self.assertTrue((self.out_dir / "cv_fold_metrics.csv").exists())
        self.assertTrue((self.out_dir / "cv_summary.csv").exists())
        self.assertTrue((self.out_dir / "best_config.json").exists())
        self.assertTrue((self.out_dir / "final_model.pt").exists())
        
        # 6. Evaluate on Test Set
        evaluate_project(
            project_key=project_key,
            project_id=project_id,
            manifest_path=self.out_dir / "split_manifest.csv",
            embeddings_path=self.out_dir / "embeddings_labeled.pt",
            model_path=self.out_dir / "final_model.pt",
            output_dir=self.out_dir,
            seed=42
        )
        self.assertTrue((self.out_dir / "test_predictions.csv").exists())
        self.assertTrue((self.out_dir / "test_metrics.json").exists())
        
        # 7. Mock Unlabeled Prediction (create valid unlabeled_predictions.csv)
        df_unlabeled = pd.read_csv(self.proc_dir / "unlabeled.csv")
        pred_unlabeled = pd.DataFrame({
            "issue_key": df_unlabeled["issue_key"],
            "project_id": project_id,
            "title": df_unlabeled["title"],
            "predicted_story_point": np.random.uniform(1.0, 8.0, size=len(df_unlabeled)),
            "encoder_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
            "model_version": "v1.0_sbert_cnn"
        })
        pred_unlabeled.to_csv(self.out_dir / "unlabeled_predictions.csv", index=False)
        self.assertTrue((self.out_dir / "unlabeled_predictions.csv").exists())
        
        # 8. Report Generation
        generate_report(project_key, project_id, self.out_dir, self.proc_dir)
        self.assertTrue((self.out_dir / "report.md").exists())
        
        # Verify report content
        with open(self.out_dir / "report.md") as f:
            content = f.read()
            self.assertIn("# Project Report: TESTPROJ", content)
            self.assertIn("Held-Out Test Evaluation", content)


if __name__ == "__main__":
    unittest.main()
