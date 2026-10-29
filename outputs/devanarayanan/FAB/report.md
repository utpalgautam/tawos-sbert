# Project Report: FAB (ID: 24)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:36:35

---

## 1. Project Profile & Data Summary
- **Total Issues:** 13682
- **Labeled Issues (Raw):** 636
- **Unlabeled Issues (Raw):** 13046
- **Excluded Issues:** 0
- **Review Flags Logged:** 160
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 1.0
  - Median: 2.0
  - 75th percentile: 3.0
  - Mean: 2.690251572327044
  - Max: 40.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 1.7233 (± 0.1766)
- **Mean Validation MdAE:** 1.0311 (± 0.2082)
- **Median Best Epoch:** 2.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 2 | 406 | 102 | 1.8626 | 1.0925 |
| 2 | 3 | 406 | 102 | 1.5911 | 1.2068 |
| 3 | 6 | 406 | 102 | 1.8728 | 1.2289 |
| 4 | 2 | 407 | 101 | 1.4819 | 0.7635 |
| 5 | 2 | 407 | 101 | 1.8082 | 0.8637 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 128
- **CNN Model MAE:** 1.4899
- **Baseline MAE (Training Median):** 1.4141
- **CNN Model MdAE:** 1.0610
- **Baseline MdAE (Training Median):** 1.0000
- **Training Median Value:** 2.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
