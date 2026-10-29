# Project Report: IS (ID: 26)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:36:53

---

## 1. Project Profile & Data Summary
- **Total Issues:** 1531
- **Labeled Issues (Raw):** 720
- **Unlabeled Issues (Raw):** 811
- **Excluded Issues:** 0
- **Review Flags Logged:** 181
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 2.0
  - Median: 3.0
  - 75th percentile: 5.0
  - Mean: 3.8208333333333333
  - Max: 13.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 1.7147 (± 0.1365)
- **Mean Validation MdAE:** 1.2799 (± 0.0977)
- **Median Best Epoch:** 5.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 5 | 460 | 116 | 1.5733 | 1.2969 |
| 2 | 3 | 461 | 115 | 1.9128 | 1.1374 |
| 3 | 5 | 461 | 115 | 1.7931 | 1.2390 |
| 4 | 5 | 461 | 115 | 1.6440 | 1.3304 |
| 5 | 5 | 461 | 115 | 1.6503 | 1.3956 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 144
- **CNN Model MAE:** 1.8087
- **Baseline MAE (Training Median):** 1.6806
- **CNN Model MdAE:** 1.4872
- **Baseline MdAE (Training Median):** 1.0000
- **Training Median Value:** 3.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
