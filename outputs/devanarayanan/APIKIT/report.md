# Project Report: APIKIT (ID: 35)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:37:06

---

## 1. Project Profile & Data Summary
- **Total Issues:** 886
- **Labeled Issues (Raw):** 473
- **Unlabeled Issues (Raw):** 413
- **Excluded Issues:** 0
- **Review Flags Logged:** 8
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 1.0
  - Median: 3.0
  - 75th percentile: 5.0
  - Mean: 3.4186046511627906
  - Max: 21.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 1.9141 (± 0.2094)
- **Mean Validation MdAE:** 1.9289 (± 0.0600)
- **Median Best Epoch:** 8.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 8 | 302 | 76 | 1.7852 | 1.9676 |
| 2 | 10 | 302 | 76 | 1.7270 | 1.9520 |
| 3 | 8 | 302 | 76 | 2.2028 | 1.9382 |
| 4 | 8 | 303 | 75 | 1.7861 | 1.9632 |
| 5 | 8 | 303 | 75 | 2.0693 | 1.8235 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 95
- **CNN Model MAE:** 1.8633
- **Baseline MAE (Training Median):** 1.8526
- **CNN Model MdAE:** 1.9758
- **Baseline MdAE (Training Median):** 2.0000
- **Training Median Value:** 3.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
