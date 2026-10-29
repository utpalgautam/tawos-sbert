# Project Report: JRACLOUD (ID: 44)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:27:25

---

## 1. Project Profile & Data Summary
- **Total Issues:** 25669
- **Labeled Issues (Raw):** 361
- **Unlabeled Issues (Raw):** 25308
- **Excluded Issues:** 0
- **Review Flags Logged:** 11
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 2.0
  - Median: 4.0
  - 75th percentile: 4.0
  - Mean: 12.246537396121884
  - Max: 3030.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 1.2588 (± 0.1976)
- **Mean Validation MdAE:** 0.6305 (± 0.8939)
- **Median Best Epoch:** 14.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 19 | 230 | 58 | 1.2125 | 0.0151 |
| 2 | 10 | 230 | 58 | 1.0156 | 0.0658 |
| 3 | 14 | 230 | 58 | 1.2140 | 0.0418 |
| 4 | 8 | 231 | 57 | 1.5618 | 2.0569 |
| 5 | 15 | 231 | 57 | 1.2900 | 0.9728 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 73
- **CNN Model MAE:** 42.9080
- **Baseline MAE (Training Median):** 42.9041
- **CNN Model MdAE:** 1.9833
- **Baseline MdAE (Training Median):** 2.0000
- **Training Median Value:** 4.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
