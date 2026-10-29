# Project Report: APSTUD (ID: 8)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:35:18

---

## 1. Project Profile & Data Summary
- **Total Issues:** 8135
- **Labeled Issues (Raw):** 890
- **Unlabeled Issues (Raw):** 7245
- **Excluded Issues:** 0
- **Review Flags Logged:** 112
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 3.0
  - Median: 8.0
  - 75th percentile: 8.0
  - Mean: 8.149438202247191
  - Max: 100.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 4.7228 (± 0.6280)
- **Mean Validation MdAE:** 3.6901 (± 1.3600)
- **Median Best Epoch:** 7.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 7 | 569 | 143 | 5.2207 | 4.9894 |
| 2 | 3 | 569 | 143 | 5.3407 | 3.6884 |
| 3 | 3 | 570 | 142 | 4.9257 | 1.7949 |
| 4 | 14 | 570 | 142 | 4.2000 | 4.9737 |
| 5 | 19 | 570 | 142 | 3.9270 | 3.0043 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 178
- **CNN Model MAE:** 5.1002
- **Baseline MAE (Training Median):** 4.9326
- **CNN Model MdAE:** 4.6189
- **Baseline MdAE (Training Median):** 5.0000
- **Training Median Value:** 8.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
