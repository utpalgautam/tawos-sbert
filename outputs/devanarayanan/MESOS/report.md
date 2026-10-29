# Project Report: MESOS (ID: 4)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:34:37

---

## 1. Project Profile & Data Summary
- **Total Issues:** 10157
- **Labeled Issues (Raw):** 3272
- **Unlabeled Issues (Raw):** 6885
- **Excluded Issues:** 0
- **Review Flags Logged:** 246
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 2.0
  - Median: 3.0
  - 75th percentile: 5.0
  - Mean: 3.223044009779951
  - Max: 40.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 1.4931 (± 0.0605)
- **Mean Validation MdAE:** 0.9967 (± 0.0108)
- **Median Best Epoch:** 4.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 25 | 2093 | 524 | 1.5853 | 0.9983 |
| 2 | 4 | 2093 | 524 | 1.5166 | 0.9782 |
| 3 | 3 | 2094 | 523 | 1.4451 | 1.0047 |
| 4 | 17 | 2094 | 523 | 1.4372 | 1.0041 |
| 5 | 2 | 2094 | 523 | 1.4812 | 0.9983 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 655
- **CNN Model MAE:** 1.4367
- **Baseline MAE (Training Median):** 1.4350
- **CNN Model MdAE:** 0.9919
- **Baseline MdAE (Training Median):** 1.0000
- **Training Median Value:** 3.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
