# Project Report: MULE (ID: 36)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:40:03

---

## 1. Project Profile & Data Summary
- **Total Issues:** 11816
- **Labeled Issues (Raw):** 4170
- **Unlabeled Issues (Raw):** 7646
- **Excluded Issues:** 0
- **Review Flags Logged:** 479
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 1.0
  - Median: 4.0
  - 75th percentile: 8.0
  - Mean: 4.458752997601919
  - Max: 60.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 2.9606 (± 0.0718)
- **Mean Validation MdAE:** 2.9103 (± 0.1425)
- **Median Best Epoch:** 5.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 13 | 2668 | 667 | 2.9916 | 2.9926 |
| 2 | 16 | 2668 | 667 | 2.8706 | 2.9663 |
| 3 | 1 | 2668 | 667 | 2.9084 | 2.9406 |
| 4 | 1 | 2668 | 667 | 3.0528 | 2.6584 |
| 5 | 5 | 2668 | 667 | 2.9794 | 2.9936 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 835
- **CNN Model MAE:** 3.1080
- **Baseline MAE (Training Median):** 2.9856
- **CNN Model MdAE:** 3.4195
- **Baseline MdAE (Training Median):** 3.0000
- **Training Median Value:** 4.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
