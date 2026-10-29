# Project Report: JSWCLOUD (ID: 19)
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** 2026-10-26 17:35:55

---

## 1. Project Profile & Data Summary
- **Total Issues:** 11702
- **Labeled Issues (Raw):** 318
- **Unlabeled Issues (Raw):** 11384
- **Excluded Issues:** 0
- **Review Flags Logged:** 69
- **Story Point Statistics (Raw Labeled):**
  - Min: 0.0
  - 25th percentile: 2.0
  - Median: 3.0
  - 75th percentile: 5.0
  - Mean: 6.138364779874214
  - Max: 688.0

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** 4.9707 (± 6.2812)
- **Mean Validation MdAE:** 1.4932 (± 0.3310)
- **Median Best Epoch:** 8.0

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
| 1 | 10 | 203 | 51 | 16.1963 | 1.6872 |
| 2 | 9 | 203 | 51 | 2.5133 | 1.8510 |
| 3 | 8 | 203 | 51 | 2.2361 | 1.6232 |
| 4 | 6 | 203 | 51 | 1.7515 | 1.2602 |
| 5 | 7 | 204 | 50 | 2.1563 | 1.0444 |

---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** 64
- **CNN Model MAE:** 1.8492
- **Baseline MAE (Training Median):** 1.8438
- **CNN Model MdAE:** 1.8192
- **Baseline MdAE (Training Median):** 2.0000
- **Training Median Value:** 3.0

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
