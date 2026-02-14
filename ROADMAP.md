# Covert Awareness Detector — Roadmap

> **Disclosure:** This project was developed with AI assistance under human supervision.
> It is actively being improved, validated, and documented.

---

## Completed

### Data Pipeline
- [x] Automated dataset downloader (OpenNeuro ds006623, XCP-D derivatives only)
- [x] Data loader matching the original paper's MATLAB preprocessing
- [x] Motion censoring (FD < 0.8)
- [x] 7-condition segmentation per subject using LOR/ROR timing
- [x] 446-ROI connectivity matrices (Pearson correlation)

### Feature Extraction
- [x] ISD (Integration-Segregation Difference) — paper's key metric
- [x] Graph topology metrics (degree, strength, density)
- [x] Statistical connectivity features (mean, std, skew, kurtosis, percentiles)
- [x] Full connectivity upper-triangle (99,235 features) with PCA reduction

### Machine Learning Models
- [x] Logistic Regression (baseline)
- [x] Random Forest with balanced class weights
- [x] SVM with RBF kernel
- [x] XGBoost with SMOTE oversampling + threshold tuning (best performer)
- [x] Leave-One-Subject-Out Cross-Validation (25 folds)
- [x] Per-subject deviation features

### Results (~approximate)
- [x] Unconscious state detection: ~96% recall
- [x] Balanced accuracy: ~93%
- [x] ROC-AUC: ~0.98

### Infrastructure
- [x] Full training pipeline script (`run_full_training.sh`)
- [x] Quick training mode (5 subjects)
- [x] Progress bars with ETA
- [x] Sphinx documentation
- [x] Overfitting validation script (permutation test, holdout)

---

## In Progress

- [ ] Thorough code review and cleanup
- [ ] Reviewing all documentation for accuracy
- [ ] Understanding and verifying the ML techniques used (ongoing learning)

---

## Planned

### Short Term
- [ ] Add unit tests for data loader and feature extraction
- [ ] Pin dependency versions for reproducibility
- [ ] Visualizations: confusion matrices, ROC curves, brain region maps
- [ ] GitHub Actions CI for automated testing

### Medium Term
- [ ] Cross-validation stability analysis
- [ ] Feature importance analysis (SHAP values)
- [ ] Compare additional preprocessing strategies
- [ ] Try SVM (which may be more suitable for this sample size)

### Ideas for the Future
- [ ] Test on external disorders-of-consciousness datasets
- [ ] Dynamic functional connectivity (sliding window)
- [ ] Deep learning models (CNN, GNN) if more data becomes available
- [ ] REST API for model inference

---

## Known Limitations

- **Small sample size** (25 subjects) — limits generalizability
- **Class imbalance** (6:1 conscious/unconscious) — mitigated with SMOTE
- **Single dataset** — no external validation yet
- **AI-generated code** — under active review and testing

---

**Last Updated**: February 2026
