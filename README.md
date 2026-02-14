# Covert Awareness Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/Dataset-OpenNeuro%20ds006623-orange.svg)](https://openneuro.org/datasets/ds006623)

Machine learning pipeline for detecting covert consciousness from fMRI functional connectivity during anesthesia.

> **Disclosure:** This software was developed with AI assistance under human supervision. It is actively being improved, validated, and documented.

## Overview

Detects hidden awareness in behaviorally unresponsive patients using the [Michigan Human Anesthesia fMRI Dataset](https://openneuro.org/datasets/ds006623) (OpenNeuro ds006623). Implements machine learning classifiers that distinguish consciousness states during propofol sedation based on functional connectivity patterns across 446 brain regions.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/cmelnulabs/covert-awareness-detector.git
cd covert-awareness-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset (~1.8 GB of preprocessed XCP-D derivatives)
python src/download_dataset.py --output-dir ../datasets/openneuro/ds006623

# Run the full training pipeline
./run_full_training.sh

# Or train directly
python src/train.py
```

## How It Works

The pipeline processes fMRI brain scans through several stages to classify consciousness states:

### 1. Data Input
- **Source**: XCP-D preprocessed fMRI timeseries from OpenNeuro ds006623
- **Format**: ~200 timepoints × 446 brain regions (ROIs) per scan
- **Atlas**: 4S456Parcels (400 cortical + 56 subcortical regions)
- **Subjects**: 25 individuals, 7 conditions each (conscious and unconscious states)

### 2. Feature Extraction
- **Connectivity Matrices**: Compute Pearson correlations between all 446 ROI pairs → 99,235 connections
- **ISD (Integration-Segregation Difference)**: 
  - Regress out principal eigenvector to remove global signal
  - Calculate multilevel efficiency (integration)
  - Calculate multilevel clustering (segregation)  
  - ISD = efficiency - clustering (key consciousness biomarker)
- **Graph Metrics**: Network topology features (centrality, modularity, etc.)
- **Statistical Features**: Connectivity distribution properties (skewness, kurtosis)

### 3. Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduces 99,280 features → 50 components
- Retains essential variance while preventing overfitting on small sample size (n=25)

### 4. Model Training
- **Algorithm**: XGBoost gradient boosting classifier
- **Class Balance**: SMOTE oversampling to handle 6:1 unconscious:conscious imbalance
- **Optimization**: Threshold tuning (optimal at 0.85) to maximize balanced accuracy
- **Validation**: Leave-One-Subject-Out Cross-Validation (LOSO-CV) - 25 folds, one per subject

### 5. Prediction
- New scan → Same feature pipeline → Trained model → Probability score
- Score > 0.85 → Unconscious | Score < 0.85 → Conscious
- **Key Discovery**: Conscious states show higher ISD (balanced integration/segregation), while unconscious states show collapsed segregation

## Project Structure

```
src/
  config.py          # Dataset paths, subject list, scan parameters
  data_loader.py     # Load timeseries, motion filtering, connectivity matrices
  download_dataset.py # OpenNeuro dataset downloader
  features.py        # ISD, graph metrics, connectivity feature extraction
  models.py          # Classifiers and LOSO-CV evaluation
  train.py           # Full training pipeline: XGBoost + PCA + SMOTE
  validate_model.py  # Overfitting checks and permutation tests

docs/                       # Sphinx documentation

run_full_training.sh    # Automated training pipeline
run_quick_training.sh   # Fast 5-subject test
requirements.txt        # Core dependencies
```

## Models

| Model | Description |
|---|---|
| **XGBoost** (advanced) | Full connectivity + PCA + SMOTE + threshold tuning |
| **SVM** | RBF kernel, balanced class weights |
| **Random Forest** | 100 trees, balanced sampling |
| **Logistic Regression** | L2-regularized baseline |

## Acknowledgments

**Original Research**: Huang, Hudetz, Mashour et al. — University of Michigan  
**Dataset**: [OpenNeuro ds006623](https://openneuro.org/datasets/ds006623) (CC0 Public Domain)  
**MATLAB Reference**: [Jang et al.](https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation)  
**This Implementation**: Independent Python ML pipeline by cmelnulabs, built with AI assistance

## Citation

```bibtex
@article{huang2018covert,
  title     = {Brain imaging reveals covert consciousness during behavioral unresponsiveness},
  author    = {Huang, Zirui and others},
  journal   = {Scientific Reports},
  volume    = {8},
  pages     = {13195},
  year      = {2018},
  doi       = {10.1038/s41598-018-31436-z}
}
```

## License

MIT License — see [LICENSE](LICENSE).  
Dataset: CC0 (Public Domain).

---

**Documentation**: [docs/](docs/) · **Updated**: February 2026
