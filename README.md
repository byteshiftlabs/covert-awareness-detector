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

# Run the full training pipeline (downloads data and trains models)
./run_full_training.sh
```

## How It Works

The pipeline processes fMRI brain scans through several stages to classify consciousness states:

### 1. Data Input
We start with preprocessed brain scans from the Michigan Human Anesthesia fMRI Dataset. Each scan captures brain activity across hundreds of brain regions over time—think of it as a recording of which brain areas are "lighting up" together at each moment. We work with data from people who were scanned while transitioning between conscious and unconscious states under controlled anesthesia.

- **Source**: XCP-D preprocessed fMRI timeseries from OpenNeuro ds006623
- **What we have**: Brain activity measurements across 446 regions, recorded over time for 25 people in different states of consciousness

### 2. Feature Extraction
This is where we transform raw brain signals into meaningful patterns that distinguish consciousness from unconsciousness.

**Connectivity Matrices**: We measure how synchronized different brain regions are with each other. If two regions consistently activate together, they're "connected." This creates a map of functional connections across the entire brain—essentially a snapshot of how the brain's regions communicate.

**ISD (Integration-Segregation Difference)**: This is the key biomarker for consciousness. The theory is that conscious brains maintain a balance between two properties:
  - *Integration*: how efficiently information flows across the entire brain network
  - *Segregation*: how well distinct brain regions maintain specialized, local processing
  
When consciousness is lost, ISD decreases significantly. ISD quantifies this shift by computing the difference between these two properties (efficiency minus clustering).

**Network Summary Statistics**: We also compute basic graph properties from the connectivity matrix—mean degree, strength, and density—to capture the overall topology of the brain network at each state.

### 3. Dimensionality Reduction (PCA)
Raw connectivity data is massive—nearly 100,000 individual connections between brain regions. Most of this information is redundant or noisy. **Principal Component Analysis (PCA)** is like finding the "essence" of the data: it identifies the main patterns that explain most of the variation, compressing the data down to the most important features while throwing away the noise. This prevents the model from overfitting to irrelevant details.

### 4. Model Training
We use **XGBoost**, a powerful machine learning algorithm that builds an "ensemble" of decision trees. Think of it as training many simple classifiers that each learn different aspects of the data, then combining their votes for a final prediction.

**Handling Class Imbalance with SMOTE**: Our dataset has more unconscious examples than conscious ones (people spend more time sedated). SMOTE (Synthetic Minority Oversampling) creates synthetic examples of the underrepresented class, ensuring the model learns to recognize both states equally well rather than just guessing "unconscious" most of the time.

**Leave-One-Subject-Out Cross-Validation**: We train the model on data from all subjects except one, then test it on the left-out subject. We repeat this for every subject. This ensures the model learns general patterns about consciousness, not just memorizing specific individuals' brain signatures.

### 5. Prediction
Give the trained model a new brain scan, and it outputs a probability: how likely is this person to be conscious or unconscious? The model draws on the full set of features it learned during training—connectivity patterns compressed via PCA, ISD metrics, and network summary statistics—to make its prediction.


## Project Structure

```
src/
  config.py               # Dataset paths, subject list, scan parameters
  data_loader.py          # Load timeseries, motion filtering, connectivity matrices
  download_dataset.py     # OpenNeuro dataset downloader
  features.py             # ISD, graph metrics, connectivity feature extraction
  models.py               # Classifiers and LOSO-CV evaluation
  train.py                # Full training pipeline: XGBoost + PCA + SMOTE
  validate_model.py       # Overfitting checks and permutation tests

docs/                     # Sphinx documentation

run_full_training.sh      # Automated training pipeline (START HERE)
requirements.txt          # Core dependencies
```

## Models

| Model | Description |
|---|---|
| **XGBoost** (advanced) | Builds hundreds of small decision trees sequentially, where each tree corrects the mistakes of the previous ones. Uses PCA, SMOTE, and threshold tuning for the full pipeline. |
| **SVM** | Finds the optimal boundary that separates conscious from unconscious states in high-dimensional feature space. Uses an RBF kernel to capture nonlinear patterns, with balanced class weights to handle imbalanced data. |
| **Random Forest** | Trains 100 independent decision trees on random subsets of the data and averages their votes. Less prone to overfitting than a single tree, with balanced sampling to handle class imbalance. |
| **Logistic Regression** | The simplest baseline — fits a linear boundary between classes and outputs a probability. L2 regularization prevents overfitting. Useful as a sanity check: if complex models can't beat this, the signal may be weak. |

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
