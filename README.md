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
git clone https://github.com/byteshiftlabs/covert-awareness-detector.git
cd covert-awareness-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full training pipeline (downloads data and trains models)
./run_full_training.sh
```

## How It Works

The pipeline processes fMRI brain scans through several stages to classify consciousness states:

### 1. Data Input & Processing

#### Dataset Authors' Pipeline (fMRIPrep → XCP-D)
Raw fMRI scans were preprocessed and post-processed by dataset authors:

- **fMRIPrep (Preprocessing)**: Motion correction, distortion correction, normalization to MNI space
- **XCP-D (Post-processing)**: Brain parcellation into 456 regions (4S456Parcels atlas), regional timeseries extraction, motion quality metrics

*We download these XCP-D derivatives (regional timeseries + motion parameters) from OpenNeuro.*

#### Our Analysis Steps
Additional processing for machine learning:

1. **Motion filtering**: Remove timepoints where head movement (FD) ≥ 0.8 mm
2. **ROI selection**: Use first 446 of 456 regions (following reference MATLAB implementation)
3. **Temporal segmentation**: Split into 7 consciousness conditions using LOR/ROR timing
4. **Connectivity**: Compute 446×446 Pearson correlation matrices (diagonal = 0)

### 2. Feature Extraction
This is where we transform processed regional timeseries into meaningful patterns that distinguish consciousness from unconsciousness.

**Connectivity Matrices**: From the motion-filtered timeseries, we compute how synchronized different brain regions are with each other using Pearson correlation. If two regions consistently activate together, they're "connected." This creates a functional connectivity matrix (446×446)—essentially a snapshot of how the brain's regions communicate. The diagonal is set to zero (a region's self-correlation is uninformative).

**ISD**: This is a brain network metric computed as the difference between efficiency (how well information flows across the entire brain network) and clustering (how much brain regions form tight local groups). Higher ISD values are associated with conscious states.

**Network Summary Statistics**: We also compute basic graph properties from the connectivity matrix—mean degree, strength, and density—to capture the overall topology of the brain network at each state.

### 3. Dimensionality Reduction (PCA)
Raw connectivity data is massive—nearly 100,000 individual connections between brain regions. Most of this information is redundant or noisy. **Principal Component Analysis (PCA)** identifies patterns that explain most of the variation, compressing the data down to fewer features while discarding noise. This prevents the model from overfitting to irrelevant details.

### 4. Model Training
We use **XGBoost**, a machine learning algorithm that builds an "ensemble" of decision trees. It trains multiple classifiers that each learn different aspects of the data, then combines their outputs for a final prediction.

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
  train.py                # Full training pipeline: XGBoost + PCA + SMOTE
  validate_model.py       # Overfitting checks and permutation tests

docs/                     # Sphinx documentation

run_full_training.sh      # Automated training pipeline (START HERE)
requirements.txt          # Core dependencies
```

## Model

The default training pipeline (`src/train.py` / `./run_full_training.sh`) trains and validates the **XGBoost** classifier only (full connectivity + PCA + SMOTE + threshold tuning).


## Acknowledgments

**Original Research**: Huang, Hudetz, Mashour et al. — University of Michigan  
**Dataset**: [OpenNeuro ds006623](https://openneuro.org/datasets/ds006623) (CC0 Public Domain)  
**MATLAB Reference**: [Jang et al.](https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation)  
**This Implementation**: Independent Python ML pipeline by byteshiftlabs, built with AI assistance

### Citation

This project builds upon the research published in:

**Huang, Z., Tarnal, V., Fotiadis, P., Vlisides, P. E., Janke, E. L., Puglia, M., McKinney, A. M., Jang, H., Dai, R., Picton, P., Mashour, G. A., & Hudetz, A. G. (2026).** *An open fMRI resource for studying human brain function and covert consciousness under anesthesia.* **Scientific Data**, *13*(1), Article 127. https://doi.org/10.1038/s41597-025-06442-2

The original paper is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). All scientific discoveries, experimental design, and neuroimaging analysis methods are credited to the University of Michigan research team.

## License

**This Software (Code)**: MIT License — see [LICENSE](LICENSE)  
**Dataset**: CC0 (Public Domain) — [OpenNeuro ds006623](https://openneuro.org/datasets/ds006623)  
**Original Research Paper**: CC BY-NC-ND 4.0 — see citation above

---

**Documentation**: [docs/](docs/) · **Updated**: February 2026
