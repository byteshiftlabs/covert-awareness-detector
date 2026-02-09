# Covert Awareness Detector

A Python machine learning implementation for detecting covert consciousness and predicting consciousness states from fMRI data during anesthesia, building on the neuroscience foundations established by Huang et al. (2018, 2021).

## Acknowledgments & Original Research

**This project is inspired by and builds upon the groundbreaking work of:**

**Huang, Hudetz, Mashour et al.** at the University of Michigan, whose pioneering research identified covert consciousness signatures in fMRI data during propofol anesthesia. Their original studies (Huang et al. 2018 _Scientific Reports_, 2021 _Cell Reports_, 2021 _NeuroImage_) established the neuroscientific foundation for detecting hidden awareness in behaviorally unresponsive patients.

**Original MATLAB Analysis Code:**  
- Repository: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation
- Authors: Hyunwoo Jang, Zirui Huang, and collaborators
- Description: MATLAB code for functional connectivity and network balance analysis

**This Project's Contribution:**  
This is an **independent Python deep learning implementation** that:
- Reimplements the analysis pipeline using modern ML frameworks (PyTorch/TensorFlow)
- Extends the original findings with neural network architectures (CNNs, GNNs)
- Aims to develop deployable models for clinical consciousness detection
- Builds on their scientific discoveries while exploring new ML approaches

**All credit for the original neuroscience discoveries, experimental design, and dataset creation belongs to the University of Michigan team.** This project focuses on the machine learning engineering and model development aspects.

---

## Overview

This project implements deep learning models to detect hidden consciousness in patients who appear behaviorally unresponsive but show neural signatures of awareness. Using the Michigan Human Anesthesia fMRI Dataset (published by Huang et al.), we train classifiers to distinguish between different levels of consciousness during propofol sedation.

**Key Innovation**: Detecting "covert consciousness" - patients who can follow mental imagery commands despite showing no behavioral response - a phenomenon first characterized by Huang et al. (2018).

## Dataset

**Michigan Human Anesthesia fMRI Dataset (OpenNeuro ds006623)**

- **26 healthy volunteers** undergoing graded propofol sedation
- **Mental imagery tasks**: Tennis, spatial navigation, hand squeeze
- **Consciousness states**: Awake → Mild → Moderate → Deep sedation → Recovery
- **Data types**: fMRI BOLD, functional connectivity, behavioral responses
- **Format**: BIDS-compliant, preprocessed with fMRIPrep and XCP-D

**Dataset DOI**: [10.18112/openneuro.ds006623.v1.0.0](https://doi.org/10.18112/openneuro.ds006623.v1.0.0)  
**OpenNeuro URL**: https://openneuro.org/datasets/ds006623

## Scientific Background

### Key Publications Using This Dataset

1. **Huang et al. (2018) - Scientific Reports**
   - *"Brain imaging reveals covert consciousness during behavioral unresponsiveness induced by propofol"*
   - **DOI**: [10.1038/s41598-018-31436-z](https://doi.org/10.1038/s41598-018-31436-z)
   - **PubMed**: [30181567](https://pubmed.ncbi.nlm.nih.gov/30181567/)
   - **Key Finding**: 5 out of 26 subjects showed brain activation during mental imagery tasks despite behavioral unresponsiveness
   - **Methods**: Mental imagery paradigm (tennis, navigation) + propofol sedation
   - **Results**: Covert consciousness detected via anterior insula activation patterns

2. **Huang et al. (2021) - Cell Reports**
   - *"Anterior insula regulates brain network transitions that gate conscious access"*
   - **DOI**: [10.1016/j.celrep.2021.109081](https://doi.org/10.1016/j.celrep.2021.109081)
   - **PubMed**: [33951427](https://pubmed.ncbi.nlm.nih.gov/33951427/)
   - **Key Finding**: Anterior insula acts as a "gating" mechanism for conscious access
   - **Methods**: Dynamic functional connectivity analysis during sedation transitions
   - **Results**: Insula connectivity predicts whether mental imagery reaches consciousness

3. **Huang et al. (2021) - NeuroImage**
   - *"Asymmetric neural dynamics characterize loss and recovery of consciousness"*
   - **DOI**: [10.1016/j.neuroimage.2021.118042](https://doi.org/10.1016/j.neuroimage.2021.118042)
   - **PubMed**: [33848623](https://pubmed.ncbi.nlm.nih.gov/33848623/)
   - **Key Finding**: Loss and recovery of consciousness follow different neural trajectories (neural hysteresis)
   - **Methods**: Temporal dynamics of brain networks across sedation and emergence
   - **Results**: Recovery requires higher brain network integration than loss

## Research Goals

**Building on Huang et al.'s neuroscience findings, this project aims to:**

### Primary Objectives
1. **Binary Classification**: Conscious (responsive) vs Unconscious (unresponsive)
   - Cross-subject generalization using deep learning
   - Extends the original connectivity-based findings
   
2. **Multi-Class Classification**: Awake / Mild / Moderate / Deep / Recovery
   - Automated classification of sedation depth
   - Beyond manual connectivity analysis
   
3. **Covert Consciousness Detection**: Identify hidden awareness during unresponsiveness
   - Replicate findings from Huang et al. (2018)
   - Develop automated detection methods for clinical deployment

### Secondary Objectives
4. **Sedation Depth Regression**: Predict propofol effect-site concentration
5. **Emergence Prediction**: Forecast time to consciousness recovery
6. **Biomarker Discovery**: Identify key brain regions/networks for consciousness

## Architecture Overview

**Python ML Implementation Structure** (building on Huang et al.'s analysis framework):

```
covert-awareness-detector/
├── data/
│   ├── loaders.py          # DataLoader for fMRI connectivity matrices
│   ├── preprocessing.py    # Feature extraction, normalization
│   └── labels.py           # Consciousness state labeling from timing data
├── models/
│   ├── baseline.py         # Logistic regression, SVM, Random Forest
│   ├── cnn.py              # 2D-CNN on connectivity matrices
│   ├── gnn.py              # Graph Neural Network on brain graphs
│   └── ensemble.py         # Multi-model ensemble
├── features/
│   ├── connectivity.py     # Functional connectivity extraction
│   ├── graph_metrics.py    # Graph theory features
│   └── temporal.py         # Dynamic connectivity features
├── training/
│   ├── trainer.py          # Training loop with cross-validation
│   ├── metrics.py          # Evaluation metrics (accuracy, AUC, F1)
│   └── visualization.py    # Training curves, confusion matrices
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_deep_learning.ipynb
├── configs/
│   ├── baseline_config.yaml
│   └── deep_config.yaml
├── tests/
│   └── test_*.py
├── README.md
├── ROADMAP.md
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/covert-awareness-detector.git
cd covert-awareness-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

The project uses the **Michigan Human Anesthesia fMRI Dataset** (OpenNeuro ds006623). We provide three methods to download the required files:

#### Method 1: Python Script (Recommended)

The Python script provides progress bars, error handling, and resume capability:

```bash
# Download all essential files (metadata + connectivity data)
python download_dataset.py --output-dir ../datasets/openneuro/ds006623

# Download only metadata (CSV files, no connectivity data)
python download_dataset.py --metadata-only

# Download only specific subjects
python download_dataset.py --subjects sub-02 sub-03 sub-04

# List files without downloading
python download_dataset.py --list-files
```

**What it downloads:**
- `derivatives/Participant_Info.csv` - Subject demographics and metadata
- `derivatives/LOR_ROR_Timing.csv` - Consciousness transition timestamps (LOR/ROR)
- `derivatives/xcp_d_without_GSR_bandpass_output/` - Preprocessed fMRI connectivity matrices
  - Gordon atlas (333 ROIs)
  - Schaefer atlas (417 ROIs)
  - Glasser atlas (360 ROIs)

#### Method 2: Bash Script (Simple Alternative)

For a simpler command-line experience:

```bash
# Run the bash script
cd scripts/
./download_essentials.sh

# Or specify custom output directory
./download_essentials.sh /path/to/dataset
```

The bash script uses `curl` or `wget` and provides colored output with progress tracking.

#### Method 3: DataLad (Full Dataset)

If you need the complete dataset including raw fMRI files:

```bash
# Install DataLad first
pip install datalad

# Clone the entire dataset structure
datalad clone https://github.com/OpenNeuroDatasets/ds006623.git ../datasets/openneuro/ds006623
cd ../datasets/openneuro/ds006623

# Download specific files
datalad get derivatives/Participant_Info.csv
datalad get derivatives/LOR_ROR_Timing.csv
datalad get derivatives/xcp_d_without_GSR_bandpass_output/sub-*/
```

**Note:** DataLad downloads take significantly longer and require ~100GB for the full dataset. Our Python/Bash scripts download only the essential files (~2-5GB).

#### Verify Download

After downloading, verify the dataset structure:

```bash
ls ../datasets/openneuro/ds006623/derivatives/
# Should show: Participant_Info.csv, LOR_ROR_Timing.csv, xcp_d_without_GSR_bandpass_output/

ls ../datasets/openneuro/ds006623/derivatives/xcp_d_without_GSR_bandpass_output/
# Should show: sub-02, sub-03, ..., sub-30 (26 subjects total)
```

### Run Baseline Model

```bash
# Train logistic regression classifier
python -m training.train_baseline \
    --data_path ../datasets/openneuro/ds006623 \
    --model logistic_regression \
    --cv leave_one_subject_out
```

### Train Deep Learning Model

```bash
# Train CNN on connectivity matrices
python -m training.train_deep \
    --model cnn \
    --epochs 100 \
    --batch_size 16 \
    --cv_folds 26
```

## Model Architectures

### 1. Baseline Models
- **Logistic Regression**: L2-regularized, flattened connectivity features
- **Random Forest**: 100 trees, max_depth=10
- **SVM**: RBF kernel, grid search for C and gamma
d
### 2. Convolutional Neural Network (CNN)
```
Input: Connectivity Matrix (N_roi × N_roi)
├── Conv2D(32, 5×5) + ReLU + MaxPool
├── Conv2D(64, 3×3) + ReLU + MaxPool
├── Conv2D(128, 3×3) + ReLU + AdaptiveAvgPool
├── Flatten
├── Dense(256) + ReLU + Dropout(0.5)
├── Dense(128) + ReLU + Dropout(0.3)
└── Dense(n_classes) + Softmax
```

### 3. Graph Neural Network (GNN)
```
Input: Brain Graph (Nodes=ROIs, Edges=Correlations)
├── GraphConv(64) + ReLU
├── GraphConv(128) + ReLU
├── GraphConv(256) + ReLU
├── GlobalAttentionPooling
├── Dense(128) + ReLU
└── Dense(n_classes) + Softmax
```

## Features

**Feature extraction approaches inspired by Huang et al.'s analysis methods:**

### Connectivity Features
- **Pearson correlation**: Time-series correlation between ROI pairs
- **Partial correlation**: Removes indirect connections
- **Covariance matrices**: Raw covariance structure

### Graph Theory Metrics
- **Global**: Modularity, global efficiency, characteristic path length
- **Nodal**: Degree centrality, betweenness, clustering coefficient
- **Network-level**: Rich club organization, small-worldness

### Temporal Features
- **Dynamic connectivity**: Sliding window correlation
- **State transitions**: HMM or clustering on connectivity patterns
- **Frequency bands**: Alpha, beta, gamma band connectivity

## Evaluation

### Cross-Validation Strategy
- **Leave-One-Subject-Out (LOSO)**: Train on 25 subjects, test on 1 (repeated 26 times)
- Ensures generalization to new individuals
- Prevents subject-specific overfitting

### Metrics
- **Accuracy**: Overall correct classification rate
- **F1-Score**: Balanced precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Per-class performance breakdown

Detailed model performance and results will be documented in the Sphinx documentation.


## Clinical Applications

**Note**: These applications were identified and validated by Huang et al. in their original research. This ML implementation aims to make these findings more accessible for clinical deployment.

1. **Intraoperative Monitoring**: Detect awareness during anesthesia
2. **Disorders of Consciousness**: Diagnose covert consciousness in coma/vegetative state
3. **Personalized Anesthesia**: Optimize drug dosing based on brain state
4. **Consciousness Research**: Understand neural correlates of awareness

## Limitations

- **Small sample size**: 26 subjects limits generalizability
- **Healthy volunteers**: May not generalize to clinical populations
- **Controlled setting**: Lab conditions differ from real clinical scenarios
- **Single anesthetic**: Propofol-specific mechanisms may not extend to other drugs

## Future Work

- [ ] Transfer learning to clinical DoC datasets (vegetative state, MCS)
- [ ] Real-time classification for online brain-computer interfaces
- [ ] Multimodal integration (fMRI + EEG + behavioral)
- [ ] Explainable AI: Visualize important brain regions for predictions
- [ ] Adversarial robustness testing

## Team & Contributing

**Project Nature**: Independent ML engineering project building on University of Michigan neuroscience research

**ML Implementation Lead**: cmelnulabs  
**Started**: February 2026  
**Scope**: Python deep learning reimplementation and extension of original MATLAB analysis

**Original Research Team**: Huang, Hudetz, Mashour et al. (University of Michigan)  
**Original Analysis Code**: Hyunwoo Jang et al.

Contributions welcome! This is an open-source ML project. See issues for current tasks.

**Note**: For questions about the original neuroscience findings, dataset, or experimental methods, please refer to the published papers or contact the University of Michigan team. This repository focuses on the ML implementation.

## Citation

If you use this code or dataset, please cite:

**1. The Original Dataset and Research Papers:**

```bibtex
@article{huang2026open,
  title={An open fMRI resource for studying human brain function and covert consciousness under anesthesia},
  author={Huang, Zirui and Tarnal, Vijay and Fotiadis, Panagiotis and Vlisides, Phillip E and 
          Janke, Ellen L and Puglia, Michael and McKinney, Amy M and Jang, Hyunwoo and 
          Dai, Rui and Picton, Paul and Mashour, George A and Hudetz, Anthony G},
  journal={Scientific Data},
  volume={13},
  number={1},
  pages={127},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-025-06442-2}
}

@article{huang2018covert,
  title={Brain imaging reveals covert consciousness during behavioral unresponsiveness induced by propofol},
  author={Huang, Zirui and Tarnal, Vijay and Hesse, Erik and Del Pin, Santiago Canales and 
          Pryor, Kendall J and Corbett, Robert and Snider, Stuart B and Janke, Ellen L and 
          Kelz, Max B and Mashour, George A and Hudetz, Anthony G},
  journal={Scientific Reports},
  volume={8},
  pages={13195},
  year={2018},
  doi={10.1038/s41598-018-31436-z}
}

@article{huang2021anterior,
  title={Anterior insula regulates brain network transitions that gate conscious access},
  author={Huang, Zirui and Vlisides, Phillip E and Tarnal, Vijay and Janke, Ellen L and 
          McKinney, Amy M and Picton, Paul and Mashour, George A and Hudetz, Anthony G},
  journal={Cell Reports},
  volume={35},
  number={6},
  pages={109081},
  year={2021},
  doi={10.1016/j.celrep.2021.109081}
}

@article{huang2021asymmetric,
  title={Asymmetric neural dynamics characterize loss and recovery of consciousness},
  author={Huang, Zirui and Tarnal, Vijay and Vlisides, Phillip E and Janke, Ellen L and 
          McKinney, Amy M and Picton, Paul and Mashour, George A and Hudetz, Anthony G},
  journal={NeuroImage},
  volume={236},
  pages={118042},
  year={2021},
  doi={10.1016/j.neuroimage.2021.118042}
}
```

**2. The Original MATLAB Analysis Code:**
- Repository: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation
- Authors: Hyunwoo Jang, Zirui Huang, et al.

**3. This Python ML Implementation (if applicable):**
- Repository: https://github.com/cmelnulabs/covert-awareness-detector
- Author: cmelnulabs
- Description: Python deep learning implementation for covert consciousness detection

## License

Code: MIT License  
Dataset: CC0 (Public Domain)

## Acknowledgments

**Primary Credit:**
- **Huang, Hudetz, Mashour, and collaborators** at the University of Michigan for the original neuroscience research, experimental design, data collection, and scientific discoveries
- **Hyunwoo Jang** and the original analysis team for the MATLAB implementation
- University of Michigan Anesthesiology Department for dataset creation and curation
- NIH grants R01GM103894, R01GM111293 for funding the original research
- OpenNeuro platform for data hosting and open science infrastructure

**This Implementation:**
- Python ML reimplementation: cmelnulabs (2026)
- Independent project for exploring modern deep learning approaches to consciousness detection

---

**Contact**: cmelnulabs  
**Project**: Independent ML research (building on University of Michigan foundations)  
**Repository**: covert-awareness-detector  
**Last Updated**: February 9, 2026
