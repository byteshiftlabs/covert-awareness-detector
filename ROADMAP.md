# Consciousness State Detector - Development Roadmap

## Project Vision

Build a production-ready machine learning system that detects covert consciousness and predicts consciousness states from fMRI brain imaging data, with potential clinical applications in anesthesia monitoring and disorders of consciousness diagnosis.

---

## Phase 1: Foundation (Weeks 1-2) ✅ CURRENT PHASE

### Objectives
- Set up development environment
- Download and validate dataset
- Implement data loading pipeline
- Establish baseline performance

### Tasks

#### 1.1 Project Setup
- [x] Create project directory structure
- [ ] Initialize git repository
- [ ] Set up virtual environment
- [ ] Create requirements.txt with dependencies:
  - Core: numpy, pandas, scipy
  - Neuroimaging: nibabel, nilearn, nipype
  - ML: scikit-learn, torch, tensorflow
  - Visualization: matplotlib, seaborn, plotly
  - Utils: tqdm, pyyaml, pytest

#### 1.2 Dataset Acquisition
- [x] Download dataset metadata from OpenNeuro
- [ ] Get participant info and timing data (LOR/ROR)
- [ ] Download 3 subjects as test set (sub-02, sub-03, sub-04)
- [ ] Validate BIDS compliance
- [ ] Document data structure

#### 1.3 Data Loading & Preprocessing
- [ ] Implement `ConnectivityDataLoader` class
  - Read preprocessed connectivity matrices from xcp_d
  - Handle multiple denoising strategies (GSR/non-GSR)
  - Extract ROI time-series
- [ ] Create `ConsciousnessLabeler` class
  - Parse LOR_ROR_Timing.csv
  - Map fMRI runs to consciousness states
  - Generate ground truth labels
- [ ] Data validation pipeline
  - Check for missing files
  - Verify temporal alignment
  - Quality control metrics

#### 1.4 Baseline Model
- [ ] Implement logistic regression classifier
  - Flatten connectivity matrices as features
  - Binary classification: responsive vs unresponsive
  - Leave-one-subject-out cross-validation
- [ ] Evaluation framework
  - Accuracy, F1-score, ROC-AUC
  - Confusion matrices
  - Statistical significance tests
- [ ] **Target**: 75-80% accuracy baseline

### Deliverables
- [ ] Working data pipeline
- [ ] Baseline model with documented performance
- [ ] Jupyter notebook with exploratory data analysis
- [ ] Unit tests for data loading

---

## Phase 2: Feature Engineering (Weeks 3-4)

### Objectives
- Extract meaningful features from brain connectivity
- Implement graph theory metrics
- Optimize feature selection

### Tasks

#### 2.1 Connectivity Features
- [ ] Pearson correlation matrices
- [ ] Partial correlation (remove confounds)
- [ ] Fisher Z-transformation for normalization
- [ ] Multiple parcellation schemes (Schaefer, AAL, Power)

#### 2.2 Graph Theory Metrics
- [ ] Global metrics:
  - Modularity (Q)
  - Global efficiency
  - Characteristic path length
  - Small-worldness index
- [ ] Nodal metrics:
  - Degree centrality
  - Betweenness centrality
  - Clustering coefficient
  - Participation coefficient
- [ ] Network metrics:
  - Rich club coefficient
  - Assortativity
  - Core-periphery structure

#### 2.3 Temporal Features
- [ ] Dynamic functional connectivity (sliding window)
- [ ] State transitions using HMM
- [ ] Frequency-specific connectivity (alpha, beta, gamma)
- [ ] Time-lagged correlations

#### 2.4 Dimensionality Reduction
- [ ] PCA on connectivity matrices
- [ ] ICA for network identification
- [ ] t-SNE/UMAP for visualization
- [ ] Feature selection (LASSO, mutual information)

### Deliverables
- [ ] Feature extraction module
- [ ] Feature importance analysis
- [ ] Improved baseline: 80-85% accuracy

---

## Phase 3: Deep Learning Models (Weeks 5-7)

### Objectives
- Implement state-of-the-art deep learning architectures
- Achieve >85% classification accuracy
- Handle imbalanced classes and subject variability

### Tasks

#### 3.1 Convolutional Neural Network (2D-CNN)
- [ ] Architecture design:
  - Input: N_roi × N_roi connectivity matrix
  - 3-4 conv layers with batch normalization
  - Spatial pooling and dropout
  - Dense layers for classification
- [ ] Data augmentation:
  - Gaussian noise injection
  - Edge dropout (random connectivity removal)
  - Time-series jittering
- [ ] Training strategy:
  - Adam optimizer, learning rate scheduling
  - Early stopping on validation loss
  - Gradient clipping

#### 3.2 Graph Neural Network (GNN)
- [ ] Graph construction from connectivity matrices
- [ ] Architecture:
  - Graph Convolutional Layers (GCN)
  - Graph Attention Networks (GAT)
  - Global pooling (attention or mean)
- [ ] Edge features and weighted graphs
- [ ] Compare GCN vs GAT vs GraphSAGE

#### 3.3 Recurrent Models (Optional)
- [ ] LSTM on dynamic connectivity sequences
- [ ] Bidirectional RNN for temporal context
- [ ] Attention mechanism for important time points

#### 3.4 Ensemble Methods
- [ ] Combine CNN + GNN predictions
- [ ] Stacking with meta-learner
- [ ] Bayesian model averaging

### Deliverables
- [ ] Trained deep learning models
- [ ] Model comparison benchmarks
- [ ] **Target**: 85-92% accuracy
- [ ] Saved model checkpoints

---

## Phase 4: Advanced Analysis (Weeks 8-9)

### Objectives
- Detect covert consciousness specifically
- Multi-class classification (5 sedation levels)
- Interpretability and explainability

### Tasks

#### 4.1 Covert Consciousness Detection
- [ ] Identify subjects with neural activity during unresponsiveness
- [ ] Train specialized classifier for "hidden awareness"
- [ ] Compare mental imagery vs resting state signatures
- [ ] **Target**: Replicate 19% detection rate from paper

#### 4.2 Multi-Class Sedation Prediction
- [ ] 5-class problem: Awake / Mild / Moderate / Deep / Recovery
- [ ] Ordinal regression (sedation as continuous scale)
- [ ] Temporal smoothing of predictions

#### 4.3 Regression Tasks
- [ ] Predict propofol effect-site concentration
- [ ] Time-to-recovery estimation
- [ ] Confidence intervals and uncertainty quantification

#### 4.4 Explainable AI
- [ ] Gradient-based saliency maps (GradCAM)
- [ ] Attention weight visualization
- [ ] SHAP values for feature importance
- [ ] Identify critical brain regions for consciousness

### Deliverables
- [ ] Multi-task model
- [ ] Covert consciousness detector
- [ ] Interpretability visualizations

---

## Phase 5: Validation & Clinical Translation (Weeks 10-12)

### Objectives
- Rigorous validation with clinical standards
- Test generalization to external datasets
- Prepare for clinical deployment

### Tasks

#### 5.1 Robustness Testing
- [ ] Cross-site validation (if multi-site data available)
- [ ] Sensitivity analysis to preprocessing choices
- [ ] Adversarial robustness tests
- [ ] Bootstrap confidence intervals

#### 5.2 Transfer Learning
- [ ] Fine-tune on external DoC datasets:
  - Vegetative state / MCS patients
  - Traumatic brain injury cohorts
  - Stroke patients with consciousness disorders
- [ ] Domain adaptation techniques
- [ ] Zero-shot transfer evaluation

#### 5.3 Real-Time Inference
- [ ] Optimize model for low-latency prediction
- [ ] Quantization and pruning for efficiency
- [ ] REST API for model serving
- [ ] Docker containerization

#### 5.4 Clinical Validation Plan
- [ ] Comparison with existing clinical tools (CRS-R, GCS)
- [ ] Sensitivity/specificity in clinical setting
- [ ] False positive/negative analysis
- [ ] Ethical considerations document

### Deliverables
- [ ] Validated production model
- [ ] Clinical validation report
- [ ] Deployment-ready inference system

---

## Phase 6: Publication & Dissemination (Weeks 13-16)

### Objectives
- Publish research findings
- Release open-source code
- Contribute to neuroscience community

### Tasks

#### 6.1 Research Paper
- [ ] Write methods section (reproducible pipeline)
- [ ] Results with statistical tests
- [ ] Discussion: compare to Huang et al. findings
- [ ] Submit to *NeuroImage* or *Scientific Reports*

#### 6.2 Code Release
- [ ] Clean and document all code
- [ ] Create tutorial notebooks
- [ ] Write API documentation (Sphinx)
- [ ] Release on GitHub with permissive license

#### 6.3 Community Engagement
- [ ] Blog post on methodology
- [ ] Present at neuroimaging conference
- [ ] Collaborate with clinical researchers
- [ ] Dataset contribution (if new data collected)

### Deliverables
- [ ] Peer-reviewed publication (submitted)
- [ ] Open-source repository with 100+ GitHub stars
- [ ] Tutorial documentation

---

## Technical Milestones

| Milestone | Metric | Target | Status |
|-----------|--------|--------|--------|
| Baseline Classifier | Accuracy | 75-80% | 🔄 In Progress |
| Feature-Engineered Model | Accuracy | 80-85% | ⏳ Pending |
| Deep Learning Model | Accuracy | 85-92% | ⏳ Pending |
| Covert Consciousness | Detection Rate | 19% (5/26) | ⏳ Pending |
| Multi-Class | Accuracy | >75% | ⏳ Pending |
| Cross-Subject LOSO | AUC | >0.90 | ⏳ Pending |
| Real-Time Inference | Latency | <1 second | ⏳ Pending |

---

## Risk Mitigation

### Technical Risks

**Risk**: Small sample size (26 subjects) limits model generalization  
**Mitigation**: 
- Heavy data augmentation
- Leave-one-subject-out CV
- Transfer learning from larger datasets
- Focus on interpretability over accuracy

**Risk**: Class imbalance (fewer covert consciousness cases)  
**Mitigation**:
- SMOTE oversampling
- Class weights in loss function
- Focal loss for hard examples

**Risk**: Preprocessing variability affects results  
**Mitigation**:
- Use standardized XCP-D outputs
- Test multiple denoising strategies
- Document all preprocessing choices

### Resource Risks

**Risk**: Large dataset requires significant storage/compute  
**Mitigation**:
- Download data incrementally
- Use cloud compute for training (Google Colab, AWS)
- Implement data streaming loaders

**Risk**: Long training times for deep models  
**Mitigation**:
- Start with small models
- Use learning rate warmup and scheduling
- Mixed precision training (FP16)

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Working baseline classifier (>75% accuracy)
- [ ] Leave-one-subject-out cross-validation
- [ ] Documented code with unit tests
- [ ] Reproducible results matching paper benchmarks

### Full Success
- [ ] Deep learning model >85% accuracy
- [ ] Covert consciousness detection replicating 19% rate
- [ ] Multi-class classifier >75% accuracy
- [ ] Published research paper
- [ ] Open-source release with community adoption

### Stretch Goals
- [ ] Real-time clinical deployment
- [ ] Transfer to external DoC datasets
- [ ] Novel biomarker discovery
- [ ] Patent for clinical application

---

## Timeline Summary

```
Week 1-2:  ████████░░ Foundation & Baseline (CURRENT)
Week 3-4:  ░░░░░░░░░░ Feature Engineering
Week 5-7:  ░░░░░░░░░░ Deep Learning Development
Week 8-9:  ░░░░░░░░░░ Advanced Analysis
Week 10-12:░░░░░░░░░░ Validation & Clinical
Week 13-16:░░░░░░░░░░ Publication & Release
```

**Estimated Total Duration**: 16 weeks (4 months)  
**Start Date**: February 8, 2026  
**Target Completion**: June 2026

---

## Resource Requirements

### Computational
- **CPU**: 8+ cores for preprocessing
- **RAM**: 32 GB minimum (64 GB recommended)
- **GPU**: NVIDIA RTX 3060+ or cloud GPU (Colab Pro, AWS)
- **Storage**: 100 GB for dataset + models

### Software
- Python 3.9+
- PyTorch 2.0+
- Nilearn, Nibabel
- DataLad for dataset management

### Knowledge
- fMRI analysis fundamentals
- Deep learning (CNNs, GNNs)
- Graph theory
- Statistical validation methods

---

## Maintenance Plan

### Continuous Integration
- [ ] GitHub Actions for automated testing
- [ ] Pre-commit hooks for code quality
- [ ] Dependency version pinning

### Documentation
- [ ] Keep README updated
- [ ] Document all hyperparameters
- [ ] Maintain CHANGELOG.md

### Community
- [ ] Respond to GitHub issues weekly
- [ ] Accept pull requests
- [ ] Quarterly blog updates on progress

---

**Next Actions**: See `TODO` list for immediate implementation tasks.

**Last Updated**: February 8, 2026
