# Covert Awareness Detector

Machine learning models for detecting covert consciousness from fMRI data during anesthesia, building on research by Huang et al. (University of Michigan).

## Overview

Detects hidden awareness in behaviorally unresponsive patients using the Michigan Human Anesthesia fMRI Dataset. Implements machine learning classifiers to distinguish consciousness states during propofol sedation.

**Dataset**: [OpenNeuro ds006623](https://openneuro.org/datasets/ds006623) - 26 subjects, mental imagery tasks, graded sedation levels

## Acknowledgments

**Original Research**: Huang, Hudetz, Mashour et al. (2018-2021) - Neuroscience foundations and dataset  
**MATLAB Code**: [Jang et al.](https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation)  
**This Project**: Independent Python ML implementation by cmelnulabs

## Quick Start

```bash
# Clone and install
git clone https://github.com/cmelnulabs/covert-awareness-detector.git
cd covert-awareness-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset
python download_dataset.py --output-dir ../datasets/openneuro/ds006623

# Train model
python src/train_improved.py
```

## Models

- **Logistic Regression**: L2-regularized with balanced class weights
- **Random Forest**: 100 trees with balanced sampling
- **SVM**: RBF kernel for non-linear classification
- **Evaluation**: Leave-One-Subject-Out cross-validation (26 folds)

## Citation

**Dataset & Original Research:**
```bibtex
@article{huang2018covert,
  title={Brain imaging reveals covert consciousness during behavioral unresponsiveness},
  author={Huang, Zirui and others},
  journal={Scientific Reports},
  volume={8}, pages={13195}, year={2018},
  doi={10.1038/s41598-018-31436-z}
}
```

**Full citations**: See [original publications](https://doi.org/10.1038/s41598-018-31436-z)

## License

MIT License | Dataset: CC0 (Public Domain)

---

**Documentation**: See [docs/](docs/) for detailed guides  
**Contact**: cmelnulabs | **Updated**: February 2026
