===========
Quick Start
===========

This guide will walk you through your first end-to-end workflow: downloading data, training the default XGBoost pipeline, and interpreting results. The repository's default workflow runs the **XGBoost** classifier (full connectivity + PCA + SMOTE + threshold tuning); other model archetypes are not executed by the default pipeline.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
=============

Before starting, make sure you've completed:

1. ✓ Installation from :doc:`installation`
2. ✓ Virtual environment activated
3. ✓ All dependencies installed

**Verify your setup:**

.. code-block:: bash

   # Activate virtual environment (if not already active)
   source venv/bin/activate
   
   # Quick test
   python -c "import torch, nibabel, nilearn; print('✓ Ready to go!')"


Step 1: Download the Dataset
=============================

Overview
--------

We'll download the Michigan Human Anesthesia fMRI Dataset from OpenNeuro.

**Dataset information:**

* **Name**: Michigan Human Anesthesia fMRI Dataset
* **OpenNeuro ID**: ds006623
* **Subjects**: 26 healthy volunteers
* **Format**: BIDS-compliant preprocessed fMRI data
* **License**: CC0 (Public Domain)


Test Dataset Download (Recommended First)
------------------------------------------

Download data for 3 subjects to test everything works:

.. code-block:: bash

   cd covert-awareness-detector
   
   python download_dataset.py \
       --output-dir ./data \
       --subjects sub-02 sub-03 sub-04 \
       --verbose

.. tip::
   You can pause and resume downloads with Ctrl+C - rerun the same command to skip already downloaded files.

**Dataset structure:**

.. code-block:: text

   data/
   ├── dataset_description.json
   ├── participants.tsv
   ├── README
   └── derivatives/
       ├── Participant_Info.csv          # Demographics and metadata
       ├── LOR_ROR_Timing.csv            # Consciousness transition times
       └── xcp_d_without_GSR_bandpass_output/  # Preprocessed fMRI
           ├── sub-02/
           │   ├── func/
           │   │   ├── sub-02_task-rest_run-1_bold.nii.gz
           │   │   ├── sub-02_task-rest_run-1_timeseries.tsv
           │   │   └── sub-02_task-rest_run-1_connectivity.tsv
           │   └── ...
           ├── sub-03/
           └── sub-04/


Understanding the Data
----------------------

**Key files:**

1. **Participant_Info.csv**: Subject demographics
2. **LOR_ROR_Timing.csv**: When each subject lost/recovered consciousness
3. **Connectivity matrices**: Preprocessed functional connectivity

   * **timeseries.tsv**: ROI time-series (time × regions)
   * **connectivity.tsv**: Correlation matrix (regions × regions)


Full Dataset Download (Optional)
---------------------------------

To download all 26 subjects:

.. code-block:: bash

   python download_dataset.py --output-dir ./data --all --verbose


Step 2: Verify Dataset
=======================

After download, verify the data integrity:

.. code-block:: bash

   # Run validation script
   python scripts/validate_dataset.py --data-dir ./data
   
   # Or manually check structure
   python -c "
   from src.data_loader import ConsciousnessDataset
   ds = ConsciousnessDataset('./data')
   print(f'✓ Found {len(ds)} usable samples')
   print(f'✓ Subjects: {ds.get_subject_ids()}')
   print(f'✓ Consciousness states: {ds.get_label_distribution()}')
   "

Understanding Data Samples
---------------------------

Each "sample" is one fMRI scan session:

* **Label**: Conscious (responsive) or Unconscious (unresponsive)
* **Features**: Connectivity matrix or time-series

**Quick data exploration:**

.. code-block:: python

   # Open Python interpreter or Jupyter notebook
   from src.data_loader import ConsciousnessDataset
   import matplotlib.pyplot as plt
   
   # Load dataset
   ds = ConsciousnessDataset('./data')
   
   # Get first sample
   sample = ds[0]
   connectivity = sample['connectivity']
   label = sample['label']
   subject = sample['subject_id']
   
   print(f"Sample from {subject}")
   print(f"Label: {'Conscious' if label == 1 else 'Unconscious'}")
   print(f"Connectivity matrix shape: {connectivity.shape}")
   
   # Visualize connectivity
   plt.figure(figsize=(8, 8))
   plt.imshow(connectivity, cmap='coolwarm', vmin=-0.5, vmax=0.5)
   plt.colorbar(label='Correlation')
   plt.title(f'Functional Connectivity - {subject} (Label: {label})')
   plt.xlabel('Brain Region')
   plt.ylabel('Brain Region')
   plt.tight_layout()
   plt.savefig('connectivity_example.png', dpi=150)
   plt.show()
   
   print(f"✓ Saved visualization to: connectivity_example.png")


Step 3: Train the default model
================================

Train the repository's default XGBoost pipeline (PCA + SMOTE + LOSO-CV). This is the workflow executed by `./run_full_training.sh` and `src/train.py`.

.. code-block:: bash

   # recommended (automated)
   ./run_full_training.sh

   # or run training script directly
   python src/train.py

View training output and validate the model:

.. code-block:: bash

   # Check the live log and results JSON
   less training_*.log
   python src/validate_model.py
   ls -lh results/


Step 4: Train CNN Model
========================

Now let's train a Convolutional Neural Network that treats connectivity matrices as images.

Why CNN for Brain Connectivity?
--------------------------------

* **Spatial patterns**: CNNs learn hierarchical patterns in connectivity
* **Translation invariance**: Similar connectivity patterns in different regions
* **Better generalization**: Deep features often outperform hand-crafted features

Training the CNN
----------------

.. code-block:: bash

   python src/train.py \
       --model cnn \
       --data-dir ./data \
       --output-dir ./results/cnn \
       --epochs 50 \
       --batch-size 16 \
       --learning-rate 0.001 \
       --early-stopping \
       --gpu

Understanding CNN Architecture
-------------------------------

.. code-block:: bash

   python -c "
   from src.models import CNN_Classifier
   import torch
   
   model = CNN_Classifier(input_size=400, num_classes=2)
   print(model)
   "


Step 5: Interpret Results
==========================

Compare Models
--------------

.. code-block:: bash

   python src/compare_models.py \
       --models ./results/baseline/model.pkl ./results/cnn/model_best.pth \
       --names "Random Forest" "CNN" \
       --data-dir ./data \
       --output ./results/comparison.png


Visualize Predictions
----------------------

Let's see what the model is learning:

.. code-block:: python

   # Create visualization script
   from src.models import load_model
   from src.data_loader import ConsciousnessDataset
   from src.visualize import plot_attention_map
   import torch
   
   # Load model and data
   model = load_model('./results/cnn/model_best.pth')
   dataset = ConsciousnessDataset('./data')
   
   # Get a test sample
   sample = dataset[0]
   connectivity = torch.tensor(sample['connectivity']).unsqueeze(0)
   true_label = sample['label']
   
   # Predict
   with torch.no_grad():
       output = model(connectivity)
       predicted_prob = torch.softmax(output, dim=1)[0]
       predicted_label = torch.argmax(predicted_prob).item()
   
   print(f"True label: {'Conscious' if true_label == 1 else 'Unconscious'}")
   print(f"Predicted: {'Conscious' if predicted_label == 1 else 'Unconscious'}")
   print(f"Confidence: {predicted_prob[predicted_label]:.1%}")
   
   # Visualize attention (which regions the model focuses on)
   plot_attention_map(model, connectivity, save_path='attention_map.png')
   print(f"✓ Attention map saved to: attention_map.png")


Analyze Errors
--------------

.. code-block:: bash

   python src/analyze_errors.py \
       --model ./results/cnn/model_best.pth \
       --data-dir ./data \
       --output ./results/error_analysis.html


Per-Subject Performance
------------------------

Check if model generalizes across subjects:

.. code-block:: python

   from src.evaluate import evaluate_per_subject
   import matplotlib.pyplot as plt
   
   # Evaluate each subject separately
   results = evaluate_per_subject(
       model_path='./results/cnn/model_best.pth',
       data_dir='./data'
   )
   
   # Plot results
   subjects = list(results.keys())
   accuracies = [results[s]['accuracy'] for s in subjects]
   
   plt.figure(figsize=(10, 5))
   plt.bar(subjects, accuracies)
   plt.axhline(y=0.75, color='r', linestyle='--', label='Baseline')
   plt.xlabel('Subject')
   plt.ylabel('Accuracy')
   plt.title('Per-Subject Classification Accuracy')
   plt.legend()
   plt.ylim([0, 1])
   plt.tight_layout()
   plt.savefig('per_subject_accuracy.png')
   plt.show()
   
   print(f"Mean accuracy: {sum(accuracies)/len(accuracies):.1%}")
   print(f"Std deviation: {np.std(accuracies):.1%}")


Step 6: Advanced Training (Optional)
=====================================

If you want to push performance further:

Hyperparameter Tuning
---------------------

.. code-block:: bash

   python src/hyperparam_search.py \
       --model cnn \
       --data-dir ./data \
       --output-dir ./results/hyperparameter_search \
       --n-trials 20 \
       --gpu


Data Augmentation
-----------------

.. code-block:: bash

   python src/train.py \
       --model cnn \
       --data-dir ./data \
       --augmentation \
       --aug-methods rotate flip noise \
       --output-dir ./results/cnn_augmented


Ensemble Methods
----------------

.. code-block:: bash

   python src/train_ensemble.py \
       --models baseline cnn gnn \
       --data-dir ./data \
       --output-dir ./results/ensemble \
       --voting soft


Step 7: Next Steps
===================

Congratulations! You've successfully:

✓ Downloaded fMRI consciousness data  
✓ Trained baseline and CNN models  
✓ Evaluated model performance  
✓ Interpreted results

Where to Go from Here
---------------------

**Immediate Next Steps:**

1. **Download full dataset** (all 26 subjects) for better generalization:

   .. code-block:: bash
   
      python download_dataset.py --output-dir ./data --all

2. **Train advanced models**:

   .. code-block:: bash
   
      python src/train.py --model gnn --data-dir ./data
      python src/train.py --model rnn --data-dir ./data
      python src/train.py --model transformer --data-dir ./data

3. **Multi-class classification**:

   .. code-block:: bash
   
      python src/train.py \
          --model cnn \
          --task multiclass \
          --classes awake mild moderate deep recovery \
          --data-dir ./data

4. **Covert consciousness detection**:

   .. code-block:: bash
   
      python src/detect_covert.py \
          --model ./results/cnn/model_best.pth \
          --data-dir ./data \
          --output ./results/covert_detection


**Explore Documentation:**

* :doc:`architecture` - Deep dive into model architectures
* :doc:`dataset` - Understand the fMRI data in detail
* :doc:`evaluation` - Advanced evaluation techniques
* :doc:`api` - Complete API reference
* :doc:`contributing` - Help improve the project


Working with Your Own Data
---------------------------

To use this framework with different datasets:

1. **Convert to BIDS format**: Use ``dcm2niix`` for DICOM → NIfTI
2. **Preprocess with fMRIPrep**: Standard preprocessing pipeline
3. **Extract connectivity**: Use Nilearn or custom ROI masks
4. **Create metadata**: Prepare CSV with labels and timing
5. **Modify data loader**: Adapt ``src/data_loader.py`` for your format

See :doc:`custom_data` for detailed instructions.


Deployment
----------

To deploy models in production:

.. code-block:: bash

   # Export model for deployment
   python src/deploy_model.py \
       --model ./results/cnn/model_best.pth \
       --output ./deployment/consciousness_detector.onnx \
       --format onnx

See :doc:`deployment` for:

* ONNX export for cross-platform inference
* REST API for web services
* Docker containers for reproducible deployment
* Real-time inference optimization


Getting Help
============

If you run into issues:

**Documentation:**

* :doc:`faq` - Frequently asked questions
* :doc:`troubleshooting` - Common problems and solutions
* :doc:`api` - API reference

**Community:**

* **GitHub Issues**: https://github.com/yourusername/consciousness_detector/issues
* **Discussions**: https://github.com/yourusername/consciousness_detector/discussions
* **Email**: maintainer@project.org

**Reporting Bugs:**

When reporting issues, include:

1. Python version and OS
2. Output of ``pip list``
3. Complete error message
4. Minimal code to reproduce the problem


Summary
=======

You've learned:

* How to download and prepare fMRI consciousness data
* Training baseline and deep learning models
* Evaluating classification performance
* Interpreting model predictions
* Next steps for advanced analysis
