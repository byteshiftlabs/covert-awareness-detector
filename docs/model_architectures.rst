====================
Model Architectures
====================

This document explains how the Covert Awareness Detector classifies brain states
(conscious vs unconscious) from fMRI connectivity data.

.. contents:: Table of Contents
   :local:
   :depth: 2


How It Works
============

The detector takes a **connectivity matrix** — a table of numbers showing how
strongly each pair of brain regions communicates — and feeds it into a
classifier that learns to tell conscious from unconscious states.

The process has three steps:

1. **Feature extraction**: Convert the connectivity matrix into a set of
   numbers (features) that summarise brain organisation. These features
   capture how *integrated* (globally connected) and how *segregated*
   (locally clustered) the brain is.

2. **Training**: Show the classifier many labelled examples
   ("this brain scan came from a conscious person, this one from an
   unconscious person") so it learns the patterns that distinguish the two.

3. **Prediction**: Given a new, unseen brain scan, the classifier outputs
   its best guess — conscious or unconscious — along with a confidence score.


Available Models
================

This codebase implements and runs a single default classifier: **XGBoost**.
The pipeline applies PCA to the full connectivity matrix, uses SMOTE to
address class imbalance, and performs threshold tuning after LOSO-CV.

**Note:** references to Logistic Regression, Random Forest, and SVM have
been removed from the documentation because those classifiers are not
executed by the default training pipeline in this repository.


Evaluation Method
=================

Leave-One-Subject-Out Cross-Validation
---------------------------------------

Because we want the model to work on **new patients it has never seen
before**, we evaluate it with a strict procedure called
Leave-One-Subject-Out (LOSO) cross-validation:

1. Pick one subject and set their data aside as the test set.
2. Train the model on all remaining subjects.
3. Test the model on the held-out subject.
4. Repeat for every subject.

This ensures the model is always tested on a person whose data it never
saw during training, which is the most realistic evaluation for a clinical
tool.


Class Balancing
===============

Conscious and unconscious samples may not appear in equal numbers. To
prevent class imbalance from biasing the classifier, the pipeline uses
SMOTE oversampling and class weighting with XGBoost so the model learns
from underrepresented conscious examples.


How to Run
==========

.. code-block:: bash

   # Train and evaluate XGBoost with LOSO cross-validation
   python src/train.py

This runs the default XGBoost pipeline (PCA + SMOTE + threshold tuning)
using Leave-One-Subject-Out CV and prints per-subject results.


Next Steps
==========

- See :doc:`feature_extraction` for details on how brain features are computed.
- See :doc:`dataset` for information about the fMRI data.
- See :doc:`installation` and `src/train.py` for instructions to run the default pipeline.
