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

The project implements three classifiers. All three are well-established
machine-learning algorithms provided by the scikit-learn library.

Logistic Regression
-------------------

Draws a straight line (in high-dimensional space) that separates conscious
from unconscious samples. Each feature gets a weight; the model sums up
the weighted features and applies a threshold.

**Interpretation**: The weights tell you which brain connections matter most
for the classification. A large positive weight means that connection is
associated with consciousness; a large negative weight means the opposite.


Random Forest
-------------

Builds many decision trees, each trained on a different random subset of the
data and features. Each tree makes its own prediction, and the final answer
is the majority vote.

**Interpretation**: The model reports *feature importance* — how much each
brain connection contributed to the classification across all trees.


Support Vector Machine (SVM)
-----------------------------

Finds the boundary between conscious and unconscious samples that has the
widest possible margin. It uses a mathematical trick (the RBF kernel) to
draw curved boundaries when a straight line is not enough.

**Interpretation**: The model identifies *support vectors* — the samples
closest to the decision boundary that are most informative for classification.


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
prevent the model from being biased towards the more frequent class, all
three classifiers use **balanced class weights**. This means the model
pays more attention to the less frequent class during training.


How to Run
==========

.. code-block:: bash

   # Compare all three models with LOSO cross-validation
   python src/train.py

This trains Logistic Regression, Random Forest, and SVM, prints
per-subject results, and shows a comparison table at the end.

To train a single model type:

.. code-block:: python

   from src.models import ConsciousnessClassifier

   clf = ConsciousnessClassifier(model_type='random_forest')
   clf.train(X_train, y_train)
   metrics = clf.evaluate(X_test, y_test)


Next Steps
==========

- See :doc:`feature_extraction` for details on how brain features are computed.
- See :doc:`dataset` for information about the fMRI data.
- See :doc:`quickstart` for a step-by-step tutorial.
