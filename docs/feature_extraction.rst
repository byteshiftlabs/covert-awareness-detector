==================
Feature Extraction
==================

This document explains how the project extracts meaningful features from
fMRI connectivity matrices for consciousness classification.

.. contents:: Table of Contents
   :local:
   :depth: 2


What Are We Extracting?
=======================

The input is a **connectivity matrix**: a table where each cell contains a
number representing how strongly two brain regions are synchronised. The
matrix has 446 rows and 446 columns (one per brain region), so there are
roughly 99,000 unique pairs of regions.

Raw connectivity matrices are too large and noisy to feed directly into a
classifier. Feature extraction condenses them into a small set of numbers
that summarise how the brain is organised.


The Key Metric: ISD
====================

The project's central feature is the **Integration-Segregation Difference
(ISD)**, derived from the reference paper's MATLAB code. It measures the
balance between two properties of brain networks:

- **Integration (efficiency)**: How easily information can travel between
  any two brain regions. High efficiency means any region can communicate
  with any other region through short paths.

- **Segregation (clustering)**: How much brain regions form tight local
  groups where neighbours are also connected to each other. High clustering
  means the brain has specialised modules.

The ISD is simply:

::

   ISD = efficiency - clustering

A higher ISD indicates a more integrated brain state, which is associated
with consciousness. Loss of consciousness (e.g. under anaesthesia) reduces
the ISD.


How ISD Is Computed
-------------------

1. **Global signal removal**: The strongest shared pattern across all
   regions is removed from the connectivity matrix. This isolates the
   more specific, region-to-region connections from the overall background
   signal.

2. **Multilevel efficiency**: The connectivity matrix is converted into a
   series of binary networks at different thresholds (from very lenient to
   very strict). At each threshold, the average path efficiency is
   computed. These values are then combined across all thresholds.

3. **Multilevel clustering**: The same thresholding procedure is applied
   to the cleaned matrix (after global signal removal). At each threshold,
   the average clustering coefficient is computed and then combined.

4. **ISD**: The difference between the integrated efficiency and the
   integrated clustering.


Additional Features
===================

Beyond ISD, the project extracts two other groups of features from each
connectivity matrix.

Graph Metrics
-------------

The connectivity matrix is turned into a network (a graph) by treating
each brain region as a node and each strong enough connection as a link.
From this graph, we compute:

- **Mean degree**: On average, how many connections does each region have?
- **Degree variability**: How much does the number of connections vary
  across regions? (Some regions may be hubs with many connections.)
- **Mean strength**: The average total connection weight per region.
- **Strength variability**: How much connection strength varies across
  regions.
- **Density**: What fraction of all possible connections actually exist?

Statistical Summary
-------------------

Basic statistics computed over all pairwise connectivity values:

- Mean, standard deviation, median
- Skewness (is the distribution of connection strengths symmetric or
  lopsided?)
- Kurtosis (does the distribution have heavy tails — are there extreme
  values?)
- Percentiles (25th, 75th) and min/max values

These numbers give a quick profile of the overall connectivity
distribution.


Putting It All Together
=======================

For each brain scan, the feature extraction produces a small dictionary:

- 3 ISD-related values (ISD, efficiency, clustering)
- 5 graph metrics
- 9 statistical summaries
- Optionally, the full set of ~99,000 pairwise connectivity values

In practice, the classifier uses the compact set of 17 features. The
full connectivity vector is available but is high-dimensional and
requires dimensionality reduction (e.g. PCA) before use.

.. code-block:: python

   from src.features import extract_all_features

   features = extract_all_features(connectivity_matrix)
   print(features['isd'])         # Integration-Segregation Difference
   print(features['efficiency'])  # Multilevel efficiency
   print(features['clustering'])  # Multilevel clustering


Next Steps
==========

- See :doc:`model_architectures` for how these features are used by the classifiers.
- See :doc:`dataset` for information about the fMRI data.
- See :doc:`quickstart` for a step-by-step tutorial.
