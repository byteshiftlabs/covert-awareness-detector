=======
Dataset
=======

This page describes the dataset used in the Covert Awareness Detector project
and how the code loads and labels it.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
========

:Dataset: Michigan Human Anesthesia fMRI Dataset
:OpenNeuro ID: ds006623
:DOI: `10.18112/openneuro.ds006623.v1.0.0 <https://doi.org/10.18112/openneuro.ds006623.v1.0.0>`_
:Participants: 25 healthy volunteers (26 scanned; one excluded for missing timing data)
:License: CC0 1.0 Universal (Public Domain)

The dataset was collected by the University of Michigan Anesthesiology Department
to study neural signatures of consciousness during propofol-induced sedation.


What the Data Contains
======================

The project uses **preprocessed derivatives** only — not the raw scanner images.
The preprocessing was performed by the dataset authors using fMRIPrep and XCP-D.

For each subject and scan, we load two files from the XCP-D output directory:

- **Timeseries file** (``*_timeseries.tsv``): A table where each row is a timepoint
  and each column is a brain region. We use the first 446 columns (brain regions
  from the 4S456Parcels atlas).

- **Motion file** (``*_motion.tsv``): Head movement parameters for each timepoint.
  Column 8 contains framewise displacement (FD), which measures how much the head
  moved between consecutive timepoints.


Atlas
=====

The brain is divided into 456 regions using the **4S456Parcels** atlas, which
combines 400 cortical regions (Schaefer atlas) with 56 subcortical regions
(Tian atlas). Following the reference paper, the code uses only the **first
446 regions**.


From Timeseries to Connectivity Matrix
=======================================

For each scan, the code:

1. Loads the timeseries (446 brain regions over time).
2. Removes timepoints where the head moved too much (FD ≥ 0.8 mm).
3. Computes the Pearson correlation between every pair of brain regions.
4. Sets the diagonal to zero.

The result is a **446 × 446 connectivity matrix** — a symmetric table where each
cell indicates how strongly two brain regions are synchronised.


Experimental Conditions
=======================

Each subject went through a sedation protocol with mental imagery tasks. The code
segments each subject's data into **7 conditions**:

.. list-table::
   :header-rows: 1
   :widths: 5 25 15

   * - ID
     - Condition
     - Label
   * - 0
     - Resting state, run 1 (baseline)
     - Conscious
   * - 1
     - Imagery, run 1 (awake, pre-sedation)
     - Conscious
   * - 2
     - Imagery, run 2 before loss of responsiveness (preLOR)
     - Conscious
   * - 3
     - Imagery, runs 2–3 during loss of responsiveness (LOR)
     - **Unconscious**
   * - 4
     - Imagery, run 3 after return of responsiveness (postROR)
     - Conscious
   * - 5
     - Imagery, run 4 (recovery)
     - Conscious
   * - 6
     - Resting state, run 2 (recovery)
     - Conscious

For binary classification: condition 3 is labelled **unconscious** (0), all
others are labelled **conscious** (1).


LOR and ROR Timing
-------------------

**Loss of Responsiveness (LOR)** and **Return of Responsiveness (ROR)** times
are defined as TR (timepoint) indices, taken directly from the reference paper's
MATLAB code. They indicate where the subject stopped and resumed responding to
auditory commands.

The code skips **375 TRs** around each transition to avoid ambiguous periods
where the subject is between states.

**Special case:** Subject sub-29 has no postROR segment (condition 4 is marked
as missing data).


Missing Data
============

Some conditions produce no usable data (e.g. a scan file is missing, or all
timepoints were censored for motion). In those cases, the connectivity matrix
is filled with NaN and excluded from training and evaluation.


Cross-Validation
================

The project uses **Leave-One-Subject-Out (LOSO)** cross-validation:

1. Hold out one subject as the test set.
2. Train on the remaining 24 subjects.
3. Test on the held-out subject.
4. Repeat for all 25 subjects.

This ensures the model is always tested on a subject it has never seen during
training. All three classifiers use **balanced class weights** to account for
the fact that there are more conscious than unconscious samples.


Subjects
========

The 25 subjects used (as defined in the code):

::

   sub-02  sub-03  sub-04  sub-05  sub-06  sub-07
   sub-11  sub-12  sub-13  sub-14  sub-15  sub-16
   sub-17  sub-18  sub-19  sub-20  sub-21  sub-22
   sub-23  sub-24  sub-25  sub-26  sub-27  sub-28
   sub-29

Subject sub-30 is excluded (missing timing data in the reference paper).


References
==========

**Original research paper:**

- Huang et al. (2018). *Scientific Reports*.
  `DOI: 10.1038/s41598-018-31436-z <https://doi.org/10.1038/s41598-018-31436-z>`_

**Preprocessing tools:**

- BIDS specification: https://bids.neuroimaging.io/
- fMRIPrep: https://fmriprep.org/
- XCP-D: https://xcp-d.readthedocs.io/
