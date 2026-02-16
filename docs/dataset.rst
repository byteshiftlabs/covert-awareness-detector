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
:Participants: 25 healthy volunteers
:License: CC0 1.0 Universal (Public Domain)

The dataset was collected by the University of Michigan Anesthesiology Department
to study neural signatures of consciousness during propofol-induced sedation.

**About fMRI and BOLD Signal**

Functional MRI measures brain activity via the **BOLD (Blood Oxygen Level Dependent)** signal.
When neurons fire, they consume oxygen, triggering increased blood flow to that region. 
Oxygenated and deoxygenated blood have different magnetic properties, creating detectable 
MRI signal changes that serve as an indirect measure of neural activity.

- **Official Reference**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3073717/
- Ogawa, S., Lee, T. M., Kay, A. R., & Tank, D. W. (1990). Brain magnetic resonance imaging 
  with contrast dependent on blood oxygenation. *PNAS*, 87(24), 9868-9872.


Data Processing Pipeline
========================

This project uses **preprocessed and post-processed derivatives** from OpenNeuro — not raw scanner images.

Stage 1: Dataset Authors (fMRIPrep → XCP-D)
--------------------------------------------

**fMRIPrep (Preprocessing)**
  Standard minimal preprocessing: motion correction, distortion correction, brain extraction, 
  and spatial normalization to MNI152 space.

**XCP-D (Post-processing)**
  Operates on fMRIPrep outputs to:
  
  - Parcellate brain into 456 regions (4S456Parcels: Schaefer-400 + Tian-56)
  - Extract regional BOLD timeseries (456 columns)
  - Compute motion quality metrics (framewise displacement)
  
  Configuration: ``xcp_d_without_GSR_bandpass_output`` (no global signal regression, no bandpass filtering)

.. important::
   We download these ready-to-use XCP-D derivatives from OpenNeuro.

Stage 2: Our Analysis Code
---------------------------

Additional processing on XCP-D outputs:

1. **Motion filtering**: Remove timepoints where FD ≥ 0.8 mm
2. **ROI selection**: Use first 446 of 456 regions (following reference MATLAB implementation)
3. **Temporal segmentation**: Split into 7 conditions using LOR/ROR timing (skip 375 TRs around transitions)
4. **Connectivity**: Compute 446×446 Pearson correlation matrix (diagonal = 0)

What We Load
============

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
(Tian atlas). 

Following the reference MATLAB implementation, this code uses only the **first
446 regions**. The rationale for excluding the last 10 regions is not documented
in the original code or paper.


Connectivity Matrix Computation
================================

For each condition:

1. Load regional timeseries (446 regions × T timepoints) and motion parameters from XCP-D
2. Filter timepoints where FD ≥ 0.8 mm
3. Compute Pearson correlation between all region pairs
4. Set diagonal to zero → 446×446 connectivity matrix

.. note::
   If all timepoints are motion-censored, the matrix is filled with NaN and excluded from analysis.


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
     - Resting state 1 (Wakeful Baseline)
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
     - Imagery 3 after return of responsiveness (ROR)
     - Conscious
   * - 5
     - Imagery 4 (Recovery Baseline)
     - Conscious
   * - 6
     - Resting state 2
     - Conscious

For binary classification: condition 3 is labelled **unconscious** (0), all
others are labelled **conscious** (1).


LOR and ROR Timing
-------------------

**Loss of Responsiveness (LOR)** and **Return of Responsiveness (ROR)** times
are defined as TR (timepoint) indices, taken from the reference MATLAB code. 
They indicate where the subject stopped and resumed responding to auditory commands.

The code skips **375 TRs** around each transition to avoid ambiguous periods
where the subject is between states.

**Special case:** Subject sub-29 has no data after ROR in Imagery 3 (condition 4 is marked as missing data).


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
training.


Subjects
========

The 25 subjects used (as defined in the code):

::

   sub-02  sub-03  sub-04  sub-05  sub-06  sub-07
   sub-11  sub-12  sub-13  sub-14  sub-15  sub-16
   sub-17  sub-18  sub-19  sub-20  sub-21  sub-22
   sub-23  sub-24  sub-25  sub-26  sub-27  sub-28
   sub-29


References
==========

**Dataset paper:**

- Huang, Z., Tarnal, V., Fotiadis, P., Vlisides, P. E., Janke, E. L., Puglia, M., McKinney, A. M., 
  Jang, H., Dai, R., Picton, P., Mashour, G. A., & Hudetz, A. G. (2026). 
  An open fMRI resource for studying human brain function and covert consciousness under anesthesia. 
  *Scientific Data*, 13(1), Article 127. https://doi.org/10.1038/s41597-025-06442-2

**Preprocessing pipelines:**

- **BIDS**: Brain Imaging Data Structure specification — https://bids.neuroimaging.io/
  
  The dataset follows BIDS format, ensuring standardized file organization and metadata.

- **fMRIPrep**: Minimal preprocessing pipeline — https://fmriprep.org/
  
  Performs anatomical preprocessing, motion correction, distortion correction, and spatial normalization.
  
  Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). fMRIPrep: a robust preprocessing 
  pipeline for functional MRI. *Nature Methods*, 16(1), 111-116. https://doi.org/10.1038/s41592-018-0235-4

- **XCP-D**: Post-processing and parcellation — https://xcp-d.readthedocs.io/
  
  Performs parcellation into brain atlases and computes quality/motion metrics.
  
  Ciric, R., Thompson, W. H., Lorenz, R., et al. (2018). Benchmarking of participant-level confound 
  regression strategies for the control of motion artifact in studies of functional connectivity. 
  *NeuroImage*, 154, 174-187. https://doi.org/10.1016/j.neuroimage.2017.03.020

**Brain atlases:**

- **Schaefer 400**: 400 cortical parcels — https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
  
  Schaefer, A., Kong, R., Gordon, E. M., et al. (2018). Local-Global Parcellation of the Human 
  Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.

- **Tian 56**: 56 subcortical parcels — https://github.com/yetianmed/subcortex
  
  Tian, Y., Margulies, D. S., Breakspear, M., & Zalesky, A. (2020). Topographic organization of 
  the human subcortex unveiled with functional connectivity gradients. *Nature Neuroscience*, 23(11), 1421-1432.
