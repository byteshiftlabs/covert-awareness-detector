========
Dataset
========

This page provides a comprehensive guide to the Michigan Human Anesthesia fMRI Dataset 
(OpenNeuro ds006623) used in the Covert Awareness Detector project. It explains the data 
structure, preprocessing, and labeling strategy for machine learning practitioners.

.. contents:: Table of Contents
   :depth: 3
   :local:


Dataset Overview
================

**Michigan Human Anesthesia fMRI Dataset**

:Dataset ID: ds006623 (OpenNeuro)
:DOI: `10.18112/openneuro.ds006623.v1.0.0 <https://doi.org/10.18112/openneuro.ds006623.v1.0.0>`_
:Participants: 26 healthy volunteers (ages 18-40, final analysis uses 25)
:Total Size: ~100GB (raw) | ~2-5GB (preprocessed derivatives only)
:Format: BIDS-compliant neuroimaging data
:License: CC0 1.0 Universal (Public Domain)

**Research Context:**

This dataset was collected by the University of Michigan Anesthesiology Department 
(Huang, Hudetz, Mashour et al.) to study neural signatures of consciousness during 
propofol-induced sedation. It represents the first large-scale fMRI investigation of 
"covert consciousness" - the phenomenon where patients can follow mental imagery 
commands despite showing no behavioral response.


What is fMRI Data?
==================

For ML Practitioners Without Neuroimaging Background
-----------------------------------------------------

**Functional Magnetic Resonance Imaging (fMRI)** measures brain activity by detecting 
changes in blood oxygenation. Think of it as a 4D movie of the brain:

- **Spatial dimensions (3D):** The brain divided into small cubes called **voxels** 
  (volume pixels), typically 2-3mm on each side
- **Temporal dimension (1D):** Snapshots taken every few seconds (here: TR = 3.0 seconds)
- **Signal:** BOLD (Blood Oxygen Level Dependent) - higher activity → more oxygen → 
  stronger signal

**Key Characteristics:**

.. code-block:: text

   Raw fMRI volume at each timepoint:
   ┌──────────────────────────────────┐
   │  Shape: (91, 109, 91) voxels     │  ~900,000 measurements
   │  Resolution: 2mm isotropic       │  per timepoint
   │  Coverage: whole brain           │
   └──────────────────────────────────┘
   
   Temporal sampling:
   ├─────┬─────┬─────┬─────┬─────┬
   0s    3s    6s    9s    12s   ...  (TR = 3.0 seconds)
   
   One "run" = ~600-2,000 timepoints = 30-100 minutes of data

**Challenges for ML:**

1. **High dimensionality:** 900k voxels × 1,000 timepoints = 900 million data points per subject
2. **Low sample size:** Neuroimaging studies typically have 20-100 subjects
3. **Noisy:** Subject movement, scanner drift, physiological noise
4. **Expensive:** Each fMRI scan costs $500-1,000 and takes hours

This is why preprocessing and dimensionality reduction (via ROI atlases) are critical.


BIDS Format Basics
==================

**Brain Imaging Data Structure (BIDS)** is a standardized way to organize neuroimaging data. 
Understanding BIDS is essential for navigating the dataset.

Directory Structure
-------------------

.. code-block:: text

   ds006623/
   ├── dataset_description.json        # Dataset metadata
   ├── participants.tsv                # Subject demographics
   ├── README                          # Study description
   ├── CHANGES                         # Version history
   │
   ├── sub-02/                         # Subject-level folders
   │   ├── anat/                       # Anatomical (structural) MRI
   │   │   └── sub-02_T1w.nii.gz      # T1-weighted image (brain structure)
   │   └── func/                       # Functional MRI (BOLD timeseries)
   │       ├── sub-02_task-rest_run-1_bold.nii.gz        # Resting state run
   │       ├── sub-02_task-imagery_run-1_bold.nii.gz     # Mental imagery run
   │       └── sub-02_task-imagery_run-1_events.tsv      # Task timing
   │
   └── derivatives/                    # Preprocessed data (our focus)
       ├── Participant_Info.csv        # Clinical metadata
       ├── LOR_ROR_Timing.csv          # Consciousness transition times
       │
       ├── fmriprep/                   # Standard preprocessing pipeline
       │   └── sub-02/
       │       ├── anat/               # Normalized anatomical scans
       │       └── func/               # Realigned, normalized BOLD data
       │
       └── xcp_d_without_GSR_bandpass_output/   # Connectivity preprocessing
           └── sub-02/
               └── func/
                   ├── *_timeseries.tsv        # ROI timeseries (what we use)
                   ├── *_motion.tsv            # Head movement parameters
                   └── *_connectivity.tsv      # Precomputed correlations

**Key Naming Conventions:**

- ``sub-02``: Subject identifier (02-30, with gaps)
- ``task-rest``: Resting state (eyes closed, no task)
- ``task-imagery``: Mental imagery task (tennis, navigation, hand)
- ``run-1``, ``run-2``: Sequential scanning sessions
- ``space-MNI152NLin2009cAsym``: Normalized to standard brain template
- ``seg-4S456Parcels``: Brain parcellated into 456 regions


Preprocessing Pipeline
======================

The dataset has been preprocessed using two industry-standard tools that handle the 
complex signal processing required for fMRI analysis.

fMRIPrep: Anatomical and Functional Preprocessing
--------------------------------------------------

**fMRIPrep** (v23.1.3) performed the initial preprocessing. This is the gold standard 
pipeline used by most neuroimaging labs.

**Steps Performed:**

1. **Head Motion Correction**
   
   - Folds brain back in time to compensate for subject movement
   - Computes framewise displacement (FD) - how much the head moved between frames
   - Critical: even 1-2mm of motion severely degrades data quality

2. **Slice-Timing Correction**
   
   - Brain slices are acquired sequentially (not all at once)
   - Interpolates to align all slices to the same temporal reference
   
3. **Coregistration**
   
   - Aligns functional data to high-resolution anatomical scan
   - Ensures we know which brain regions correspond to which voxels

4. **Spatial Normalization**
   
   - Warps individual brains to MNI152 standard template space
   - Allows cross-subject comparison (different brain shapes → common coordinates)

5. **Confound Estimation**
   
   - Extracts nuisance signals: motion, CSF pulsation, white matter
   - Saved for later regression

**Output:** Preprocessed 4D BOLD timeseries in standard space, ready for connectivity analysis.


XCP-D: Connectivity-Specific Postprocessing
--------------------------------------------

**XCP-D** (eXtensible Connectivity Pipeline - DCAN labs) performed connectivity-focused 
denoising and parcellation. This pipeline is specialized for functional connectivity analysis.

**Key Configuration:**

.. code-block:: text

   Pipeline: xcp_d_without_GSR_bandpass
   ├── Motion censoring: FD threshold = 0.8mm
   ├── Confound regression: 36-parameter model (NO global signal regression)
   ├── Bandpass filter: DISABLED (preserves low-frequency connectivity)
   └── Spatial smoothing: 6mm FWHM Gaussian kernel

**Why No Global Signal Regression (GSR)?**

GSR is controversial in consciousness research:

- **Pro:** Removes physiological noise, increases specificity
- **Con:** Can introduce spurious anti-correlations, removes consciousness-related global changes

The researchers chose to **preserve global signal** because consciousness states show 
large-scale network-level changes that GSR would eliminate.

**Why No Bandpass Filtering?**

Typical fMRI studies filter to 0.01-0.1 Hz (slow fluctuations). This dataset keeps the 
full frequency spectrum because:

- Anesthesia affects slow cortical rhythms (<0.01 Hz)
- Consciousness may involve infraslow fluctuations
- Overly aggressive filtering can remove signal of interest

**Parcellation and ROI Timeseries Extraction:**

The preprocessed 4D data (900k voxels) is reduced to a manageable number of regions 
using brain atlases. For each atlas parcel (ROI), the mean BOLD signal across all 
voxels in that region is computed at each timepoint.

**Output format (``*_timeseries.tsv``):**

.. code-block:: text

   timepoint    ROI_1    ROI_2    ROI_3  ...  ROI_456
   ─────────────────────────────────────────────────────
   0            0.142   -0.031    0.089  ...   0.201
   1            0.156   -0.028    0.091  ...   0.198
   2            0.148   -0.035    0.087  ...   0.203
   ...          ...      ...      ...    ...   ...

This is the **input to our ML pipeline** - much more tractable than raw voxels!


ROI Atlases
===========

Brain atlases divide the brain into discrete regions based on anatomy, function, or 
connectivity. We use three complementary atlases.

Gordon Atlas (333 ROIs)
-----------------------

:Reference: Gordon et al. (2016) *Cerebral Cortex*
:Regions: 333 cortical parcels
:Basis: Data-driven functional connectivity boundaries
:Advantages: Respects functional organization, avoids arbitrary anatomical divisions

The Gordon atlas was created by identifying "cliffs" in connectivity patterns - places 
where connectivity abruptly changes, indicating functional boundaries.


Schaefer Atlas (400/417 ROIs)
------------------------------

:Reference: Schaefer et al. (2018) *Cerebral Cortex*
:Regions: 400 cortical (7-network) or 417 (17-network)
:Basis: Multi-resolution functional gradients + Yeo networks
:Advantages: Hierarchical, matches canonical brain networks (DMN, attention, etc.)

The Schaefer atlas provides parcellations at different "zoom levels" - coarser (7 networks) 
or finer (17 networks) depending on your research question.


Glasser Atlas (360 ROIs)
-------------------------

:Reference: Glasser et al. (2016) *Nature*
:Regions: 360 cortical areas (HCP Multi-Modal Parcellation)
:Basis: Multimodal features (myelin, thickness, connectivity, task activations)
:Advantages: Gold standard from Human Connectome Project, integrates multiple data types

The Glasser atlas represents the state-of-the-art in cortical parcellation, combining 
structural, functional, and architectural features.


4S456Parcels Atlas (446 ROIs Used)
-----------------------------------

**This is the atlas used in our analysis** (from XCP-D).

:Full Name: 4S456Parcels (Tian subcortex + Schaefer 400 cortex)
:Regions: 456 total (400 cortical + 56 subcortical)
:Used in Analysis: **First 446 ROIs only** (matches original paper)
:Why 446?: Excludes certain brainstem/cerebellar regions with low SNR

**Composition:**

.. code-block:: text

   4S456Parcels Atlas (456 total)
   ├── Schaefer 400 (cortex)
   │   ├── Visual network: 37 regions
   │   ├── Somatomotor: 55 regions
   │   ├── Dorsal attention: 45 regions
   │   ├── Ventral attention: 23 regions
   │   ├── Limbic: 19 regions
   │   ├── Frontoparietal: 49 regions
   │   └── Default mode: 72 regions
   └── Tian S4 (subcortex)
       ├── Thalamus: 16 regions
       ├── Striatum: 16 regions
       ├── Amygdala: 6 regions
       ├── Hippocampus: 8 regions
       └── Other subcortical: 10 regions

**Why Subcortex Matters for Consciousness:**

Subcortical structures (thalamus, striatum) are **critical** for consciousness:

- **Thalamus:** Gateway for cortical communication, "consciousness switch"
- **Striatum:** Involved in arousal and goal-directed behavior
- **Brainstem:** Regulates sleep/wake states (excluded here due to signal quality)

The 4S456Parcels atlas is ideal because it captures both cortical networks 
(high-level cognition) and subcortical regulators (arousal/awareness).


Connectivity Matrices vs Timeseries
====================================

Understanding the Data Representation
--------------------------------------

**Timeseries** and **connectivity matrices** are two different views of the same data:

.. code-block:: text

   ROI Timeseries (temporal view):
   ┌────────────────────────────────────────────────────────────┐
   │  Shape: (n_timepoints, 446)                                │
   │  Example: (1200, 446) = 20 minutes at TR=3s               │
   │  Each column = signal from one brain region over time      │
   │  Each row = snapshot of all 446 regions at one moment      │
   └────────────────────────────────────────────────────────────┘
   
             ROI_1    ROI_2    ROI_3  ...  ROI_446
   time=0    0.142   -0.031   0.089  ...   0.201
   time=3    0.156   -0.028   0.091  ...   0.198
   time=6    0.148   -0.035   0.087  ...   0.203
   ...       ...      ...     ...    ...   ...

   
   Functional Connectivity Matrix (spatial view):
   ┌────────────────────────────────────────────────────────────┐
   │  Shape: (446, 446)                                         │
   │  Entry [i,j] = Pearson correlation between ROI_i and ROI_j │
   │  Symmetric matrix (correlation is bidirectional)           │
   │  Diagonal = 0 (we set self-correlation to zero)           │
   └────────────────────────────────────────────────────────────┘
   
          ROI_1  ROI_2  ROI_3  ...  ROI_446
   ROI_1  0.00   0.72  -0.15  ...   0.08
   ROI_2  0.72   0.00   0.45  ...   0.12
   ROI_3 -0.15   0.45   0.00  ...  -0.03
   ...    ...    ...    ...   ...   ...

**Computing Connectivity:**

1. Take timeseries for all 446 ROIs
2. Compute Pearson correlation between every pair
3. Result: 446 × 446 symmetric matrix with 99,235 unique connections

.. math::

   r_{ij} = \frac{\text{cov}(X_i, X_j)}{\sigma_i \sigma_j} 
          = \frac{\sum_t (x_{it} - \bar{x}_i)(x_{jt} - \bar{x}_j)}
                 {\sqrt{\sum_t (x_{it} - \bar{x}_i)^2} \sqrt{\sum_t (x_{jt} - \bar{x}_j)^2}}

Where:
- :math:`X_i` = timeseries of ROI *i*
- :math:`t` indexes timepoints
- :math:`r_{ij} \in [-1, 1]` = functional connectivity strength

**Why Use Connectivity Matrices?**

- **Dimensionality reduction:** 1200 × 446 → 446 × 446 (removes temporal complexity)
- **Interpretability:** Each entry = strength of communication between two brain regions
- **Stability:** Averaged over time, less sensitive to moment-to-moment fluctuations
- **Graph structure:** Natural representation for graph neural networks


Motion Censoring
----------------

Head motion is the #1 confound in fMRI connectivity analysis. Even small movements 
create spurious correlations.

**Framewise Displacement (FD):**

FD quantifies how much the head moved between consecutive timepoints:

.. math::

   FD_t = |\Delta d_{x,t}| + |\Delta d_{y,t}| + |\Delta d_{z,t}| 
        + r|\Delta \alpha_t| + r|\Delta \beta_t| + r|\Delta \gamma_t|

Where:
- :math:`\Delta d` = translational movement (mm)
- :math:`\Delta \alpha, \beta, \gamma` = rotational movement (degrees)
- :math:`r` = 50mm (assumed radius of head sphere)

**Our Censoring Strategy:**

.. code-block:: python

   # From config.py
   FD_THRESHOLD = 0.8  # mm
   
   # From data_loader.py
   good_timepoints = (framewise_displacement < 0.8)
   timeseries_clean = timeseries[good_timepoints]

Timepoints with FD ≥ 0.8mm are **excluded** before computing connectivity. This is a 
conservative threshold (many studies use 0.5mm) balancing data quality vs. retention.

**Impact on Data:**

- Typical subject: 5-15% of timepoints censored
- High-motion subject: 30-50% censored
- If <100 good timepoints remain → mark condition as NaN (insufficient data)


Ground Truth Labels
===================

Consciousness State Definitions
--------------------------------

The experiment tracks consciousness through behavioral responsiveness to auditory commands 
during mental imagery tasks.

**7 Experimental Conditions:**

.. code-block:: text

   Condition 0: rest_run-1          └─ Baseline (awake, eyes closed)
   Condition 1: imagery_run-1       ├─ CONSCIOUS (responsive)
   Condition 2: imagery_preLOR      ├─ CONSCIOUS (sedated but responsive)
   ─────────────────────────────── LOSS OF RESPONSIVENESS (LOR) ───────────
   Condition 3: imagery_LOR         ├─ UNCONSCIOUS (unresponsive)
   ─────────────────────────────── RETURN OF RESPONSIVENESS (ROR) ─────────
   Condition 4: imagery_postROR     ├─ CONSCIOUS (recovering)
   Condition 5: imagery_run-4       ├─ CONSCIOUS (recovered)
   Condition 6: rest_run-2          └─ CONSCIOUS (awake, eyes closed)

**Binary Labels:**

- **Label 0 (unconscious):** Condition 3 only
- **Label 1 (conscious):** Conditions 0, 1, 2, 4, 5, 6

**Multi-class Labels:**

- **Class 0:** Baseline rest
- **Class 1:** Awake imagery
- **Class 2:** Sedated conscious (preLOR)
- **Class 3:** Unconscious (LOR)
- **Class 4:** Recovering (postROR)
- **Class 5:** Recovered imagery
- **Class 6:** Recovery rest


LOR/ROR Timing
--------------

**Loss of Responsiveness (LOR):** The timepoint (TR index) when the subject stops 
responding to commands despite repeated prompts.

**Return of Responsiveness (ROR):** The timepoint when the subject resumes responding 
to commands as propofol wears off.

**Data Format (``LOR_ROR_Timing.csv``):**

.. code-block:: text

   subject_id, lor_time, ror_time, propofol_dose, ...
   sub-02,     1160,     673,      ...
   sub-03,     1385,     410,      ...

**Temporal Segmentation:**

The imagery scanning protocol consists of long continuous runs (30-60 minutes) with 
changing consciousness states. We segment these runs based on LOR/ROR timing:

.. code-block:: text

   Imagery Run-2 (before LOR):
   ├──────────────────┬──────────────────────────────────────┐
   │  preLOR          │          LOR period                  │
   │  (conscious)     │     (unconscious, continues into     │
   │                  │           run-3)                     │
   0 ──────────→ lor_time ───→ lor + 375 ─────→ end_of_run
                      ▲              ▲
                      │              └─ transition buffer (6.25 min)
                      └─ loss of responsiveness
   
   Imagery Run-3 (after ROR):
   ├──────────────────────────┬────────────┬──────────────────┐
   │    LOR period            │ transition │   postROR        │
   │  (unconscious,           │   buffer   │  (conscious)     │
   │   continued from run-2)  │            │                  │
   0 ────→ ror - 375 ────→ ror_time ─────────────→ end_of_run
                  ▲               ▲
                  │               └─ return of responsiveness
                  └─ transition buffer (6.25 min)

**Transition Buffer (375 TRs = 18.75 minutes):**

We exclude ±375 TRs (~19 minutes) around LOR/ROR transitions because:

1. Consciousness changes are gradual, not instantaneous
2. Behavioral assessment has temporal uncertainty (~1-2 minutes)
3. Hemodynamic lag: BOLD signal lags neural activity by 4-6 seconds
4. Propofol pharmacokinetics: effect-site concentration changes slowly

Excluding these transition periods ensures "pure" conscious vs unconscious states.

**Special Cases:**

- **sub-29:** No postROR segment (remained unconscious longer)
- **sub-30:** Excluded from analysis (missing timing data)


Computing Condition Labels
---------------------------

**Pseudocode for label assignment:**

.. code-block:: python

   def get_condition_label(subject, condition_id):
       """Map condition to consciousness state."""
       if condition_id == 3:
           return "unconscious"
       else:
           return "conscious"
   
   def segment_run_2(subject):
       """Split run-2 into preLOR and LOR segments."""
       lor_time = LOR_TIME[subject]
       
       # preLOR: timepoints 0 to lor_time (condition 2)
       preLOR_timeseries = run2[:lor_time]
       preLOR_label = "conscious"
       
       # LOR: timepoints (lor+375) to end (part of condition 3)
       LOR_timeseries = run2[lor_time + 375:]
       LOR_label = "unconscious"


Data Splits and Cross-Validation Strategy
==========================================

Challenge: Small Sample Size
-----------------------------

With only **25 subjects** and **7 conditions each** (175 samples total), standard 
train/test splits are problematic:

- Random 80/20 split: Only 5 test subjects → high variance in performance estimates
- Fixed test set: Can't assess generalization robustness
- Condition imbalance: 6 conscious conditions vs. 1 unconscious

**Solution:** Leave-One-Subject-Out Cross-Validation (LOSO-CV)

Leave-One-Subject-Out Cross-Validation (LOSO-CV)
------------------------------------------------

**Rationale:** Test on completely unseen subjects to assess **cross-subject generalization**, 
which is critical for clinical deployment.

**Procedure:**

.. code-block:: text

   Fold 1:  Train on [sub-03, sub-04, ..., sub-29]  →  Test on [sub-02]
   Fold 2:  Train on [sub-02, sub-04, ..., sub-29]  →  Test on [sub-03]
   Fold 3:  Train on [sub-02, sub-03, sub-05, ...]  →  Test on [sub-04]
   ...
   Fold 25: Train on [sub-02, sub-03, ..., sub-28]  →  Test on [sub-29]
   
   Final performance: Average across all 25 folds

**Implementation:**

.. code-block:: python

   from sklearn.model_selection import LeaveOneGroupOut
   
   # X: (175, 446, 446) connectivity matrices
   # y: (175,) labels
   # groups: (175,) subject IDs
   
   logo = LeaveOneGroupOut()
   for train_idx, test_idx in logo.split(X, y, groups=subjects):
       X_train, X_test = X[train_idx], X[test_idx]
       y_train, y_test = y[train_idx], y[test_idx]
       
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)

**Advantages:**

- Realistic evaluation: models never see the test subject during training
- Maximizes training data: uses 24/25 subjects for training
- No data leakage: test subject is completely held out
- Clinically relevant: mimics deploying to new patients

**Disadvantages:**

- Expensive: must train 25 models
- High variance: some subjects are "easy", others "hard"
- Small test sets: only 7 samples per fold


Stratified Sampling for Balanced Training
------------------------------------------

The dataset has severe class imbalance: 6 conscious conditions vs. 1 unconscious.

**Problem:**

.. code-block:: text

   Class distribution (per subject):
   ├─ Conscious (0,1,2,4,5,6):   6 samples (86%)
   └─ Unconscious (3):            1 sample (14%)
   
   In training set (24 subjects):
   ├─ Conscious:   144 samples
   └─ Unconscious:  24 samples   ← severe minority

**Solution: Balanced Sampling**

**Option 1: Oversample minority class**

.. code-block:: python

   from imblearn.over_sampling import RandomOverSampler
   
   ros = RandomOverSampler(random_state=42)
   X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
   # Now: 144 conscious + 144 unconscious (duplicated)

**Option 2: Undersample majority class**

.. code-block:: python

   from imblearn.under_sampling import RandomUnderSampler
   
   rus = RandomUnderSampler(random_state=42)
   X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
   # Now: 24 conscious + 24 unconscious (conscious samples discarded)

**Option 3: Class weights** (preferred)

.. code-block:: python

   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       'balanced', classes=np.unique(y_train), y=y_train
   )
   # conscious: 0.167, unconscious: 1.0
   # Model loss penalizes misclassifying unconscious samples more

**We use class weights** to avoid discarding data (undersampling) or creating 
artificial duplicates (oversampling).


Validation Strategy Within Training
------------------------------------

During hyperparameter tuning, we need a validation set **within** the training fold.

**Nested Cross-Validation:**

.. code-block:: text

   Outer Loop (LOSO): 25 folds
   ├─ Fold 1 (test on sub-02):
   │   ├─ Training subjects: [sub-03, ..., sub-29] (24 subjects)
   │   └─ Inner loop: 5-fold CV within training subjects
   │       ├─ Inner fold 1: train on 19 subjects, validate on 5
   │       ├─ Inner fold 2: train on 19 subjects, validate on 5
   │       ├─ ... (find best hyperparameters)
   │       └─ Retrain on all 24 with best hyperparameters
   │   └─ Test on sub-02
   └─ Repeat for all 25 outer folds

**Practical Simplification (used in this project):**

.. code-block:: python

   # Option A: Fixed validation subjects for speed
   VALIDATION_SUBJECTS = ['sub-02', 'sub-03', 'sub-04', 'sub-05']
   
   for test_subject in all_subjects:
       if test_subject in VALIDATION_SUBJECTS:
           continue  # Skip (becomes test in another fold)
       
       train_subjects = [s for s in all_subjects 
                        if s != test_subject 
                        and s not in VALIDATION_SUBJECTS]
       val_subjects = VALIDATION_SUBJECTS
       
       # Tune on train→val, then evaluate on test

**This approach is faster but less rigorous than full nested CV.**


Handling Missing Data
---------------------

Some conditions are missing for some subjects due to:

1. High motion corruption (all timepoints censored)
2. Scanner/acquisition failures
3. Incomplete runs (subject discomfort, early termination)

**Missing Data Strategy:**

.. code-block:: python

   # During loading
   connectivity = compute_connectivity(timeseries_filtered)
   
   if len(timeseries_filtered) < 100:  # arbitrary threshold
       return np.full((446, 446), np.nan)
   
   # During ML
   valid_mask = ~np.isnan(connectivity).any(axis=(1,2))
   X_clean = X[valid_mask]
   y_clean = y[valid_mask]

**Affected subjects/conditions:**

- sub-29: No postROR segment (condition 4 = NaN)
- sub-06, sub-14: High motion in some runs (sparse NaNs)

**Typical outcome:** ~165-170 valid samples out of 175 total.


Dataset Summary
===============

**Quick Reference Table:**

+------------------------+--------------------------------------------------------+
| **Feature**            | **Value**                                              |
+========================+========================================================+
| Subjects               | 25 (26 scanned, 1 excluded)                            |
+------------------------+--------------------------------------------------------+
| Samples                | 175 (7 conditions × 25 subjects)                       |
+------------------------+--------------------------------------------------------+
| ROIs                   | 446 brain regions (4S456Parcels atlas)                 |
+------------------------+--------------------------------------------------------+
| Connectivity shape     | (446, 446) per sample                                  |
+------------------------+--------------------------------------------------------+
| Feature dimensionality | 99,235 unique connections                              |
+------------------------+--------------------------------------------------------+
| Label distribution     | 86% conscious, 14% unconscious                         |
+------------------------+--------------------------------------------------------+
| Cross-validation       | 25-fold LOSO-CV                                        |
+------------------------+--------------------------------------------------------+
| Motion threshold       | FD < 0.8mm (timepoint censoring)                       |
+------------------------+--------------------------------------------------------+
| Temporal resolution    | TR = 3.0 seconds                                       |
+------------------------+--------------------------------------------------------+
| Spatial resolution     | 446 ROIs (2mm voxels before parcellation)              |
+------------------------+--------------------------------------------------------+
| Dataset size           | ~2GB (preprocessed derivatives only)                   |
+------------------------+--------------------------------------------------------+


Further Reading
===============

**Original Research Papers:**

1. Huang et al. (2018). *Scientific Reports* - Covert consciousness discovery
   [`DOI: 10.1038/s41598-018-31436-z <https://doi.org/10.1038/s41598-018-31436-z>`_]

2. Huang et al. (2021). *Cell Reports* - Anterior insula gating mechanism
   [`DOI: 10.1016/j.celrep.2021.109081 <https://doi.org/10.1016/j.celrep.2021.109081>`_]

3. Huang et al. (2021). *NeuroImage* - Asymmetric loss/recovery dynamics
   [`DOI: 10.1016/j.neuroimage.2021.118042 <https://doi.org/10.1016/j.neuroimage.2021.118042>`_]

**BIDS and Preprocessing:**

- BIDS specification: https://bids.neuroimaging.io/
- fMRIPrep documentation: https://fmriprep.org/
- XCP-D documentation: https://xcp-d.readthedocs.io/

**Brain Atlases:**

- Gordon et al. (2016). Cerebral Cortex. [`DOI: 10.1093/cercor/bhu239 <https://doi.org/10.1093/cercor/bhu239>`_]
- Schaefer et al. (2018). Cerebral Cortex. [`DOI: 10.1093/cercor/bhx179 <https://doi.org/10.1093/cercor/bhx179>`_]
- Glasser et al. (2016). Nature. [`DOI: 10.1038/nature18933 <https://doi.org/10.1038/nature18933>`_]
