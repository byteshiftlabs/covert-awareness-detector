"""
Configuration for the consciousness detection pipeline.

Derived from the reference MATLAB code (main.m) by Jang et al.:
  - 446 ROIs from 4S456Parcels atlas (first 446 of 456; reason for exclusion undocumented)
  - xcp_d_without_GSR_bandpass preprocessing
  - 7 conditions: Wakeful Baseline, Imagery 1, PreLOR, LOR, Imagery 3 after ROR, Recovery Baseline, Rest 2
  - Motion censoring: FD column (col 8) < 0.8
  
Reference: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "openneuro" / "ds006623"
DERIVATIVES = DATA_ROOT / "derivatives"
XCP_DIR = DERIVATIVES / "xcp_d_without_GSR_bandpass_output"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Subject list (ordered as in paper) ──────────────────────────────────────
SUBJECTS = [
    "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-15", "sub-16",
    "sub-17", "sub-18", "sub-19", "sub-20", "sub-21", "sub-22",
    "sub-23", "sub-24", "sub-25", "sub-26", "sub-27", "sub-28",
    "sub-29",
]
N_SUBJECTS = len(SUBJECTS)  # 25 (sub-30 has no timing data in paper)

# ── LOR / ROR times (TR indices, from paper main.m) ────────────────────────
# LOR_TIME: TR index in run-2 at which the subject lost responsiveness
# ROR_TIME: TR index in run-3 at which the subject regained responsiveness
LOR_TIME = {
    "sub-02": 1160, "sub-03": 1385, "sub-04": 1573, "sub-05": 1010,
    "sub-06":  898, "sub-07": 1385, "sub-11": 1085, "sub-12": 1310,
    "sub-13": 1573, "sub-14":  898, "sub-15":  485, "sub-16": 2248,
    "sub-17": 1010, "sub-18": 1573, "sub-19":  898, "sub-20": 1985,
    "sub-21": 1310, "sub-22": 1310, "sub-23": 1310, "sub-24":  635,
    "sub-25": 1573, "sub-26": 1010, "sub-27":  485, "sub-28": 1385,
    "sub-29": 1835,
}

ROR_TIME = {
    "sub-02":  673, "sub-03":  410, "sub-04":  935, "sub-05":  673,
    "sub-06":  935, "sub-07": 1348, "sub-11":  673, "sub-12": 1535,
    "sub-13": 1460, "sub-14": 2270, "sub-15": 2135, "sub-16": 1760,
    "sub-17": 1535, "sub-18": 2270, "sub-19": 1348, "sub-20": 2023,
    "sub-21": 1160, "sub-22": 2270, "sub-23": 2023, "sub-24": 2270,
    "sub-25": 2023, "sub-26": 1348, "sub-27": 1760, "sub-28": 2270,
    "sub-29": 2270,
}

# sub-29 is special: no data after ROR in Imagery 3 (paper uses all of run-3 for LOR)
SPECIAL_SUBJECTS = {"sub-29"}

# ── Scan / atlas parameters ────────────────────────────────────────────────
N_ROIS = 446               # first 446 of 456 from 4S456Parcels (per reference MATLAB code)
ATLAS = "4S456Parcels"
FD_THRESHOLD = 0.8          # framewise displacement cutoff
FD_COLUMN = 7               # 0-indexed (column 8 in MATLAB 1-indexed)
TR = 3.0                    # repetition time in seconds
RUN2_TOTAL_TRS = 2270       # total TRs in imagery run-2
TRANSITION_BUFFER = 375     # TRs to skip around LOR/ROR transitions

# ── 7 conditions (FC matrices) per subject ──────────────────────────────────
CONDITIONS = {
    0: "rest_run-1",         # Wakeful Baseline (resting state)
    1: "imagery_run-1",      # Imagery 1 (fully awake, pre-sedation)
    2: "imagery_preLOR",     # Imagery 2 before loss of responsiveness
    3: "imagery_LOR",        # Imagery 2-3 during LOR (unconscious)
    4: "imagery_afterROR",   # Imagery 3 after return of responsiveness
    5: "imagery_run-4",      # Recovery Baseline (Imagery 4)
    6: "rest_run-2",         # Rest 2
}

CONSCIOUS_CONDITIONS = [0, 1, 2, 4, 5, 6]
UNCONSCIOUS_CONDITIONS = [3]
N_CONDITIONS = len(CONDITIONS)       # 7 conditions per subject

# ── ML parameters ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5  # Default probability cutoff for classification

# ── ISD calculation parameters ──────────────────────────────────────────────
ISD_THRESHOLD_COUNT = 50             # Number of threshold values for multilevel metrics
ISD_THRESHOLD_LOG_MIN = -3           # log10 of minimum threshold
ISD_THRESHOLD_LOG_MAX = 0            # log10 of maximum threshold (10^0 = 1)
NUMERICAL_EPSILON = 1e-10            # Small value to avoid division by zero

# ── XGBoost hyperparameters ─────────────────────────────────────────────────
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8

# ── PCA and threshold optimization ──────────────────────────────────────────
PCA_N_COMPONENTS = 50                # Number of PCA components for connectivity
THRESHOLD_SEARCH_MIN = 0.1           # Min probability threshold to search
THRESHOLD_SEARCH_MAX = 0.95          # Max probability threshold to search
THRESHOLD_SEARCH_STEP = 0.05         # Step size for threshold search

# ── Validation parameters ───────────────────────────────────────────────────
HOLDOUT_TEST_SUBJECTS = 5            # Number of subjects held out for validation
CV_STABILITY_SUBJECTS = 10           # Number of subjects for CV stability check
ENGINEERED_FEATURES_COUNT = 68       # Basic (34) + deviation (34) features
HOLDOUT_ACCURACY_THRESHOLD = 0.65    # Min balanced accuracy for holdout pass
PCA_IMPORTANCE_THRESHOLD = 0.25      # Min PCA feature importance for pass
CV_VARIANCE_THRESHOLD = 0.30         # Max coefficient of variation for pass
PERMUTATION_MARGIN = 0.15            # Min improvement over permuted for pass
