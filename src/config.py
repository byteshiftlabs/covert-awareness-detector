"""
Configuration for the consciousness detection pipeline.

Derived directly from the paper's MATLAB code (main.m):
  - 446 ROIs from 4S456Parcels atlas
  - xcp_d_without_GSR_bandpass preprocessing
  - 7 conditions: rest1, imagery1(awake), preLOR, LOR, postROR, imagery4, rest2
  - Motion censoring: FD column (col 8) < 0.8
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

# sub-29 is special: no post-ROR segment (paper uses all of run-3 for LOR)
SPECIAL_SUBJECTS = {"sub-29"}

# ── Scan / atlas parameters ────────────────────────────────────────────────
N_ROIS = 446               # first 446 of 456 parcels used by paper
ATLAS = "4S456Parcels"
FD_THRESHOLD = 0.8          # framewise displacement cutoff
FD_COLUMN = 7               # 0-indexed (column 8 in MATLAB 1-indexed)
TR = 3.0                    # repetition time in seconds
RUN2_TOTAL_TRS = 2270       # total TRs in imagery run-2
TRANSITION_BUFFER = 375     # TRs to skip around LOR/ROR transitions

# ── 7 conditions (FC matrices) per subject ──────────────────────────────────
CONDITIONS = {
    0: "rest_run-1",         # resting state baseline
    1: "imagery_awake",      # imagery run-1 (fully awake, pre-sedation)
    2: "imagery_preLOR",     # imagery run-2 before loss of responsiveness
    3: "imagery_LOR",        # imagery runs 2–3 during LOR (unconscious)
    4: "imagery_postROR",    # imagery run-3 after return of responsiveness
    5: "imagery_run-4",      # imagery run-4 (recovery)
    6: "rest_run-2",         # resting state recovery
}

CONSCIOUS_CONDITIONS = [0, 1, 2, 4, 5, 6]
UNCONSCIOUS_CONDITIONS = [3]

# ── ML parameters ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
