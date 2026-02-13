"""
Data loading functions - matches the paper's preprocessing exactly.

From paper's main.m:
  1. Load timeseries TSV (446 ROIs)
  2. Load motion TSV
  3. Filter timepoints where FD < 0.8
  4. Compute Pearson correlation → connectivity matrix
  5. Segment by LOR/ROR timing to create 7 conditions per subject
"""

import numpy as np
import pandas as pd
from typing import Tuple

from config import (
    XCP_DIR, N_ROIS, FD_COLUMN, FD_THRESHOLD,
    LOR_TIME, ROR_TIME, TRANSITION_BUFFER, SPECIAL_SUBJECTS,
    SUBJECTS
)


def load_timeseries(subject: str, task: str, run: int) -> pd.DataFrame:
    """Load XCP-D timeseries TSV file."""
    func_dir = XCP_DIR / subject / "func"
    filename = (
        f"{subject}_task-{task}_run-{run}_"
        f"space-MNI152NLin2009cAsym_seg-4S456Parcels_stat-mean_timeseries.tsv"
    )
    path = func_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing timeseries: {path}")

    # Load and take first 446 ROIs (paper uses 446 of 456)
    df = pd.read_csv(path, sep='\t')
    return df.iloc[:, :N_ROIS]


def load_motion(subject: str, task: str, run: int) -> pd.DataFrame:
    """Load motion parameters (framewise displacement)."""
    func_dir = XCP_DIR / subject / "func"
    filename = f"{subject}_task-{task}_run-{run}_motion.tsv"
    path = func_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing motion file: {path}")

    return pd.read_csv(path, sep='\t')


def filter_by_motion(
    timeseries: pd.DataFrame,
    motion: pd.DataFrame
) -> pd.DataFrame:
    """
    Keep only timepoints with FD < 0.8 (paper's criterion).

    Args:
        timeseries: (n_timepoints, 446) ROI time-series
        motion: Motion parameters with FD in column 8 (0-indexed col 7)

    Returns:
        Filtered timeseries (n_good_timepoints, 446)
    """
    fd = motion.iloc[:, FD_COLUMN].values
    good_mask = fd < FD_THRESHOLD
    return timeseries[good_mask]


def compute_connectivity(timeseries: pd.DataFrame) -> np.ndarray:
    """
    Compute 446×446 Pearson correlation matrix.

    Returns:
        Connectivity matrix (446, 446) with diagonal set to 0
    """
    if len(timeseries) == 0:
        return np.full((N_ROIS, N_ROIS), np.nan)

    conn = np.corrcoef(timeseries.T)
    np.fill_diagonal(conn, 0)  # paper sets diagonal to 0
    return conn


def load_condition_0(subject: str) -> np.ndarray:
    """Condition 0: rest_run-1 (baseline resting state)."""
    ts = load_timeseries(subject, "rest", 1)
    motion = load_motion(subject, "rest", 1)
    ts_filtered = filter_by_motion(ts, motion)
    return compute_connectivity(ts_filtered)


def load_condition_1(subject: str) -> np.ndarray:
    """Condition 1: imagery_run-1 (awake, pre-sedation)."""
    ts = load_timeseries(subject, "imagery", 1)
    motion = load_motion(subject, "imagery", 1)
    ts_filtered = filter_by_motion(ts, motion)
    return compute_connectivity(ts_filtered)


def load_condition_2(subject: str) -> np.ndarray:
    """Condition 2: imagery_run-2 PRE-LOR (before unconsciousness)."""
    lor_time = LOR_TIME[subject]

    ts = load_timeseries(subject, "imagery", 2)
    motion = load_motion(subject, "imagery", 2)

    # Take timepoints BEFORE lor_time
    ts_pre = ts.iloc[:lor_time]
    motion_pre = motion.iloc[:lor_time]

    ts_filtered = filter_by_motion(ts_pre, motion_pre)
    return compute_connectivity(ts_filtered)


def load_condition_3(subject: str) -> np.ndarray:
    """
    Condition 3: imagery LOR period (during unconsciousness).

    From paper: concatenate run-2 (from lor+375 to end)
    + run-3 (from start to ror-375)
    Skip 375 TRs around transitions to avoid mixed states.

    Special case (sub-29): no ROR segment, use all of run-3.
    """
    lor_time = LOR_TIME[subject]
    ror_time = ROR_TIME[subject]

    # Load both runs
    ts2 = load_timeseries(subject, "imagery", 2)
    motion2 = load_motion(subject, "imagery", 2)
    ts3 = load_timeseries(subject, "imagery", 3)
    motion3 = load_motion(subject, "imagery", 3)

    if subject not in SPECIAL_SUBJECTS:
        # Normal case: concatenate segments
        # run-2: from (lor + 375) to end
        ts2_lor = ts2.iloc[lor_time + TRANSITION_BUFFER:]
        motion2_lor = motion2.iloc[lor_time + TRANSITION_BUFFER:]

        # run-3: from start to (ror - 375)
        ts3_lor = ts3.iloc[:ror_time - TRANSITION_BUFFER]
        motion3_lor = motion3.iloc[:ror_time - TRANSITION_BUFFER]

        # Concatenate
        ts_combined = pd.concat([ts2_lor, ts3_lor], ignore_index=True)
        motion_combined = pd.concat(
            [motion2_lor, motion3_lor], ignore_index=True
        )
    else:
        # sub-29: all of run-3 is LOR
        ts_combined = pd.concat([
            ts2.iloc[lor_time + TRANSITION_BUFFER:],
            ts3
        ], ignore_index=True)
        motion_combined = pd.concat([
            motion2.iloc[lor_time + TRANSITION_BUFFER:],
            motion3
        ], ignore_index=True)

    ts_filtered = filter_by_motion(ts_combined, motion_combined)
    return compute_connectivity(ts_filtered)


def load_condition_4(subject: str) -> np.ndarray:
    """
    Condition 4: imagery POST-ROR (after regaining consciousness).

    From paper: run-3 from (ror + 1) to end.
    """
    if subject in SPECIAL_SUBJECTS:
        # sub-29 has no post-ROR
        return np.full((N_ROIS, N_ROIS), np.nan)

    ror_time = ROR_TIME[subject]

    ts = load_timeseries(subject, "imagery", 3)
    motion = load_motion(subject, "imagery", 3)

    ts_post = ts.iloc[ror_time + 1:]
    motion_post = motion.iloc[ror_time + 1:]

    if len(ts_post) == 0:
        return np.full((N_ROIS, N_ROIS), np.nan)

    ts_filtered = filter_by_motion(ts_post, motion_post)
    return compute_connectivity(ts_filtered)


def load_condition_5(subject: str) -> np.ndarray:
    """Condition 5: imagery_run-4 (recovery)."""
    try:
        ts = load_timeseries(subject, "imagery", 4)
        motion = load_motion(subject, "imagery", 4)
        ts_filtered = filter_by_motion(ts, motion)
        return compute_connectivity(ts_filtered)
    except FileNotFoundError:
        return np.full((N_ROIS, N_ROIS), np.nan)


def load_condition_6(subject: str) -> np.ndarray:
    """Condition 6: rest_run-2 (recovery resting state)."""
    try:
        ts = load_timeseries(subject, "rest", 2)
        motion = load_motion(subject, "rest", 2)
        ts_filtered = filter_by_motion(ts, motion)
        return compute_connectivity(ts_filtered)
    except FileNotFoundError:
        return np.full((N_ROIS, N_ROIS), np.nan)


def load_subject_all_conditions(subject: str) -> np.ndarray:
    """
    Load all 7 connectivity matrices for one subject.

    Returns:
        Array of shape (7, 446, 446) with connectivity matrices
        NaN-filled if data missing
    """
    fc = np.zeros((7, N_ROIS, N_ROIS))

    loaders = [
        load_condition_0, load_condition_1, load_condition_2,
        load_condition_3, load_condition_4, load_condition_5,
        load_condition_6
    ]

    for i, loader in enumerate(loaders):
        try:
            fc[i] = loader(subject)
        except Exception as e:
            print(f"  Warning: {subject} condition {i} failed: {e}")
            fc[i] = np.full((N_ROIS, N_ROIS), np.nan)

    return fc


def load_all_subjects() -> Tuple[np.ndarray, list]:
    """
    Load connectivity matrices for all 25 subjects.

    Returns:
        fc: (25, 7, 446, 446) connectivity matrices
        subjects: List of subject IDs
    """
    n_subjects = len(SUBJECTS)
    fc = np.zeros((n_subjects, 7, N_ROIS, N_ROIS))

    for i, subject in enumerate(SUBJECTS):
        print(f"[{i+1}/{n_subjects}] Loading {subject}...")
        fc[i] = load_subject_all_conditions(subject)

    return fc, SUBJECTS
