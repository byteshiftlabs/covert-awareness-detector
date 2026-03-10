"""
Data loading functions - matches the paper's preprocessing exactly.

From paper's main.m:
  1. Load timeseries TSV (446 ROIs)
  2. Load motion TSV
  3. Filter timepoints where FD < 0.8
  4. Compute Pearson correlation → connectivity matrix
  5. Segment by LOR/ROR timing to create 7 conditions per subject
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from config import (
    XCP_DIR, N_ROIS, FD_COLUMN, FD_THRESHOLD,
    LOR_TIME, ROR_TIME, TRANSITION_BUFFER, SPECIAL_SUBJECTS,
    SUBJECTS, N_CONDITIONS, CONSCIOUS_CONDITIONS, RANDOM_STATE,
    PCA_N_COMPONENTS
)


def load_timeseries(subject: str, task: str, run: int) -> pd.DataFrame:
    """Load XCP-D timeseries TSV file."""
    functional_directory = XCP_DIR / subject / "func"
    filename = (
        f"{subject}_task-{task}_run-{run}_"
        f"space-MNI152NLin2009cAsym_seg-4S456Parcels_stat-mean_timeseries.tsv"
    )
    path = functional_directory / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing timeseries: {path}")

    # Load and take first 446 ROIs (reference MATLAB uses columns 1:446 of 456)
    timeseries_data = pd.read_csv(path, sep='\t')
    return timeseries_data.iloc[:, :N_ROIS]


def load_motion(subject: str, task: str, run: int) -> pd.DataFrame:
    """Load motion parameters (framewise displacement)."""
    functional_directory = XCP_DIR / subject / "func"
    filename = f"{subject}_task-{task}_run-{run}_motion.tsv"
    path = functional_directory / filename

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
    framewise_displacement = motion.iloc[:, FD_COLUMN].values
    good_timepoints_mask = framewise_displacement < FD_THRESHOLD
    return timeseries[good_timepoints_mask]


def compute_connectivity(timeseries: pd.DataFrame) -> np.ndarray:
    """
    Compute 446×446 Pearson correlation matrix.

    Returns:
        Connectivity matrix (446, 446) with diagonal set to 0
    """
    if len(timeseries) == 0:
        return np.full((N_ROIS, N_ROIS), np.nan)

    connectivity = np.corrcoef(timeseries.T)
    np.fill_diagonal(connectivity, 0)  # paper sets diagonal to 0
    return connectivity


def load_condition_0(subject: str) -> np.ndarray:
    """Condition 0: rest_run-1 (baseline resting state)."""
    timeseries = load_timeseries(subject, "rest", 1)
    motion = load_motion(subject, "rest", 1)
    timeseries_filtered = filter_by_motion(timeseries, motion)
    return compute_connectivity(timeseries_filtered)


def load_condition_1(subject: str) -> np.ndarray:
    """Condition 1: imagery_run-1 (awake, pre-sedation)."""
    timeseries = load_timeseries(subject, "imagery", 1)
    motion = load_motion(subject, "imagery", 1)
    timeseries_filtered = filter_by_motion(timeseries, motion)
    return compute_connectivity(timeseries_filtered)


def load_condition_2(subject: str) -> np.ndarray:
    """Condition 2: imagery_run-2 PRE-LOR (before unconsciousness)."""
    lor_time = LOR_TIME[subject]

    timeseries = load_timeseries(subject, "imagery", 2)
    motion = load_motion(subject, "imagery", 2)

    # Take timepoints BEFORE lor_time
    timeseries_pre = timeseries.iloc[:lor_time]
    motion_pre = motion.iloc[:lor_time]

    timeseries_filtered = filter_by_motion(timeseries_pre, motion_pre)
    return compute_connectivity(timeseries_filtered)


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
    timeseries_run2 = load_timeseries(subject, "imagery", 2)
    motion_run2 = load_motion(subject, "imagery", 2)
    timeseries_run3 = load_timeseries(subject, "imagery", 3)
    motion_run3 = load_motion(subject, "imagery", 3)

    if subject not in SPECIAL_SUBJECTS:
        # Normal case: concatenate segments
        # run-2: from (lor + 375) to end
        timeseries_run2_lor = timeseries_run2.iloc[lor_time + TRANSITION_BUFFER:]
        motion_run2_lor = motion_run2.iloc[lor_time + TRANSITION_BUFFER:]

        # run-3: from start to (ror - 375)
        timeseries_run3_lor = timeseries_run3.iloc[:ror_time - TRANSITION_BUFFER]
        motion_run3_lor = motion_run3.iloc[:ror_time - TRANSITION_BUFFER]

        # Concatenate
        timeseries_combined = pd.concat([timeseries_run2_lor, timeseries_run3_lor], ignore_index=True)
        motion_combined = pd.concat(
            [motion_run2_lor, motion_run3_lor], ignore_index=True
        )
    else:
        # sub-29: all of run-3 is LOR
        timeseries_combined = pd.concat([
            timeseries_run2.iloc[lor_time + TRANSITION_BUFFER:],
            timeseries_run3
        ], ignore_index=True)
        motion_combined = pd.concat([
            motion_run2.iloc[lor_time + TRANSITION_BUFFER:],
            motion_run3
        ], ignore_index=True)

    timeseries_filtered = filter_by_motion(timeseries_combined, motion_combined)
    return compute_connectivity(timeseries_filtered)


def load_condition_4(subject: str) -> np.ndarray:
    """
    Condition 4: Imagery 3 after ROR (after regaining consciousness).

    From paper: run-3 from (ror + 1) to end.
    """
    if subject in SPECIAL_SUBJECTS:
        # sub-29 has no data after ROR
        return np.full((N_ROIS, N_ROIS), np.nan)

    ror_time = ROR_TIME[subject]

    timeseries = load_timeseries(subject, "imagery", 3)
    motion = load_motion(subject, "imagery", 3)

    timeseries_post = timeseries.iloc[ror_time + 1:]
    motion_post = motion.iloc[ror_time + 1:]

    if len(timeseries_post) == 0:
        return np.full((N_ROIS, N_ROIS), np.nan)

    timeseries_filtered = filter_by_motion(timeseries_post, motion_post)
    return compute_connectivity(timeseries_filtered)


def load_condition_5(subject: str) -> np.ndarray:
    """Condition 5: Recovery Baseline (Imagery 4)."""
    try:
        timeseries = load_timeseries(subject, "imagery", 4)
        motion = load_motion(subject, "imagery", 4)
        timeseries_filtered = filter_by_motion(timeseries, motion)
        return compute_connectivity(timeseries_filtered)
    except FileNotFoundError:
        return np.full((N_ROIS, N_ROIS), np.nan)


def load_condition_6(subject: str) -> np.ndarray:
    """Condition 6: Rest 2."""
    try:
        timeseries = load_timeseries(subject, "rest", 2)
        motion = load_motion(subject, "rest", 2)
        timeseries_filtered = filter_by_motion(timeseries, motion)
        return compute_connectivity(timeseries_filtered)
    except FileNotFoundError:
        return np.full((N_ROIS, N_ROIS), np.nan)


def load_subject_all_conditions(subject: str) -> np.ndarray:
    """
    Load all 7 connectivity matrices for one subject.

    Returns:
        Array of shape (7, 446, 446) with connectivity matrices
        NaN-filled if data missing
    """
    connectivity_matrices = np.zeros((7, N_ROIS, N_ROIS))

    loaders = [
        load_condition_0, load_condition_1, load_condition_2,
        load_condition_3, load_condition_4, load_condition_5,
        load_condition_6
    ]

    for i, loader in enumerate(loaders):
        try:
            connectivity_matrices[i] = loader(subject)
        except Exception as e:
            print(f"  Warning: {subject} condition {i} failed: {e}")
            connectivity_matrices[i] = np.full((N_ROIS, N_ROIS), np.nan)

    return connectivity_matrices


def load_all_subjects() -> Tuple[np.ndarray, list]:
    """
    Load connectivity matrices for all 25 subjects.

    Returns:
        connectivity_matrices: (25, 7, 446, 446) connectivity matrices
        subjects: List of subject IDs
    """
    n_subjects = len(SUBJECTS)
    connectivity_matrices = np.zeros((n_subjects, 7, N_ROIS, N_ROIS))

    for i, subject in enumerate(SUBJECTS):
        print(f"[{i+1}/{n_subjects}] Loading {subject}...")
        connectivity_matrices[i] = load_subject_all_conditions(subject)

    return connectivity_matrices, SUBJECTS


def prepare_data(
    progress_callback=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, SimpleImputer]:
    """
    Load all subjects and prepare feature matrix for ML training.

    This function performs:
    1. Load connectivity matrices for all subjects/conditions
    2. Extract features (ISD, graph metrics, connectivity)
    3. Compute per-subject deviation features
    4. Apply PCA to connectivity features
    5. Combine all features into final matrix

    Args:
        progress_callback: Optional callable(current, total, start_time, prefix)
                          for progress reporting

    Returns:
        X_combined: (n_samples, n_features) feature matrix
        y: (n_samples,) labels (1=conscious, 0=unconscious)
        subject_ids: (n_samples,) subject IDs for each sample
        pca: Fitted PCA transformer for connectivity features
        imputer: Fitted SimpleImputer for handling NaN values
    """
    # Import here to avoid circular import
    from features import extract_all_features

    all_features = []
    all_connectivity = []

    for subject in SUBJECTS:
        connectivity_matrices = load_subject_all_conditions(subject)
        for condition_index in range(N_CONDITIONS):
            connectivity_matrix = connectivity_matrices[condition_index]
            features = extract_all_features(connectivity_matrix)
            all_connectivity.append(features['connectivity'])
            features.update({
                'subject': subject,
                'condition': condition_index,
                'label': 1 if condition_index in CONSCIOUS_CONDITIONS else 0
            })
            all_features.append(features)

    # Extract basic features (excluding metadata and connectivity)
    feature_names = [
        k for k in all_features[0].keys()
        if k not in ['subject', 'condition', 'label', 'connectivity']
    ]
    X_basic = np.array([[f[name] for name in feature_names] for f in all_features])
    subject_ids = np.array([f['subject'] for f in all_features])
    y = np.array([f['label'] for f in all_features])

    # Per-subject deviation features (how each condition deviates from baseline)
    X_deviations = np.zeros_like(X_basic)
    for current_subject in np.unique(subject_ids):
        mask = subject_ids == current_subject
        data = X_basic[mask]
        conscious_mask = y[mask] == 1
        if conscious_mask.sum() > 0:
            baseline = data[conscious_mask].mean(axis=0)
        else:
            baseline = data.mean(axis=0)
        X_deviations[mask] = data - baseline

    # Handle NaN values
    imputer = SimpleImputer(strategy='median')

    # PCA on connectivity features (reduce 99K → PCA_N_COMPONENTS dimensions)
    X_connectivity_clean = imputer.fit_transform(np.array(all_connectivity))
    n_components = min(PCA_N_COMPONENTS, X_connectivity_clean.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_connectivity_pca = pca.fit_transform(X_connectivity_clean)

    # Combine: basic features + deviation features + PCA connectivity
    X_engineered = np.hstack([X_basic, X_deviations])
    X_engineered_clean = imputer.fit_transform(X_engineered)
    X_combined = np.hstack([X_engineered_clean, X_connectivity_pca])

    return X_combined, y, subject_ids, pca, imputer
