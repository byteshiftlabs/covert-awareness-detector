"""Unit tests for data_loader.py.

These tests use synthetic data (no dataset download required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from data_loader import filter_by_motion, compute_connectivity


# ── filter_by_motion ────────────────────────────────────────────────────────

def test_filter_keeps_good_timepoints():
    """Should keep rows where FD < 0.8."""
    timeseries = pd.DataFrame(np.ones((5, 3)), columns=['r1', 'r2', 'r3'])
    # FD is in column 7 (0-indexed)
    motion_data = np.zeros((5, 8))
    motion_data[:, 7] = [0.1, 0.9, 0.2, 1.5, 0.3]  # 3 good, 2 bad
    motion = pd.DataFrame(motion_data)

    result = filter_by_motion(timeseries, motion)
    assert len(result) == 3


def test_filter_removes_all_bad():
    """If all timepoints exceed FD threshold, result should be empty."""
    timeseries = pd.DataFrame(np.ones((3, 2)))
    motion_data = np.zeros((3, 8))
    motion_data[:, 7] = [1.0, 1.2, 0.9]  # all >= 0.8
    motion = pd.DataFrame(motion_data)

    result = filter_by_motion(timeseries, motion)
    assert len(result) == 0


def test_filter_keeps_all_good():
    """If all timepoints are below threshold, keep everything."""
    timeseries = pd.DataFrame(np.ones((4, 2)))
    motion_data = np.zeros((4, 8))
    motion_data[:, 7] = [0.1, 0.2, 0.3, 0.4]
    motion = pd.DataFrame(motion_data)

    result = filter_by_motion(timeseries, motion)
    assert len(result) == 4


# ── compute_connectivity ───────────────────────────────────────────────────

def test_connectivity_shape():
    """Output should be (n_rois, n_rois)."""
    timeseries = pd.DataFrame(np.random.RandomState(42).randn(50, 10))
    conn = compute_connectivity(timeseries)
    assert conn.shape == (10, 10)


def test_connectivity_symmetric():
    """Correlation matrix should be symmetric."""
    timeseries = pd.DataFrame(np.random.RandomState(42).randn(50, 10))
    conn = compute_connectivity(timeseries)
    np.testing.assert_array_almost_equal(conn, conn.T)


def test_connectivity_diagonal_zero():
    """Paper sets diagonal to 0."""
    timeseries = pd.DataFrame(np.random.RandomState(42).randn(50, 10))
    conn = compute_connectivity(timeseries)
    np.testing.assert_array_equal(np.diag(conn), np.zeros(10))


def test_connectivity_values_in_range():
    """Pearson correlations should be in [-1, 1]."""
    timeseries = pd.DataFrame(np.random.RandomState(42).randn(50, 10))
    conn = compute_connectivity(timeseries)
    assert np.all(conn >= -1.0)
    assert np.all(conn <= 1.0)


def test_connectivity_empty_timeseries():
    """Empty timeseries should return NaN matrix with N_ROIS shape."""
    from config import N_ROIS
    timeseries = pd.DataFrame(np.zeros((0, N_ROIS)))
    conn = compute_connectivity(timeseries)
    assert conn.shape == (N_ROIS, N_ROIS)
    assert np.all(np.isnan(conn))
