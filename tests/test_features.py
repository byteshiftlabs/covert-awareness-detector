"""Unit tests for features.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from features import (
    regress_principal_eigenvector,
    multilevel_efficiency,
    multilevel_clustering,
    compute_isd,
    extract_connectivity_features,
    extract_all_features,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def identity_matrix():
    """10×10 identity matrix (no connectivity)."""
    return np.eye(10)


@pytest.fixture
def random_connectivity():
    """10×10 symmetric connectivity matrix with realistic values."""
    rng = np.random.RandomState(42)
    m = rng.randn(10, 10)
    m = (m + m.T) / 2  # make symmetric
    np.fill_diagonal(m, 1.0)
    return m


@pytest.fixture
def full_connectivity():
    """10×10 matrix where all regions are fully connected."""
    m = np.ones((10, 10))
    np.fill_diagonal(m, 0)
    return m


@pytest.fixture
def thresholds():
    """Default threshold range used by the pipeline."""
    return np.logspace(-3, 0, 20)


# ── regress_principal_eigenvector ───────────────────────────────────────────

def test_regress_output_shape(random_connectivity):
    """Output should have same shape as input."""
    result = regress_principal_eigenvector(random_connectivity)
    assert result.shape == random_connectivity.shape


def test_regress_non_negative(random_connectivity):
    """Paper clips values to >= 0 after regression."""
    result = regress_principal_eigenvector(random_connectivity)
    valid = ~np.isnan(result)
    assert np.all(result[valid] >= 0)


def test_regress_handles_nan():
    """Should handle NaN rows/columns gracefully."""
    m = np.ones((5, 5))
    m[0, :] = np.nan
    m[:, 0] = np.nan
    result = regress_principal_eigenvector(m)
    assert result.shape == (5, 5)
    assert np.all(np.isnan(result[0, :]))


# ── multilevel_efficiency ───────────────────────────────────────────────────

def test_efficiency_returns_scalar(random_connectivity, thresholds):
    """Efficiency should be a single float."""
    result = multilevel_efficiency(random_connectivity, thresholds)
    assert isinstance(result, (float, np.floating))


def test_efficiency_full_connectivity(full_connectivity, thresholds):
    """Fully connected graph should have high efficiency."""
    result = multilevel_efficiency(full_connectivity, thresholds)
    assert not np.isnan(result)
    assert result > 0


def test_efficiency_tiny_matrix(thresholds):
    """Should return NaN for matrix smaller than 2×2."""
    m = np.array([[1.0]])
    result = multilevel_efficiency(m, thresholds)
    assert np.isnan(result)


# ── multilevel_clustering ───────────────────────────────────────────────────

def test_clustering_returns_scalar(random_connectivity, thresholds):
    """Clustering should be a single float."""
    result = multilevel_clustering(random_connectivity, thresholds)
    assert isinstance(result, (float, np.floating))


def test_clustering_tiny_matrix(thresholds):
    """Should return NaN for matrix smaller than 2×2."""
    m = np.array([[1.0]])
    result = multilevel_clustering(m, thresholds)
    assert np.isnan(result)


# ── compute_isd ─────────────────────────────────────────────────────────────

def test_isd_returns_three_values(random_connectivity):
    """ISD should return (isd, efficiency, clustering) tuple."""
    result = compute_isd(random_connectivity)
    assert len(result) == 3


def test_isd_equals_efficiency_minus_clustering(random_connectivity):
    """ISD = efficiency - clustering by definition."""
    isd, efficiency, clustering = compute_isd(random_connectivity)
    assert np.isclose(isd, efficiency - clustering, atol=1e-10)


def test_isd_with_zeros():
    """All-zero matrix should not crash."""
    m = np.zeros((10, 10))
    isd, eff, clust = compute_isd(m)
    # Should complete without error, values may be NaN or 0
    assert isinstance(isd, (float, np.floating))


# ── extract_connectivity_features ───────────────────────────────────────────

def test_connectivity_features_length():
    """Upper triangle of n×n matrix has n*(n-1)/2 elements."""
    n = 10
    m = np.random.randn(n, n)
    result = extract_connectivity_features(m)
    expected = n * (n - 1) // 2
    assert len(result) == expected


def test_connectivity_features_446():
    """Should produce 99,235 features for 446 ROIs."""
    m = np.zeros((446, 446))
    result = extract_connectivity_features(m)
    assert len(result) == 446 * 445 // 2  # 99,235


# ── extract_all_features ───────────────────────────────────────────────────

def test_all_features_keys():
    """Should return all expected feature keys."""
    m = np.random.RandomState(42).randn(10, 10)
    m = (m + m.T) / 2
    features = extract_all_features(m)

    expected_keys = {
        'isd', 'efficiency', 'clustering',
        'mean_degree', 'std_degree', 'mean_strength', 'std_strength', 'density',
        'mean_conn', 'std_conn', 'skew_conn', 'kurtosis_conn',
        'q25_conn', 'median_conn', 'q75_conn', 'max_conn', 'min_conn',
        'connectivity',
    }
    assert expected_keys.issubset(features.keys())


def test_all_features_handles_nan_matrix():
    """Should not crash on matrix with NaN values."""
    m = np.full((10, 10), np.nan)
    features = extract_all_features(m)
    assert 'isd' in features


def test_all_features_connectivity_is_array():
    """Connectivity feature should be a numpy array."""
    m = np.random.RandomState(42).randn(10, 10)
    m = (m + m.T) / 2
    features = extract_all_features(m)
    assert isinstance(features['connectivity'], np.ndarray)
