"""
Feature extraction from connectivity matrices.

Implements the paper's ISD (Integration-Segregation Difference) calculation:
  1. Regress out principal eigenvector to remove global signal
  2. Compute multilevel efficiency (integration)
  3. Compute multilevel clustering (segregation)
  4. ISD = efficiency - clustering
"""

import numpy as np
from typing import Tuple
from scipy.stats import skew, kurtosis


def regress_principal_eigenvector(fc: np.ndarray) -> np.ndarray:
    """
    Remove global signal by regressing out principal eigenvector.

    From paper's ISD_calculation.m:
        [V, D] = eig(FC);
        lambda1 = max(diag(D));
        u1 = V(:, idx_of_lambda1);
        FC_regressed = max(0, FC - lambda1 * (u1 * u1'));

    Args:
        fc: (n_rois, n_rois) connectivity matrix

    Returns:
        fc_regressed: With global signal removed, negative values set to 0
    """
    # Remove NaN rows/cols
    valid = ~np.isnan(fc[:, 0])
    fc_clean = fc[valid][:, valid]

    if fc_clean.shape[0] == 0:
        return fc

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eig(fc_clean)
    idx = np.argmax(eigvals)
    lambda1 = eigvals[idx]
    u1 = eigvecs[:, idx].real

    # Regress out
    fc_regressed = fc_clean - lambda1 * np.outer(u1, u1)

    # Clip to non-negative (paper does max(0, ...))
    fc_regressed = np.maximum(0, fc_regressed)

    # Put back into original shape
    result = np.full_like(fc, np.nan)
    result[np.ix_(valid, valid)] = fc_regressed

    return result


def multilevel_efficiency(fc: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute multilevel efficiency (integration measure).

    From paper's multilevel_efficiency.m:
        For each threshold T:
            Binary graph = (FC > T)
            Compute shortest path distances
            Efficiency = mean of inverse distances
        Integrate over thresholds using trapezoidal rule

    Args:
        fc: (n_rois, n_rois) connectivity matrix
        thresholds: Array of threshold values (e.g., logspace(-3, 0, 50))

    Returns:
        Integrated efficiency (scalar)
    """
    valid = ~np.isnan(fc[:, 0])
    fc_clean = fc[valid][:, valid]

    if fc_clean.shape[0] < 2:
        return np.nan

    # Zero diagonal
    np.fill_diagonal(fc_clean, 0)

    ml_efficiency = []

    for thresh in thresholds:
        # Binary graph
        binary = (fc_clean > thresh).astype(float)

        # Compute distances (shortest paths)
        # Simplified: use 1/connectivity as distance
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.where(binary > 0, 1.0 / (fc_clean + 1e-10), np.inf)
            np.fill_diagonal(dist, np.nan)

        # Efficiency = mean of 1/distance
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = 1.0 / dist
            eff[np.isinf(eff)] = np.nan

        ml_efficiency.append(np.nanmean(eff))

    # Integrate using trapezoidal rule
    return np.trapz(ml_efficiency, thresholds)


def multilevel_clustering(fc: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute multilevel clustering coefficient (segregation measure).

    From paper's multilevel_clustering.m:
        For each threshold T:
            Binary graph = (FC_regressed > T)
            Clustering = clustering_coef_bu(graph)
        Integrate over thresholds

    Args:
        fc: (n_rois, n_rois) connectivity matrix (should be regressed)
        thresholds: Array of threshold values

    Returns:
        Integrated clustering coefficient
    """
    valid = ~np.isnan(fc[:, 0])
    fc_clean = fc[valid][:, valid]

    if fc_clean.shape[0] < 2:
        return np.nan

    np.fill_diagonal(fc_clean, 0)

    ml_clustering = []

    for thresh in thresholds:
        # Binary graph
        binary = (fc_clean > thresh).astype(float)

        # Clustering coefficient (simplified implementation)
        # Full implementation requires BCT toolbox's clustering_coef_bu
        k = binary.sum(axis=1)  # degree

        # Triangles: A^2 .* A
        A2 = binary @ binary
        triangles = (A2 * binary).sum(axis=1)

        # Clustering = triangles / (k * (k-1))
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = k * (k - 1)
            clust = np.where(denom > 0, triangles / denom, 0)

        ml_clustering.append(np.nanmean(clust))

    return np.trapz(ml_clustering, thresholds)


def compute_isd(
    fc: np.ndarray,
    thresholds: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Compute Integration-Segregation Difference (ISD).

    Paper's key metric for consciousness:
        ISD = Efficiency - Clustering

    Higher ISD indicates more integrated (conscious) brain states.
    LOR states show significantly lower ISD (p < 0.05).

    Args:
        fc: (446, 446) connectivity matrix
        thresholds: Threshold range (default: logspace(-3, 0, 50))

    Returns:
        (isd, efficiency, clustering) tuple
    """
    if thresholds is None:
        thresholds = np.logspace(-3, 0, 50)

    # Efficiency on original FC
    fc_orig = fc.copy()
    np.fill_diagonal(fc_orig, 0)
    fc_orig = np.maximum(0, fc_orig)
    efficiency = multilevel_efficiency(fc_orig, thresholds)

    # Clustering on regressed FC
    fc_regressed = regress_principal_eigenvector(fc)
    clustering = multilevel_clustering(fc_regressed, thresholds)

    isd = efficiency - clustering

    return isd, efficiency, clustering


def extract_connectivity_features(fc: np.ndarray) -> np.ndarray:
    """
    Extract upper-triangle connectivity values as feature vector.

    Args:
        fc: (446, 446) connectivity matrix

    Returns:
        Feature vector of length 446*445/2 = 99,235
    """
    triu_idx = np.triu_indices(fc.shape[0], k=1)
    return fc[triu_idx]


def extract_all_features(fc: np.ndarray) -> dict:
    """
    Extract comprehensive feature set from connectivity matrix.

    Args:
        fc: (446, 446) connectivity matrix

    Returns:
        Dictionary with:
            - ISD metrics: isd, efficiency, clustering
            - Graph metrics: degree, strength, density
            - Statistical features: mean, std, skewness, kurtosis, percentiles
            - Connectivity: upper-triangle (99,235)
              optional, high-dimensional
    """
    # Clean NaNs/Infs
    fc_clean = fc.copy()
    fc_clean[np.isnan(fc_clean)] = 0
    fc_clean[np.isinf(fc_clean)] = 0

    # ISD metrics (paper's key features)
    isd, efficiency, clustering = compute_isd(fc_clean)

    # Graph metrics
    fc_abs = np.abs(fc_clean)
    threshold = np.median(fc_abs[fc_abs > 0]) if np.any(fc_abs > 0) else 0
    binary_graph = (fc_abs > threshold).astype(float)
    np.fill_diagonal(binary_graph, 0)
    degrees = binary_graph.sum(axis=1)
    strengths = fc_abs.sum(axis=1)

    # Connectivity values for statistics
    conn_values = extract_connectivity_features(fc_clean)

    return {
        # Paper's key metrics
        'isd': isd,
        'efficiency': efficiency,
        'clustering': clustering,

        # Graph topology
        'mean_degree': np.mean(degrees),
        'std_degree': np.std(degrees),
        'mean_strength': np.mean(strengths),
        'std_strength': np.std(strengths),
        'density': degrees.sum() / (len(fc) * (len(fc) - 1)),

        # Statistical features
        'mean_conn': np.mean(conn_values),
        'std_conn': np.std(conn_values),
        'skew_conn': skew(conn_values),
        'kurtosis_conn': kurtosis(conn_values),
        'q25_conn': np.percentile(conn_values, 25),
        'median_conn': np.median(conn_values),
        'q75_conn': np.percentile(conn_values, 75),
        'max_conn': np.max(conn_values),
        'min_conn': np.min(conn_values),

        # Full connectivity (99,235 dims) - use sparingly
        'connectivity': conn_values,
    }
