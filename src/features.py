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


def regress_principal_eigenvector(connectivity_matrix: np.ndarray) -> np.ndarray:
    """
    Remove global signal by regressing out principal eigenvector.

    From paper's ISD_calculation.m:
        [V, D] = eig(FC);
        lambda1 = max(diag(D));
        u1 = V(:, idx_of_lambda1);
        FC_regressed = max(0, FC - lambda1 * (u1 * u1'));

    Args:
        connectivity_matrix: (n_rois, n_rois) connectivity matrix

    Returns:
        connectivity_matrix_regressed: With global signal removed, negative values set to 0
    """
    # Remove NaN rows/cols
    valid = ~np.isnan(connectivity_matrix[:, 0])
    connectivity_matrix_clean = connectivity_matrix[valid][:, valid]

    if connectivity_matrix_clean.shape[0] == 0:
        return connectivity_matrix

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eig(connectivity_matrix_clean)
    idx = np.argmax(eigvals)
    lambda1 = eigvals[idx]
    u1 = eigvecs[:, idx].real

    # Regress out
    connectivity_matrix_regressed = connectivity_matrix_clean - lambda1 * np.outer(u1, u1)

    # Clip to non-negative (paper does max(0, ...))
    connectivity_matrix_regressed = np.maximum(0, connectivity_matrix_regressed)

    # Put back into original shape
    result = np.full_like(connectivity_matrix, np.nan)
    result[np.ix_(valid, valid)] = connectivity_matrix_regressed

    return result


def multilevel_efficiency(connectivity_matrix: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute multilevel efficiency (integration measure).

    From paper's multilevel_efficiency.m:
        For each threshold T:
            Binary graph = (FC > T)
            Compute shortest path distances
            Efficiency = mean of inverse distances
        Integrate over thresholds using trapezoidal rule

    Args:
        connectivity_matrix: (n_rois, n_rois) connectivity matrix
        thresholds: Array of threshold values (e.g., logspace(-3, 0, 50))

    Returns:
        Integrated efficiency (scalar)
    """
    valid = ~np.isnan(connectivity_matrix[:, 0])
    connectivity_matrix_clean = connectivity_matrix[valid][:, valid]

    if connectivity_matrix_clean.shape[0] < 2:
        return np.nan

    # Zero diagonal
    np.fill_diagonal(connectivity_matrix_clean, 0)

    ml_efficiency = []

    for threshold in thresholds:
        # Binary graph
        binary = (connectivity_matrix_clean > threshold).astype(float)

        # Compute distances (shortest paths)
        # Simplified: use 1/connectivity as distance
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.where(binary > 0, 1.0 / (connectivity_matrix_clean + 1e-10), np.inf)
            np.fill_diagonal(dist, np.nan)

        # Efficiency = mean of 1/distance
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = 1.0 / dist
            eff[np.isinf(eff)] = np.nan

        ml_efficiency.append(np.nanmean(eff))

    # Integrate using trapezoidal rule
    return np.trapz(ml_efficiency, thresholds)


def multilevel_clustering(connectivity_matrix: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute multilevel clustering coefficient (segregation measure).

    From paper's multilevel_clustering.m:
        For each threshold T:
            Binary graph = (FC_regressed > T)
            Clustering = clustering_coef_bu(graph)
        Integrate over thresholds

    Args:
        connectivity_matrix: (n_rois, n_rois) connectivity matrix (should be regressed)
        thresholds: Array of threshold values

    Returns:
        Integrated clustering coefficient
    """
    valid = ~np.isnan(connectivity_matrix[:, 0])
    connectivity_matrix_clean = connectivity_matrix[valid][:, valid]

    if connectivity_matrix_clean.shape[0] < 2:
        return np.nan

    np.fill_diagonal(connectivity_matrix_clean, 0)

    ml_clustering = []

    for threshold in thresholds:
        # Binary graph
        binary = (connectivity_matrix_clean > threshold).astype(float)

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
    connectivity_matrix: np.ndarray,
    thresholds: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Compute Integration-Segregation Difference (ISD).

    Paper's key metric for consciousness:
        ISD = Efficiency - Clustering

    Higher ISD indicates more integrated (conscious) brain states.
    LOR states show significantly lower ISD (p < 0.05).

    Args:
        connectivity_matrix: (446, 446) connectivity matrix
        thresholds: Threshold range (default: logspace(-3, 0, 50))

    Returns:
        (isd, efficiency, clustering) tuple
    """
    if thresholds is None:
        thresholds = np.logspace(-3, 0, 50)

    # Efficiency on original connectivity matrix
    connectivity_matrix_orig = connectivity_matrix.copy()
    np.fill_diagonal(connectivity_matrix_orig, 0)
    connectivity_matrix_orig = np.maximum(0, connectivity_matrix_orig)
    efficiency = multilevel_efficiency(connectivity_matrix_orig, thresholds)

    # Clustering on regressed connectivity matrix
    connectivity_matrix_regressed = regress_principal_eigenvector(connectivity_matrix)
    clustering = multilevel_clustering(connectivity_matrix_regressed, thresholds)

    isd = efficiency - clustering

    return isd, efficiency, clustering


def extract_connectivity_features(connectivity_matrix: np.ndarray) -> np.ndarray:
    """
    Extract upper-triangle connectivity values as feature vector.

    Args:
        connectivity_matrix: (446, 446) connectivity matrix

    Returns:
        Feature vector of length 446*445/2 = 99,235
    """
    triu_idx = np.triu_indices(connectivity_matrix.shape[0], k=1)
    return connectivity_matrix[triu_idx]


def extract_all_features(connectivity_matrix: np.ndarray) -> dict:
    """
    Extract comprehensive feature set from connectivity matrix.

    Args:
        connectivity_matrix: (446, 446) connectivity matrix

    Returns:
        Dictionary with:
            - ISD metrics: isd, efficiency, clustering
            - Graph metrics: degree, strength, density
            - Statistical features: mean, std, skewness, kurtosis, percentiles
            - Connectivity: upper-triangle (99,235)
              optional, high-dimensional
    """
    # Clean NaNs/Infs
    connectivity_matrix_clean = connectivity_matrix.copy()
    connectivity_matrix_clean[np.isnan(connectivity_matrix_clean)] = 0
    connectivity_matrix_clean[np.isinf(connectivity_matrix_clean)] = 0

    # ISD metrics (paper's key features)
    isd, efficiency, clustering = compute_isd(connectivity_matrix_clean)

    # Graph metrics
    connectivity_matrix_abs = np.abs(connectivity_matrix_clean)
    threshold = np.median(connectivity_matrix_abs[connectivity_matrix_abs > 0]) if np.any(connectivity_matrix_abs > 0) else 0
    binary_graph = (connectivity_matrix_abs > threshold).astype(float)
    np.fill_diagonal(binary_graph, 0)
    degrees = binary_graph.sum(axis=1)
    strengths = connectivity_matrix_abs.sum(axis=1)

    # Connectivity values for statistics
    conn_values = extract_connectivity_features(connectivity_matrix_clean)

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
        'density': degrees.sum() / (len(connectivity_matrix) * (len(connectivity_matrix) - 1)),

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
