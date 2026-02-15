"""Integration test: run the training pipeline on synthetic data.

Verifies the full pipeline works end-to-end without needing
the real dataset. Uses small synthetic matrices to keep it fast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from features import extract_all_features
from config import CONSCIOUS_CONDITIONS


N_SUBJECTS = 3
N_CONDITIONS = 7
N_ROIS = 20  # small for speed (real pipeline uses 446)


def make_synthetic_connectivity(n_rois, conscious=True):
    """Generate a synthetic connectivity matrix.

    Conscious matrices have slightly higher mean connectivity
    to simulate the real data pattern.
    """
    rng = np.random.RandomState(42 if conscious else 99)
    base = 0.3 if conscious else 0.1
    m = rng.randn(n_rois, n_rois) * 0.2 + base
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return np.clip(m, -1, 1)


def test_full_pipeline():
    """Run the complete training pipeline on synthetic data."""

    # Step 1: Generate synthetic data (mimics load_subject_all_conditions)
    all_features = []
    all_connectivity = []
    subjects = [f"sub-{i:02d}" for i in range(N_SUBJECTS)]

    for subject in subjects:
        for cond_idx in range(N_CONDITIONS):
            conscious = cond_idx in CONSCIOUS_CONDITIONS
            conn = make_synthetic_connectivity(N_ROIS, conscious=conscious)

            features = extract_all_features(conn)
            all_connectivity.append(features['connectivity'])

            features['subject'] = subject
            features['condition'] = cond_idx
            features['label'] = 1 if conscious else 0
            all_features.append(features)

    assert len(all_features) == N_SUBJECTS * N_CONDITIONS

    # Step 2: Feature engineering
    feature_names = [
        k for k in all_features[0].keys()
        if k not in ['subject', 'condition', 'label', 'connectivity']
    ]
    X_basic = np.array([[f[n] for n in feature_names] for f in all_features])
    subject_ids = np.array([f['subject'] for f in all_features])
    y = np.array([f['label'] for f in all_features])

    X_deviations = np.zeros_like(X_basic)
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        conscious_mask = y[mask] == 1
        if conscious_mask.sum() > 0:
            baseline = X_basic[mask][conscious_mask].mean(axis=0)
            X_deviations[mask] = X_basic[mask] - baseline

    X_engineered = np.hstack([X_basic, X_deviations])

    # Step 3: PCA on connectivity
    X_conn = np.array(all_connectivity)
    imputer = SimpleImputer(strategy='median')
    X_conn_clean = imputer.fit_transform(X_conn)

    n_components = min(5, X_conn_clean.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_conn_pca = pca.fit_transform(X_conn_clean)

    # Step 4: Combine features
    X_eng_clean = imputer.fit_transform(X_engineered)
    X_combined = np.hstack([X_eng_clean, X_conn_pca])

    assert X_combined.shape[0] == N_SUBJECTS * N_CONDITIONS
    assert not np.any(np.isnan(X_combined))

    # Step 5: LOSO-CV with XGBoost + SMOTE
    all_preds = []
    all_labels = []

    for test_subject in subjects:
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask

        X_train, y_train = X_combined[train_mask], y[train_mask]
        X_test, y_test = X_combined[test_mask], y[test_mask]

        # SMOTE
        n_minority = np.sum(y_train == 0)
        if n_minority >= 2:
            k = min(5, n_minority - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        clf = xgb.XGBClassifier(
            n_estimators=10, max_depth=3,
            random_state=42, eval_metric='logloss'
        )
        clf.fit(X_train_s, y_train, verbose=0)

        preds = clf.predict(X_test_s)
        all_preds.extend(preds)
        all_labels.extend(y_test)

    # Step 6: Verify outputs are valid
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    assert len(all_preds) == N_SUBJECTS * N_CONDITIONS
    assert set(all_preds).issubset({0, 1})
    assert set(all_labels).issubset({0, 1})

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    assert 0.0 <= bal_acc <= 1.0
