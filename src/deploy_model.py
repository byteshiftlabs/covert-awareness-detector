#!/usr/bin/env python3
"""
Deploy trained model - predict on new subject data.

Usage:
    python deploy_model.py --subject sub-02
    python deploy_model.py --subject sub-02 --condition 3
"""

import argparse
import numpy as np
import pickle
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_loader import load_subject_all_conditions
from features import extract_all_features
from config import SUBJECTS, CONSCIOUS_CONDITIONS


def train_final_model():
    """Train final model on ALL data for deployment."""
    print("Training final model on all subjects...")

    # Load all data
    all_features = []
    all_connectivity_matrices = []

    for subject in SUBJECTS:
        all_connectivity_matrices = load_subject_all_conditions(subject)
        for cond_idx in range(7):
            connectivity_matrix = all_connectivity_matrices[cond_idx]
            features = extract_all_features(connectivity_matrix)
            all_connectivity_matrices.append(features['connectivity'])
            features['subject'] = subject
            features['condition'] = cond_idx
            features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0
            all_features.append(features)

    # Feature engineering
    feature_names_basic = [
        k for k in all_features[0].keys()
        if k not in [
            'subject', 'condition',
            'label', 'connectivity'
        ]
    ]
    X_basic = np.array(
        [[f[name] for name in feature_names_basic]
         for f in all_features]
    )
    subject_ids = np.array([f['subject'] for f in all_features])
    y = np.array([f['label'] for f in all_features])

    X_deviations = np.zeros((len(all_features), X_basic.shape[1]))
    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        subj_data = X_basic[subj_mask]
        conscious_mask = y[subj_mask] == 1
        if conscious_mask.sum() > 0:
            baseline = subj_data[conscious_mask].mean(axis=0)
            X_deviations[subj_mask] = subj_data - baseline
        else:
            X_deviations[subj_mask] = subj_data

    X_engineered = np.hstack([X_basic, X_deviations])

    # PCA on connectivity
    X_connectivity = np.array(all_connectivity_matrices)
    imputer = SimpleImputer(strategy='median')
    X_connectivity_clean = imputer.fit_transform(X_connectivity)

    n_components = min(
        50, X_connectivity_clean.shape[0] - 1
    )
    pca = PCA(n_components=n_components, random_state=42)
    X_connectivity_pca = pca.fit_transform(X_connectivity_clean)

    # Combine
    X_engineered_clean = imputer.fit_transform(X_engineered)
    X_combined = np.hstack([X_engineered_clean, X_connectivity_pca])

    # SMOTE
    n_minority = np.sum(y == 0)
    k_neighbors = min(5, n_minority - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_combined, y)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Train XGBoost
    scale_pos_weight = np.sum(y_balanced == 0) / np.sum(y_balanced == 1)
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    clf.fit(X_scaled, y_balanced, verbose=0)

    # Save model artifacts
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / "final_model.pkl", "wb") as f:
        pickle.dump({
            'classifier': clf,
            'scaler': scaler,
            'pca': pca,
            'imputer': imputer,
            'feature_names': feature_names_basic,
            'threshold': 0.85  # Optimized threshold
        }, f)

    print(f"✓ Model saved to {model_dir / 'final_model.pkl'}")
    return clf, scaler, pca, imputer


def predict_subject(subject, condition=None, model_path=None):
    """Predict consciousness state for a subject."""

    # Load model
    if model_path is None:
        model_path = (
            Path(__file__).parent.parent
            / "models" / "final_model.pkl"
        )

    if not model_path.exists():
        print("Model not found. Training new model...")
        clf, scaler, pca, imputer = train_final_model()
        threshold = 0.85
    else:
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
            clf = artifacts['classifier']
            scaler = artifacts['scaler']
            pca = artifacts['pca']
            imputer = artifacts['imputer']
            threshold = artifacts['threshold']

    # Load subject data
    print(f"\nPredicting for {subject}...")
    all_fc = load_subject_all_conditions(subject)

    conditions_to_predict = [condition] if condition is not None else range(7)

    for cond_idx in conditions_to_predict:
        connectivity_matrix = all_fc[cond_idx]
        features = extract_all_features(connectivity_matrix)

        # Extract features (same as training)
        feature_names = [
            k for k in features.keys()
            if k not in [
                'subject', 'condition',
                'label', 'connectivity'
            ]
        ]
        X_basic = np.array(
            [[features[name] for name in feature_names]]
        )

        # For deployment, use all-condition mean
        X_deviation = np.zeros_like(X_basic)
        X_engineered = np.hstack([X_basic, X_deviation])

        # PCA on connectivity
        conn_full = features['connectivity'].reshape(1, -1)
        conn_clean = imputer.transform(conn_full)
        conn_pca = pca.transform(conn_clean)

        # Combine
        X_eng_clean = imputer.transform(X_engineered)
        X_final = np.hstack([X_eng_clean, conn_pca])

        # Scale
        X_scaled = scaler.transform(X_final)

        # Predict
        proba = clf.predict_proba(X_scaled)[0, 1]
        pred = 1 if proba >= threshold else 0

        cond_names = [
            'rest1', 'imagery_awake', 'pre-LOR',
            'LOR', 'post-ROR', 'imagery4', 'rest2'
        ]
        state = "CONSCIOUS" if pred == 1 else "UNCONSCIOUS"
        confidence = proba if pred == 1 else (1 - proba)

        print(f"\nCondition {cond_idx} ({cond_names[cond_idx]}):")
        print(f"  Prediction: {state}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  P(conscious): {proba:.3f}")

        if cond_idx == 3:  # LOR
            print(
                "  Expected: UNCONSCIOUS"
                " (loss of responsiveness)"
            )
        elif cond_idx in CONSCIOUS_CONDITIONS:
            print("  Expected: CONSCIOUS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Consciousness detection model deployment'
        )
    )
    parser.add_argument(
        '--subject', type=str,
        default='sub-02', help='Subject ID'
    )
    parser.add_argument(
        '--condition', type=int, default=None,
        help='Condition (0-6), or all if not specified'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Train and save final model'
    )

    args = parser.parse_args()

    if args.train:
        train_final_model()
    else:
        predict_subject(args.subject, args.condition)


if __name__ == '__main__':
    main()
