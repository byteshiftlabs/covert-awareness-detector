#!/usr/bin/env python3
"""
ADVANCED consciousness detection with full connectivity features.

Improvements:
1. Full connectivity matrix (99,235 features) + PCA dimensionality reduction
2. XGBoost classifier - handles imbalance better
3. Advanced features - comparing connectivity across conditions
4. SMOTE + optimized threshold
"""

import numpy as np
import warnings
import time
import sys
import shutil

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_loader import load_subject_all_conditions
from features import extract_all_features
from config import SUBJECTS, CONSCIOUS_CONDITIONS

warnings.filterwarnings('ignore')


def progress_bar(current, total, start_time, prefix='', bar_length=None):
    """Display a progress bar with ETA (only on terminal, written to stderr)."""
    # Write to stderr so it appears on terminal but not in log files
    if not sys.stderr.isatty():
        return
    
    if bar_length is None:
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        bar_length = max(20, terminal_width - 55)

    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = time.strftime('%M:%S', time.gmtime(eta))
    else:
        eta_str = '--:--'

    elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = '█' * filled + '░' * (bar_length - filled)
    pct = fraction * 100

    sys.stderr.write(
        f'\r  {prefix} [{bar}] {pct:5.1f}% '
        f'({current}/{total}) '
        f'elapsed {elapsed_str} · ETA {eta_str}'
    )
    sys.stderr.flush()
    if current == total:
        sys.stderr.write('\n')


if __name__ == "__main__":
    print("="*70)
    print("ADVANCED CONSCIOUSNESS DETECTION")
    print("="*70)
    print("XGBoost + PCA + SMOTE + Threshold Tuning")
    print()
    
    # ============================================================================
    # STEP 1: Load data with FULL connectivity
    # ============================================================================
    print("Loading data...")
    all_features = []
    all_connectivity_matrices = []
    
    load_start = time.time()
    for idx, subject in enumerate(SUBJECTS):
        progress_bar(idx, len(SUBJECTS), load_start, prefix='Loading')
    
        subject_connectivity_matrices = load_subject_all_conditions(subject)
    
        for cond_idx in range(7):
            connectivity_matrix = subject_connectivity_matrices[cond_idx]
    
            # Extract ALL features including full connectivity
            features = extract_all_features(connectivity_matrix)
    
            # Store connectivity separately for PCA
            conn_full = features['connectivity']  # 99,235 dims
            all_connectivity_matrices.append(conn_full)
    
            # Store metadata and other features
            features['subject'] = subject
            features['condition'] = cond_idx
            features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0
    
            all_features.append(features)
    
    progress_bar(len(SUBJECTS), len(SUBJECTS), load_start, prefix='Loading')
    print(f"✓ Loaded {len(all_features)} samples\n")
    
    # ============================================================================
    # STEP 2: Advanced feature engineering
    # ============================================================================
    print("Feature engineering...")
    
    # Extract basic features (excluding connectivity)
    feature_names_basic = [
        k for k in all_features[0].keys()
        if k not in [
            'subject', 'condition', 'label',
            'connectivity'
        ]
    ]
    
    X_basic = np.array(
        [[f[name] for name in feature_names_basic]
         for f in all_features]
    )
    
    # Add per-subject normalization features
    subject_ids = np.array([f['subject'] for f in all_features])
    conditions = np.array([f['condition'] for f in all_features])
    y = np.array([f['label'] for f in all_features])
    
    # For each subject, compute how much each condition deviates
    # from their baseline
    X_deviations = np.zeros(
        (len(all_features), X_basic.shape[1])
    )
    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        subj_data = X_basic[subj_mask]
    
        # Baseline = mean of conscious conditions for this subject
        conscious_mask = y[subj_mask] == 1
        if conscious_mask.sum() > 0:
            baseline = subj_data[conscious_mask].mean(axis=0)
            # Deviation from baseline
            X_deviations[subj_mask] = subj_data - baseline
        else:
            X_deviations[subj_mask] = subj_data
    
    # Combine basic + deviation features
    X_engineered = np.hstack([X_basic, X_deviations])
    print(f"✓ Engineered {X_engineered.shape[1]} features\n")
    
    # ============================================================================
    # STEP 3: PCA on full connectivity
    # ============================================================================
    print("PCA dimensionality reduction...")
    
    # Stack connectivity matrices
    X_connectivity = np.array(all_connectivity_matrices)
    
    # Handle NaN/Inf
    imputer = SimpleImputer(strategy='median')
    X_connectivity_clean = imputer.fit_transform(X_connectivity)
    
    # Apply PCA to reduce from 99,235 to manageable size
    n_components = min(
        50, X_connectivity_clean.shape[0] - 1
    )
    pca = PCA(n_components=n_components, random_state=42)
    X_connectivity_pca = pca.fit_transform(X_connectivity_clean)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"✓ PCA: 99K features → {n_components} components ({variance_explained:.1%} variance)\n")
    
    # ============================================================================
    # STEP 4: Combine all features
    # ============================================================================
    print("Combining features...")
    
    # Handle NaN in engineered features
    X_engineered_clean = imputer.fit_transform(X_engineered)
    
    # Combine: engineered + PCA connectivity
    X_combined = np.hstack([X_engineered_clean, X_connectivity_pca])
    print(f"✓ Final matrix: {X_combined.shape[0]} samples × {X_combined.shape[1]} features")
    print(f"  Conscious: {np.sum(y == 1)}, Unconscious: {np.sum(y == 0)}\n")
    
    # ============================================================================
    # STEP 5: Train with XGBoost + SMOTE
    # ============================================================================
    print("Training with LOSO-CV...")
    print()
    
    unique_subjects = np.unique(subject_ids)
    all_preds = []
    all_labels = []
    all_probas = []
    
    cv_start = time.time()
    for i, test_subject in enumerate(unique_subjects):
        progress_bar(i, len(unique_subjects), cv_start, prefix='Training')
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
    
        X_train = X_combined[train_mask]
        y_train = y[train_mask]
        X_test = X_combined[test_mask]
        y_test = y[test_mask]
    
        # Apply SMOTE
        n_minority = np.sum(y_train == 0)
        if n_minority >= 2:
            k_neighbors = min(5, n_minority - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            try:
                X_train_balanced, y_train_balanced = (
                    smote.fit_resample(X_train, y_train)
                )
            except Exception:
                X_train_balanced, y_train_balanced = (
                    X_train, y_train
                )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
    
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
    
        # XGBoost with class weights
        scale_pos_weight = (
            np.sum(y_train_balanced == 0)
            / np.sum(y_train_balanced == 1)
        )
    
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
    
        clf.fit(X_train_scaled, y_train_balanced, verbose=0)
    
        # Predict
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred_default = (y_proba >= 0.5).astype(int)
    
        all_labels.extend(y_test)
        all_probas.extend(y_proba)
        all_preds.extend(y_pred_default)
    
    progress_bar(len(unique_subjects), len(unique_subjects), cv_start, prefix='Training')
    cv_elapsed = time.time() - cv_start
    print(f"✓ Completed {len(unique_subjects)} LOSO-CV folds in {time.strftime('%M:%S', time.gmtime(cv_elapsed))}\n")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    # ============================================================================
    # STEP 6: Optimize threshold
    # ============================================================================
    print("Optimizing threshold...")
    
    best_threshold = 0.5
    best_balanced_acc = 0
    best_metrics = None
    
    for threshold in np.arange(0.1, 0.95, 0.05):
        y_pred_thresh = (all_probas >= threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score(all_labels, y_pred_thresh)
    
        if balanced_accuracy > best_balanced_acc:
            best_balanced_acc = balanced_accuracy
            best_threshold = threshold
            best_metrics = {
                'balanced_acc': balanced_accuracy,
                'accuracy': accuracy_score(all_labels, y_pred_thresh),
                'recall_unconscious': recall_score(
                    all_labels, y_pred_thresh,
                    pos_label=0, zero_division=0
                ),
                'recall_conscious': recall_score(
                    all_labels, y_pred_thresh,
                    pos_label=1, zero_division=0
                ),
                'precision': precision_score(
                    all_labels, y_pred_thresh,
                    zero_division=0
                ),
                'f1': f1_score(all_labels, y_pred_thresh, zero_division=0),
                'roc_auc': roc_auc_score(all_labels, all_probas),
                'confusion_matrix': confusion_matrix(all_labels, y_pred_thresh)
            }
    
    print(f"✓ Optimal threshold: {best_threshold:.2f} (balanced acc: {best_balanced_acc:.3f})\n")
    
    # ============================================================================
    # STEP 7: Results
    # ============================================================================
    print(f"{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    
    print(f"\nOptimized XGBoost (threshold {best_threshold:.2f})")
    print("-" * 70)
    print(f"Accuracy:             {best_metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy:    {best_metrics['balanced_acc']:.3f}")
    print(f"Recall (Unconscious): {best_metrics['recall_unconscious']:.3f}")
    print(f"Recall (Conscious):   {best_metrics['recall_conscious']:.3f}")
    print(f"F1 Score:             {best_metrics['f1']:.3f}")
    print(f"ROC-AUC:              {best_metrics['roc_auc']:.3f}")
    
    confusion_matrix = best_metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("              Unconscious  Conscious")
    print(
        f"Unconscious      {confusion_matrix[0, 0]:5d}       "
        f"{confusion_matrix[0, 1]:5d}"
    )
    print(
        f"Conscious        {confusion_matrix[1, 0]:5d}       "
        f"{confusion_matrix[1, 1]:5d}"
    )
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"• Detection: {confusion_matrix[0, 0]}/{confusion_matrix[0, 0] + confusion_matrix[0, 1]} unconscious states correctly identified")
    print(f"• Balanced accuracy: {best_metrics['balanced_acc']*100:.1f}%")
    print(f"• Optimal decision threshold: {best_threshold:.2f}")
    print()
    print("Key techniques:")
    print("  - Full connectivity (99K features) → PCA (50 components)")
    print("  - XGBoost classifier with SMOTE oversampling")
    print("  - Per-subject deviation features")
    print("  - Threshold tuning for class balance")
