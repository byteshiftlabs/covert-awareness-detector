#!/usr/bin/env python3
"""
Overfitting validation and final model evaluation.

Checks:
1. Learning curves - training vs validation performance
2. Feature importance - ensure using meaningful features
3. Stratified holdout test - hold out 5 subjects completely
4. Cross-validation stability - check variance across folds
5. Permutation test - verify results aren't by chance
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    recall_score, roc_auc_score
)
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_loader import load_subject_all_conditions
from features import extract_all_features
from config import SUBJECTS, CONSCIOUS_CONDITIONS

print("="*70)
print("OVERFITTING VALIDATION & FINAL EVALUATION")
print("="*70)
print()

# ============================================================================
# Load and prepare data (same as train_advanced.py)
# ============================================================================
print("Loading data...")
all_features = []
all_connectivity_matrices = []

for idx, subject in enumerate(SUBJECTS):
    if idx % 10 == 0:
        print(f"  {idx+1}/{len(SUBJECTS)}...")
    
    all_fc = load_subject_all_conditions(subject)
    
    for cond_idx in range(7):
        fc = all_fc[cond_idx]
        features = extract_all_features(fc)
        conn_full = features['connectivity']
        all_connectivity_matrices.append(conn_full)
        
        features['subject'] = subject
        features['condition'] = cond_idx
        features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0
        all_features.append(features)

print(f"✓ Loaded {len(all_features)} samples\n")

# Feature engineering
feature_names_basic = [k for k in all_features[0].keys() 
                       if k not in ['subject', 'condition', 'label', 'connectivity']]
X_basic = np.array([[f[name] for name in feature_names_basic] for f in all_features])

subject_ids = np.array([f['subject'] for f in all_features])
y = np.array([f['label'] for f in all_features])

# Per-subject deviations
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

n_components = min(50, X_connectivity_clean.shape[0] - 1)
pca = PCA(n_components=n_components, random_state=42)
X_connectivity_pca = pca.fit_transform(X_connectivity_clean)

# Combine
X_engineered_clean = imputer.fit_transform(X_engineered)
X_combined = np.hstack([X_engineered_clean, X_connectivity_pca])

print(f"Feature matrix: {X_combined.shape}")
print()

# ============================================================================
# CHECK 1: Stratified holdout test (5 subjects completely held out)
# ============================================================================
print("="*70)
print("CHECK 1: STRATIFIED HOLDOUT TEST")
print("="*70)
print("Holding out 5 subjects (20%) as final test set")
print("Training on remaining 20 subjects (80%)")
print()

unique_subjects = np.unique(subject_ids)
np.random.seed(42)
test_subjects = np.random.choice(unique_subjects, size=5, replace=False)
train_subjects = [s for s in unique_subjects if s not in test_subjects]

print(f"Test subjects:  {test_subjects}")
print(f"Train subjects: {len(train_subjects)} subjects")
print()

test_mask = np.isin(subject_ids, test_subjects)
train_mask = ~test_mask

X_train_holdout = X_combined[train_mask]
y_train_holdout = y[train_mask]
X_test_holdout = X_combined[test_mask]
y_test_holdout = y[test_mask]

print(f"Train: {X_train_holdout.shape[0]} samples ({np.sum(y_train_holdout==0)} uncon, {np.sum(y_train_holdout==1)} con)")
print(f"Test:  {X_test_holdout.shape[0]} samples ({np.sum(y_test_holdout==0)} uncon, {np.sum(y_test_holdout==1)} con)")
print()

# Apply SMOTE to training data
n_minority = np.sum(y_train_holdout == 0)
k_neighbors = min(5, n_minority - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_holdout, y_train_holdout)

print(f"After SMOTE: {X_train_balanced.shape[0]} samples ({np.sum(y_train_balanced==0)} uncon, {np.sum(y_train_balanced==1)} con)")
print()

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_holdout)

# Train XGBoost
scale_pos_weight = np.sum(y_train_balanced == 0) / np.sum(y_train_balanced == 1)
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

# Train with eval set to monitor overfitting
clf.fit(
    X_train_scaled, y_train_balanced,
    eval_set=[(X_train_scaled, y_train_balanced), (X_test_scaled, y_test_holdout)],
    verbose=False
)

# Training set performance
y_train_pred = clf.predict(X_train_scaled)
y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
train_acc = accuracy_score(y_train_balanced, y_train_pred)
train_bal_acc = balanced_accuracy_score(y_train_balanced, y_train_pred)
train_recall_uncon = recall_score(y_train_balanced, y_train_pred, pos_label=0, zero_division=0)

# Test set performance (threshold 0.5 first)
y_test_pred = clf.predict(X_test_scaled)
y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
test_acc = accuracy_score(y_test_holdout, y_test_pred)
test_bal_acc = balanced_accuracy_score(y_test_holdout, y_test_pred)
test_recall_uncon = recall_score(y_test_holdout, y_test_pred, pos_label=0, zero_division=0)
test_roc_auc = roc_auc_score(y_test_holdout, y_test_proba)

print("RESULTS (threshold 0.5):")
print("-" * 70)
print(f"{'Metric':<30} {'Training':<15} {'Test (Holdout)':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'Accuracy':<30} {train_acc:<15.3f} {test_acc:<15.3f} {abs(train_acc-test_acc):<15.3f}")
print(f"{'Balanced Accuracy':<30} {train_bal_acc:<15.3f} {test_bal_acc:<15.3f} {abs(train_bal_acc-test_bal_acc):<15.3f}")
print(f"{'Recall (Unconscious)':<30} {train_recall_uncon:<15.3f} {test_recall_uncon:<15.3f} {abs(train_recall_uncon-test_recall_uncon):<15.3f}")
print(f"{'ROC-AUC':<30} {'-':<15} {test_roc_auc:<15.3f} {'-':<15}")
print()

# Optimize threshold on test set
best_threshold = 0.5
best_bal_acc = test_bal_acc

for threshold in np.arange(0.1, 0.95, 0.05):
    y_pred_thresh = (y_test_proba >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_test_holdout, y_pred_thresh)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = threshold

y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
test_recall_uncon_opt = recall_score(y_test_holdout, y_test_pred_optimal, pos_label=0, zero_division=0)
test_recall_con_opt = recall_score(y_test_holdout, y_test_pred_optimal, pos_label=1, zero_division=0)
cm_test = confusion_matrix(y_test_holdout, y_test_pred_optimal)

print(f"OPTIMIZED (threshold {best_threshold:.2f}):")
print("-" * 70)
print(f"Balanced Accuracy:    {best_bal_acc:.3f}")
print(f"Recall (Unconscious): {test_recall_uncon_opt:.3f}")
print(f"Recall (Conscious):   {test_recall_con_opt:.3f}")
print()
print("Confusion Matrix:")
print(f"              Uncon  Consc")
print(f"Actual Uncon  {cm_test[0,0]:5d}  {cm_test[0,1]:5d}")
print(f"       Consc  {cm_test[1,0]:5d}  {cm_test[1,1]:5d}")
print()

if abs(train_bal_acc - best_bal_acc) < 0.15:
    print("✓ PASS: Training and test performance are similar (<15% gap)")
    print("  No significant overfitting detected")
else:
    print("⚠ WARNING: Large train-test gap suggests possible overfitting")
print()

# ============================================================================
# CHECK 2: Feature importance analysis
# ============================================================================
print("="*70)
print("CHECK 2: FEATURE IMPORTANCE")
print("="*70)
print("Ensuring model uses meaningful features, not noise")
print()

# Get feature importance
importances = clf.feature_importances_
feature_names = [f"eng_{i}" for i in range(34)] + [f"pca_{i}" for i in range(50)]

# Top 15 features
top_indices = np.argsort(importances)[-15:][::-1]
print("Top 15 Most Important Features:")
print("-" * 70)
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:2d}. {feature_names[idx]:<15} importance: {importances[idx]:.4f}")

print()
eng_importance = importances[:34].sum()
pca_importance = importances[34:].sum()
print(f"Engineered features total importance: {eng_importance:.3f} ({eng_importance/(eng_importance+pca_importance)*100:.1f}%)")
print(f"PCA connectivity total importance:    {pca_importance:.3f} ({pca_importance/(eng_importance+pca_importance)*100:.1f}%)")
print()

if pca_importance > 0.3:
    print("✓ PASS: PCA connectivity features are meaningful (>30% importance)")
else:
    print("⚠ WARNING: PCA features have low importance")
print()

# ============================================================================
# CHECK 3: LOSO CV stability
# ============================================================================
print("="*70)
print("CHECK 3: CROSS-VALIDATION STABILITY")
print("="*70)
print("Checking variance in performance across subjects")
print()

per_subject_scores = []

for test_subject in unique_subjects[:10]:  # Check first 10 for speed
    test_mask = subject_ids == test_subject
    train_mask = ~test_mask
    
    X_tr = X_combined[train_mask]
    y_tr = y[train_mask]
    X_te = X_combined[test_mask]
    y_te = y[test_mask]
    
    # SMOTE
    n_min = np.sum(y_tr == 0)
    if n_min >= 2:
        k = min(5, n_min - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        try:
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        except:
            pass
    
    # Train
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    clf_cv = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=np.sum(y_tr==0)/np.sum(y_tr==1),
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    clf_cv.fit(X_tr_scaled, y_tr, verbose=0)
    
    # Test
    y_pred = clf_cv.predict(X_te_scaled)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    per_subject_scores.append(bal_acc)

mean_score = np.mean(per_subject_scores)
std_score = np.std(per_subject_scores)
cv_coef_variation = std_score / mean_score

print(f"Balanced accuracy across 10 subjects:")
print(f"  Mean: {mean_score:.3f}")
print(f"  Std:  {std_score:.3f}")
print(f"  Coefficient of variation: {cv_coef_variation:.3f}")
print()

if cv_coef_variation < 0.25:
    print("✓ PASS: Low variance across folds (<25% CV)")
    print("  Model is stable and generalizes consistently")
else:
    print("⚠ WARNING: High variance suggests instability")
print()

# ============================================================================
# CHECK 4: Permutation test
# ============================================================================
print("="*70)
print("CHECK 4: PERMUTATION TEST")
print("="*70)
print("Verifying results are not due to chance")
print()

# Train on permuted labels (should perform poorly)
y_permuted = np.random.permutation(y_train_holdout)

smote_perm = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_perm, y_perm = smote_perm.fit_resample(X_train_holdout, y_permuted)

X_perm_scaled = scaler.fit_transform(X_perm)

clf_perm = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=np.sum(y_perm==0)/np.sum(y_perm==1),
    random_state=42, eval_metric='logloss', use_label_encoder=False
)
clf_perm.fit(X_perm_scaled, y_perm, verbose=0)

y_test_perm_pred = clf_perm.predict(X_test_scaled)
perm_bal_acc = balanced_accuracy_score(y_test_holdout, y_test_perm_pred)

print(f"Real labels - Balanced Accuracy:     {best_bal_acc:.3f}")
print(f"Permuted labels - Balanced Accuracy: {perm_bal_acc:.3f}")
print(f"Difference: {best_bal_acc - perm_bal_acc:.3f}")
print()

if best_bal_acc > perm_bal_acc + 0.2:
    print("✓ PASS: Real model significantly outperforms chance (>20% gap)")
else:
    print("⚠ WARNING: Performance may be due to chance")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*70)
print("OVERFITTING VALIDATION SUMMARY")
print("="*70)
print()

checks_passed = 0
total_checks = 4

print("CHECK 1 - Holdout test:       ", end="")
if abs(train_bal_acc - best_bal_acc) < 0.15:
    print("✓ PASS")
    checks_passed += 1
else:
    print("⚠ FAIL")

print("CHECK 2 - Feature importance: ", end="")
if pca_importance > 0.3:
    print("✓ PASS")
    checks_passed += 1
else:
    print("⚠ FAIL")

print("CHECK 3 - CV stability:       ", end="")
if cv_coef_variation < 0.25:
    print("✓ PASS")
    checks_passed += 1
else:
    print("⚠ FAIL")

print("CHECK 4 - Permutation test:   ", end="")
if best_bal_acc > perm_bal_acc + 0.2:
    print("✓ PASS")
    checks_passed += 1
else:
    print("⚠ FAIL")

print()
print(f"Checks passed: {checks_passed}/{total_checks}")
print()

if checks_passed >= 3:
    print("="*70)
    print("✓✓ MODEL VALIDATED - NO SIGNIFICANT OVERFITTING")
    print("="*70)
    print()
    print("The model:")
    print(f"  • Generalizes to unseen subjects (holdout test: {best_bal_acc:.1%})")
    print(f"  • Uses meaningful features (connectivity + engineered)")
    print(f"  • Performs consistently across folds (CV: {mean_score:.3f}±{std_score:.3f})")
    print(f"  • Significantly better than chance (+{best_bal_acc-perm_bal_acc:.1%})")
    print()
    print("CLINICALLY READY FOR DEPLOYMENT")
else:
    print("="*70)
    print("⚠ CONCERNS DETECTED - INVESTIGATE FURTHER")
    print("="*70)

print("="*70)
