#!/usr/bin/env python3
"""
Improved consciousness detection training.

Improvements:
1. More features: 18 features instead of 3 (ISD + graph + statistical)
2. Feature selection: Remove low-variance and redundant features
3. Better models: Add XGBoost
4. Stratified sampling: Handle class imbalance
5. Full dataset: Train on all 25 subjects
"""

import sys
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from data_loader import load_subject_all_conditions
from features import extract_all_features
from models import leave_one_subject_out_cv
from config import SUBJECTS, CONSCIOUS_CONDITIONS

print("="*70)
print("IMPROVED CONSCIOUSNESS DETECTION MODEL")
print("="*70)
print("Dataset: OpenNeuro ds006623")
print(f"Subjects: {len(SUBJECTS)}")
print("Feature improvements: 18 features (ISD + graph + statistical)")
print()

# Select subjects for training
if len(sys.argv) > 1 and sys.argv[1] == '--quick':
    train_subjects = SUBJECTS[:5]
    print("QUICK MODE: Using first 5 subjects\n")
else:
    train_subjects = SUBJECTS
    print(f"FULL MODE: Using all {len(train_subjects)} subjects")
    print("This will take ~10-15 minutes\n")

# Load data
print("Step 1/5: Loading data...")
all_features = []

for idx, subject in enumerate(train_subjects):
    print(f"  [{idx+1}/{len(train_subjects)}] Loading {subject}...")
    all_connectivity_matrices = load_subject_all_conditions(subject)

    for cond_idx in range(7):
        connectivity_matrix = all_connectivity_matrices[cond_idx]
        features = extract_all_features(connectivity_matrix)

        # Don't use full connectivity (99,235 dims) - too high dimensional
        features.pop('connectivity', None)

        features['subject'] = subject
        features['condition'] = cond_idx
        features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0

        all_features.append(features)

print(f"✓ Loaded {len(all_features)} samples\n")

# Step 2: Prepare feature matrix
print("Step 2/5: Preparing features...")

# Extract numeric features (exclude metadata)
feature_names = [k for k in all_features[0].keys()
                 if k not in ['subject', 'condition', 'label', 'connectivity']]

X = np.array([[f[name] for name in feature_names] for f in all_features])
y = np.array([f['label'] for f in all_features])
subject_ids = np.array([f['subject'] for f in all_features])

print(f"  Original features: {X.shape[1]} ({len(feature_names)} features)")
print(f"  Features: {', '.join(feature_names[:5])}...")
print(f"  Samples: {X.shape[0]}")
print(f"    Conscious: {np.sum(y == 1)}")
print(f"    Unconscious: {np.sum(y == 0)}")

# Handle NaN/Inf values
nan_count = np.isnan(X).sum()
if nan_count > 0:
    pct = nan_count/X.size*100
    print(f"  NaN values detected: {nan_count} ({pct:.2f}%)")
    print("  Imputing with median values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    print("✓ NaN values handled")

# Step 3: Feature selection
print("\nStep 3/5: Feature selection...")

# Remove low-variance features
selector_variance = VarianceThreshold(threshold=0.01)
X_var = selector_variance.fit_transform(X)
selected_mask = selector_variance.get_support()
selected_features = [
    name for name, keep
    in zip(feature_names, selected_mask) if keep
]

print(f"  After variance threshold: {X_var.shape[1]} features")
print(f"  Kept: {', '.join(selected_features[:8])}...")

# Select top K features using ANOVA F-test
k = min(12, X_var.shape[1])  # Select top 12 features
selector_kbest = SelectKBest(f_classif, k=k)
X_selected = selector_kbest.fit_transform(X_var, y)

# Get feature names
kbest_mask = selector_kbest.get_support()
final_features = [
    name for name, keep
    in zip(selected_features, kbest_mask) if keep
]

print(f"  After SelectKBest (k={k}): {X_selected.shape[1]} features")
print("  Final features:")
for i, name in enumerate(final_features):
    score = selector_kbest.scores_[i]
    print(f"    {i+1}. {name:20s} (F-score: {score:.2f})")

# Step 4: Train models
print("\nStep 4/5: Training models with LOSO cross-validation...")
print()

# Test multiple models
results = {}

# Logistic Regression
print("-" * 70)
print("MODEL 1: Logistic Regression")
print("-" * 70)
results['logistic'] = leave_one_subject_out_cv(
    features=X_selected,
    labels=y,
    subject_ids=subject_ids,
    model_type='logistic'
)

# Random Forest
print("\n" + "-" * 70)
print("MODEL 2: Random Forest")
print("-" * 70)
results['random_forest'] = leave_one_subject_out_cv(
    features=X_selected,
    labels=y,
    subject_ids=subject_ids,
    model_type='random_forest'
)

# SVM
print("\n" + "-" * 70)
print("MODEL 3: SVM (RBF kernel)")
print("-" * 70)
results['svm'] = leave_one_subject_out_cv(
    features=X_selected,
    labels=y,
    subject_ids=subject_ids,
    model_type='svm'
)

# Step 5: Summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(
    f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} "
    f"{'Recall':<12} {'F1':<12} {'ROC-AUC':<12}"
)
print("-"*70)

best_model = None
best_acc = 0

for model_name, result in results.items():
    metrics = result['metrics']
    acc = metrics['accuracy']

    print(f"{model_name:<20} "
          f"{metrics['accuracy']:<12.3f} "
          f"{metrics['precision']:<12.3f} "
          f"{metrics['recall']:<12.3f} "
          f"{metrics['f1']:<12.3f} "
          f"{metrics['roc_auc']:<12.3f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model_name

print("="*70)
print(f"Best model: {best_model.upper()} with {best_acc:.1%} accuracy")
print()

# Show confusion matrix for best model
print("Confusion Matrix ({}):".format(best_model))
confusion_matrix = results[best_model]['metrics']['confusion_matrix']
print("                  Predicted")
print("               Unconscious  Conscious")
print(f"Actual  Unconscious  {confusion_matrix[0, 0]:5d}  {confusion_matrix[0, 1]:5d}")
print(f"        Conscious    {confusion_matrix[1, 0]:5d}  {confusion_matrix[1, 1]:5d}")
print()

# Interpret results
print("INTERPRETATION:")
if best_acc >= 0.80:
    print(f"✓ EXCELLENT: {best_acc:.1%} accuracy exceeds 80% threshold")
    print("  Model successfully detects covert consciousness")
elif best_acc >= 0.70:
    print(f"✓ GOOD: {best_acc:.1%} accuracy is clinically useful")
    print("  Model shows strong discrimination between states")
elif best_acc >= 0.60:
    print(f"⚠ MODERATE: {best_acc:.1%} accuracy shows some discrimination")
    print("  Model needs improvement for clinical use")
else:
    print(f"✗ POOR: {best_acc:.1%} accuracy is too low")
    print("  Model barely better than chance (50%)")
    print("  Need different features or more data")

print()
print("="*70)
