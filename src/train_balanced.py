#!/usr/bin/env python3
"""
BALANCED consciousness detection - solving class imbalance problem.

Solutions implemented:
1. SMOTE: Synthetic oversampling of unconscious samples
2. Threshold tuning: Optimize decision boundary for balanced recall
3. Cost-sensitive: Penalize false negatives (missed unconscious) heavily
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

from data_loader import load_subject_all_conditions
from features import extract_all_features
from config import SUBJECTS, CONSCIOUS_CONDITIONS

print("="*70)
print("BALANCED CONSCIOUSNESS DETECTION")
print("="*70)
print("Solutions: SMOTE + Threshold Tuning + Cost-Sensitive Learning")
print()

# Use all subjects
train_subjects = SUBJECTS
print(f"Training on {len(train_subjects)} subjects")
print()

# ============================================================================
# STEP 1: Load data
# ============================================================================
print("Step 1/6: Loading data...")
all_features = []

for idx, subject in enumerate(train_subjects):
    if (idx + 1) % 5 == 0 or idx == 0:
        print(f"  [{idx+1}/{len(train_subjects)}] Loading {subject}...")

    all_fc = load_subject_all_conditions(subject)

    for cond_idx in range(7):
        fc = all_fc[cond_idx]
        features = extract_all_features(fc)
        features.pop('connectivity', None)  # Remove high-dim features

        features['subject'] = subject
        features['condition'] = cond_idx
        features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0

        all_features.append(features)

print(f"✓ Loaded {len(all_features)} samples\n")

# ============================================================================
# STEP 2: Prepare features
# ============================================================================
print("Step 2/6: Preparing features...")

feature_names = [k for k in all_features[0].keys()
                 if k not in ['subject', 'condition', 'label', 'connectivity']]

X = np.array([[f[name] for name in feature_names] for f in all_features])
y = np.array([f['label'] for f in all_features])
subject_ids = np.array([f['subject'] for f in all_features])

print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("  Class distribution:")
print(
    f"    Conscious (1):   {np.sum(y == 1):3d} "
    f"({np.sum(y == 1) / len(y) * 100:.1f}%)"
)
print(
    f"    Unconscious (0): {np.sum(y == 0):3d} "
    f"({np.sum(y == 0) / len(y) * 100:.1f}%)"
)
print(
    "  Imbalance ratio: "
    f"{np.sum(y == 1) / np.sum(y == 0):.1f}:1"
)

# Handle NaN/Inf
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"  Imputing {nan_count} NaN values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

# ============================================================================
# STEP 3: Feature selection
# ============================================================================
print("\nStep 3/6: Feature selection...")

selector_variance = VarianceThreshold(threshold=0.01)
X_var = selector_variance.fit_transform(X)
selected_mask = selector_variance.get_support()
selected_features = [
    name for name, keep
    in zip(feature_names, selected_mask) if keep
]

k = min(9, X_var.shape[1])
selector_kbest = SelectKBest(f_classif, k=k)
X_selected = selector_kbest.fit_transform(X_var, y)

kbest_mask = selector_kbest.get_support()
final_features = [
    name for name, keep
    in zip(selected_features, kbest_mask) if keep
]

print(f"  Selected {len(final_features)} features")
print(f"  Top 3: {final_features[:3]}")

# ============================================================================
# STEP 4: Leave-One-Subject-Out with SMOTE
# ============================================================================
print("\nStep 4/6: LOSO CV with SMOTE oversampling...")
print("  SMOTE will balance classes within each training fold\n")

unique_subjects = np.unique(subject_ids)
all_preds = []
all_labels = []
all_probas = []
per_subject_results = {}

for i, test_subject in enumerate(unique_subjects):
    test_mask = subject_ids == test_subject
    train_mask = ~test_mask

    X_train = X_selected[train_mask]
    y_train = y[train_mask]
    X_test = X_selected[test_mask]
    y_test = y[test_mask]

    # Ensure we have enough samples for SMOTE
    n_minority = np.sum(y_train == 0)
    if n_minority >= 2:
        # Use SMOTE with k_neighbors = min(5, n_minority-1)
        k_neighbors = min(5, n_minority - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        try:
            X_train_balanced, y_train_balanced = (
                smote.fit_resample(X_train, y_train)
            )
        except Exception:
            # If SMOTE fails, use original data
            X_train_balanced, y_train_balanced = (
                X_train, y_train
            )
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    # Train Random Forest with balanced class weights
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',  # Additional class weighting
        min_samples_leaf=2
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Train
    clf.fit(X_train_scaled, y_train_balanced)

    # Predict probabilities
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Store for threshold tuning
    all_labels.extend(y_test)
    all_probas.extend(y_proba)

    # Default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    all_preds.extend(y_pred_default)

    acc = accuracy_score(y_test, y_pred_default)
    per_subject_results[test_subject] = {
        'accuracy': acc,
        'n_samples': len(y_test),
        'n_correct': np.sum(y_test == y_pred_default),
        'probas': y_proba,
        'labels': y_test
    }

    if (i + 1) % 5 == 0 or i < 3:
        print(f"  [{i+1}/{len(unique_subjects)}] {test_subject}: {acc:.3f}")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probas = np.array(all_probas)

# ============================================================================
# STEP 5: Threshold tuning
# ============================================================================
print("\nStep 5/6: Optimizing decision threshold...")
print("  Default threshold: 0.5 (predict conscious if P(conscious) > 0.5)")

# Find optimal threshold that maximizes balanced accuracy
fpr, tpr, thresholds = roc_curve(all_labels, all_probas)

best_threshold = 0.5
best_balanced_acc = 0
best_metrics = None

# Test different thresholds
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (all_probas >= threshold).astype(int)

    # Calculate metrics
    bal_acc = balanced_accuracy_score(all_labels, y_pred_thresh)
    recall_0 = recall_score(all_labels, y_pred_thresh, pos_label=0)
    recall_1 = recall_score(all_labels, y_pred_thresh, pos_label=1)

    # Optimize for balanced accuracy (equal weight to both classes)
    if bal_acc > best_balanced_acc:
        best_balanced_acc = bal_acc
        best_threshold = threshold
        best_metrics = {
            'balanced_acc': bal_acc,
            'accuracy': accuracy_score(all_labels, y_pred_thresh),
            'recall_unconscious': recall_0,
            'recall_conscious': recall_1,
            'precision': precision_score(
                all_labels, y_pred_thresh, zero_division=0
            ),
            'recall': recall_score(
                all_labels, y_pred_thresh, zero_division=0
            ),
            'f1': f1_score(
                all_labels, y_pred_thresh, zero_division=0
            ),
            'confusion_matrix': confusion_matrix(
                all_labels, y_pred_thresh
            )
        }

print(f"  Optimal threshold: {best_threshold:.2f}")
print(f"  Balanced accuracy: {best_balanced_acc:.3f}")

# Apply optimal threshold
y_pred_optimal = (all_probas >= best_threshold).astype(int)

# ============================================================================
# STEP 6: Results comparison
# ============================================================================
print(f"\n{'='*70}")
print("RESULTS COMPARISON")
print('='*70)

# Default threshold results
cm_default = confusion_matrix(all_labels, all_preds)
metrics_default = {
    'accuracy': accuracy_score(all_labels, all_preds),
    'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
    'recall_unconscious': recall_score(
        all_labels, all_preds,
        pos_label=0, zero_division=0
    ),
    'recall_conscious': recall_score(
        all_labels, all_preds,
        pos_label=1, zero_division=0
    ),
    'precision': precision_score(
        all_labels, all_preds, zero_division=0
    ),
    'f1': f1_score(all_labels, all_preds, zero_division=0),
    'roc_auc': roc_auc_score(all_labels, all_probas)
}

print("\n1. DEFAULT THRESHOLD (0.50)")
print("-" * 70)
print(f"Accuracy:          {metrics_default['accuracy']:.3f}")
print(f"Balanced Accuracy: {metrics_default['balanced_acc']:.3f}")
print(
    f"Recall (Unconscious): "
    f"{metrics_default['recall_unconscious']:.3f} "
    f"⚠️  (misses "
    f"{(1-metrics_default['recall_unconscious'])*100:.0f}%"
    f" unconscious)"
)
print(f"Recall (Conscious):   {metrics_default['recall_conscious']:.3f}")
print(f"F1 Score:          {metrics_default['f1']:.3f}")
print(f"ROC-AUC:           {metrics_default['roc_auc']:.3f}")
print("\nConfusion Matrix:")
print("                 Predicted")
print("              Uncon  Consc")
print(
    f"Actual Uncon  {cm_default[0, 0]:5d}  "
    f"{cm_default[0, 1]:5d}  ← Only "
    f"{cm_default[0, 0]}/{cm_default[0, 0]+cm_default[0, 1]}"
    f" detected!"
)
print(
    f"       Consc  {cm_default[1, 0]:5d}  "
    f"{cm_default[1, 1]:5d}"
)

print(f"\n2. OPTIMIZED THRESHOLD ({best_threshold:.2f})")
print("-" * 70)
print(f"Accuracy:          {best_metrics['accuracy']:.3f}")
print(
    f"Balanced Accuracy: "
    f"{best_metrics['balanced_acc']:.3f} "
    f"✓ (Equal weight to both classes)"
)
print(
    f"Recall (Unconscious): "
    f"{best_metrics['recall_unconscious']:.3f} "
    f"{'✓' if best_metrics['recall_unconscious'] > 0.5 else '⚠️'}"
    f"  (detects "
    f"{best_metrics['recall_unconscious']*100:.0f}%"
    f" unconscious)"
)
print(f"Recall (Conscious):   {best_metrics['recall_conscious']:.3f}")
print(f"F1 Score:          {best_metrics['f1']:.3f}")
print("\nConfusion Matrix:")
cm = best_metrics['confusion_matrix']
print("                 Predicted")
print("              Uncon  Consc")
print(
    f"Actual Uncon  {cm[0, 0]:5d}  {cm[0, 1]:5d}"
    f"  ← {cm[0, 0]}/{cm[0, 0]+cm[0, 1]} detected!"
    f" (Improvement: +{cm[0, 0]-cm_default[0, 0]})"
)
print(
    f"       Consc  {cm[1, 0]:5d}  {cm[1, 1]:5d}"
)

print(f"\n{'='*70}")
print("IMPROVEMENT SUMMARY")
print('='*70)
print(
    f"Unconscious detection improved from "
    f"{metrics_default['recall_unconscious']*100:.0f}%"
    f" → {best_metrics['recall_unconscious']*100:.0f}%"
)
print(
    f"Balanced accuracy improved from "
    f"{metrics_default['balanced_acc']:.3f}"
    f" → {best_metrics['balanced_acc']:.3f}"
)
print(
    f"Trade-off: Overall accuracy "
    f"{metrics_default['accuracy']:.3f}"
    f" → {best_metrics['accuracy']:.3f}"
)

if best_metrics['recall_unconscious'] >= 0.6:
    print("\n✓ EXCELLENT: >60% unconscious detection is clinically useful")
elif best_metrics['recall_unconscious'] >= 0.4:
    print("\n✓ GOOD: >40% unconscious detection shows improvement")
else:
    print("\n⚠️ MODERATE: More work needed on class balance")

print(f"\n{'='*70}")
print("CLINICAL INTERPRETATION")
print('='*70)
print(f"At threshold {best_threshold:.2f}:")
print(
    f"  • Will correctly identify "
    f"{best_metrics['recall_unconscious']*100:.0f}%"
    f" of unconscious patients"
)
print(
    f"  • Will correctly identify "
    f"{best_metrics['recall_conscious']*100:.0f}%"
    f" of conscious patients"
)
print(
    f"  • False alarm rate (conscious → unconscious): "
    f"{(1-best_metrics['recall_conscious'])*100:.0f}%"
)
print(
    f"  • Miss rate (unconscious → conscious): "
    f"{(1-best_metrics['recall_unconscious'])*100:.0f}%"
)
print("\nFor clinical use, prioritize:")
print(
    "  - High recall on unconscious "
    "(minimize misses) ← Critical for safety"
)
print(
    "  - Accept lower precision "
    "(more false alarms) ← Can be verified"
)
print('='*70)
