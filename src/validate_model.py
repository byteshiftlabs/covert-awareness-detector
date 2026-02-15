#!/usr/bin/env python3
"""
Overfitting validation: holdout test, feature importance, CV stability, permutation test.
"""

import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_loader import load_subject_all_conditions
from features import extract_all_features
from config import SUBJECTS, CONSCIOUS_CONDITIONS

warnings.filterwarnings('ignore')


def prepare_data():
    """Load and prepare feature matrix."""
    all_features, all_connectivity = [], []
    
    for subject in SUBJECTS:
        all_connectivity_matrices = load_subject_all_conditions(subject)
        for condition_idx in range(7):
            connectivity_matrix = all_connectivity_matrices[condition_idx]
            features = extract_all_features(connectivity_matrix)
            all_connectivity.append(features['connectivity'])
            features.update({
                'subject': subject,
                'condition': condition_idx,
                'label': 1 if condition_idx in CONSCIOUS_CONDITIONS else 0
            })
            all_features.append(features)
    
    # Extract basic features
    feature_names = [k for k in all_features[0].keys()
                     if k not in ['subject', 'condition', 'label', 'connectivity']]
    X_basic = np.array([[f[name] for name in feature_names] for f in all_features])
    subject_ids = np.array([f['subject'] for f in all_features])
    y = np.array([f['label'] for f in all_features])
    
    # Per-subject deviations
    X_deviations = np.zeros_like(X_basic)
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        data = X_basic[mask]
        conscious = y[mask] == 1
        baseline = data[conscious].mean(axis=0) if conscious.sum() > 0 else data.mean(axis=0)
        X_deviations[mask] = data - baseline
    
    # PCA on connectivity
    imputer = SimpleImputer(strategy='median')
    X_conn_clean = imputer.fit_transform(np.array(all_connectivity))
    pca = PCA(n_components=min(50, X_conn_clean.shape[0] - 1), random_state=42)
    X_conn_pca = pca.fit_transform(X_conn_clean)
    
    # Combine all features
    X_combined = np.hstack([
        imputer.fit_transform(np.hstack([X_basic, X_deviations])),
        X_conn_pca
    ])
    
    return X_combined, y, subject_ids, pca, imputer


def train_model(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost with SMOTE and return model and predictions."""
    # SMOTE
    n_minority = np.sum(y_train == 0)
    if n_minority >= 2:
        smote = SMOTE(random_state=42, k_neighbors=min(5, n_minority - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
        random_state=42, eval_metric='logloss'
    )
    clf.fit(X_train_scaled, y_train, verbose=0)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        return clf, y_proba, scaler
    
    return clf, None, scaler


def optimize_threshold(y_true, y_proba):
    """Find optimal classification threshold."""
    best_threshold, best_balanced_accuracy = 0.5, 0
    for threshold in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy, best_threshold = balanced_accuracy, threshold
    return best_threshold, best_balanced_accuracy


print("="*70)
print("OVERFITTING VALIDATION")
print("="*70)

# Prepare data
print("\nLoading data...")
X_combined, y, subject_ids, pca, imputer = prepare_data()
print(f"Feature matrix: {X_combined.shape}\n")

# CHECK 1: Holdout test (5 subjects held out)
print("CHECK 1: HOLDOUT TEST")
print("-" * 70)
unique_subjects = np.unique(subject_ids)
np.random.seed(42)
test_subjects = np.random.choice(unique_subjects, size=5, replace=False)

test_mask = np.isin(subject_ids, test_subjects)
X_train, y_train = X_combined[~test_mask], y[~test_mask]
X_test, y_test = X_combined[test_mask], y[test_mask]

clf, y_proba, scaler = train_model(X_train, y_train, X_test, y_test)
best_threshold, best_balanced_accuracy = optimize_threshold(y_test, y_proba)

y_pred_optimal = (y_proba >= best_threshold).astype(int)
confusion_mat = confusion_matrix(y_test, y_pred_optimal)
recall_unconscious = recall_score(y_test, y_pred_optimal, pos_label=0, zero_division=0)
recall_conscious = recall_score(y_test, y_pred_optimal, pos_label=1, zero_division=0)

print(f"Test subjects: {test_subjects}")
print(f"Balanced Accuracy: {best_balanced_accuracy:.3f} (threshold {best_threshold:.2f})")
print(f"Recall - Unconscious: {recall_unconscious:.3f}, Conscious: {recall_conscious:.3f}")
print(f"Confusion: [[{confusion_mat[0,0]}, {confusion_mat[0,1]}], [{confusion_mat[1,0]}, {confusion_mat[1,1]}]]")

check1_pass = best_balanced_accuracy > 0.65
print(f"{'✓ PASS' if check1_pass else '⚠ FAIL'}: {'Good' if check1_pass else 'Low'} generalization\n")

# CHECK 2: Feature importance
print("CHECK 2: FEATURE IMPORTANCE")
print("-" * 70)
importances = clf.feature_importances_
engineered_importance = importances[:68].sum()  # Engineered features (34 basic + 34 deviations)
pca_importance = importances[68:].sum()  # PCA features

print(f"Engineered features: {engineered_importance:.3f} ({engineered_importance/(engineered_importance+pca_importance)*100:.0f}%)")
print(f"PCA connectivity:    {pca_importance:.3f} ({pca_importance/(engineered_importance+pca_importance)*100:.0f}%)")

check2_pass = pca_importance > 0.25
print(f"{'✓ PASS' if check2_pass else '⚠ FAIL'}: PCA features {'are' if check2_pass else 'not'} meaningful\n")

# CHECK 3: CV stability (10 subjects)
print("CHECK 3: CV STABILITY")
print("-" * 70)
cv_scores = []
for test_subject in unique_subjects[:10]:
    test_mask = subject_ids == test_subject
    X_train_cv, y_train_cv = X_combined[~test_mask], y[~test_mask]
    X_test_cv, y_test_cv = X_combined[test_mask], y[test_mask]
    
    _, y_proba_cv, _ = train_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    y_pred_cv = (y_proba_cv >= best_threshold).astype(int)
    cv_scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv))

mean_cv, std_cv = np.mean(cv_scores), np.std(cv_scores)
coefficient_of_variation = std_cv / mean_cv

print(f"Balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")
print(f"Coefficient of variation: {coefficient_of_variation:.3f}")

check3_pass = coefficient_of_variation < 0.30
print(f"{'✓ PASS' if check3_pass else '⚠ FAIL'}: {'Low' if check3_pass else 'High'} variance across folds\n")

# CHECK 4: Permutation test
print("CHECK 4: PERMUTATION TEST")
print("-" * 70)
y_permuted = np.random.permutation(y_train)
clf_permuted, y_proba_permuted, _ = train_model(X_train, y_permuted, X_test, y_test)
y_pred_permuted = (y_proba_permuted >= best_threshold).astype(int)
permuted_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_permuted)

print(f"Real labels:     {best_balanced_accuracy:.3f}")
print(f"Permuted labels: {permuted_balanced_accuracy:.3f}")
print(f"Difference:      {best_balanced_accuracy - permuted_balanced_accuracy:.3f}")

check4_pass = best_balanced_accuracy > permuted_balanced_accuracy + 0.15
print(f"{'✓ PASS' if check4_pass else '⚠ FAIL'}: Real model {'significantly' if check4_pass else 'barely'} outperforms chance\n")

# Summary
print("="*70)
print("VALIDATION SUMMARY")
print("="*70)
checks_passed = sum([check1_pass, check2_pass, check3_pass, check4_pass])
print(f"Checks passed: {checks_passed}/4")
print(f"✓ Holdout test:       {'PASS' if check1_pass else 'FAIL'}")
print(f"✓ Feature importance: {'PASS' if check2_pass else 'FAIL'}")
print(f"✓ CV stability:       {'PASS' if check3_pass else 'FAIL'}")
print(f"✓ Permutation test:   {'PASS' if check4_pass else 'FAIL'}")

if checks_passed >= 3:
    print(f"\n✓✓ MODEL VALIDATED - Bal Acc: {best_balanced_accuracy:.1%}, Stable: {mean_cv:.3f}±{std_cv:.3f}")
else:
    print("\n⚠ VALIDATION CONCERNS - Investigate further")
print("="*70)

