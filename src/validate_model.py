#!/usr/bin/env python3
"""
Overfitting validation: holdout test, feature importance, CV stability, permutation test.
"""

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_loader import prepare_data
from config import (
    RANDOM_STATE, N_CONDITIONS,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE,
    XGBOOST_SUBSAMPLE, XGBOOST_COLSAMPLE_BYTREE,
    THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX, THRESHOLD_SEARCH_STEP,
    HOLDOUT_TEST_SUBJECTS, CV_STABILITY_SUBJECTS, ENGINEERED_FEATURES_COUNT,
    HOLDOUT_ACCURACY_THRESHOLD, PCA_IMPORTANCE_THRESHOLD,
    CV_VARIANCE_THRESHOLD, PERMUTATION_MARGIN
)

warnings.filterwarnings('ignore')


def train_model(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost with SMOTE and return model and predictions."""
    # SMOTE
    n_minority = np.sum(y_train == 0)
    if n_minority >= 2:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, n_minority - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    classifier = xgb.XGBClassifier(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    classifier.fit(X_train_scaled, y_train, verbose=0)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        prediction_probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
        return classifier, prediction_probabilities, scaler

    return classifier, None, scaler


def optimize_threshold(y_true, prediction_probabilities):
    """Find optimal classification threshold."""
    best_threshold, best_balanced_accuracy = 0.5, 0
    for threshold in np.arange(THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX, THRESHOLD_SEARCH_STEP):
        y_pred = (prediction_probabilities >= threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy, best_threshold = balanced_accuracy, threshold
    return best_threshold, best_balanced_accuracy


def main():
    """Run all overfitting validation checks."""
    print("="*70)
    print("OVERFITTING VALIDATION")
    print("="*70)

    # Prepare data
    print("\nLoading data...")
    X_combined, y, subject_ids, _pca, _imputer = prepare_data()
    print(f"Feature matrix: {X_combined.shape}\n")

    # CHECK 1: Holdout test (configurable number of subjects held out)
    print("CHECK 1: HOLDOUT TEST")
    print("-" * 70)
    unique_subjects = np.unique(subject_ids)
    np.random.seed(RANDOM_STATE)
    test_subjects = np.random.choice(unique_subjects, size=HOLDOUT_TEST_SUBJECTS, replace=False)

    test_mask = np.isin(subject_ids, test_subjects)
    X_train, y_train = X_combined[~test_mask], y[~test_mask]
    X_test, y_test = X_combined[test_mask], y[test_mask]

    classifier, prediction_probabilities, _scaler = train_model(X_train, y_train, X_test, y_test)
    best_threshold, best_balanced_accuracy = optimize_threshold(y_test, prediction_probabilities)

    y_pred_optimal = (prediction_probabilities >= best_threshold).astype(int)
    confusion_matrix_result = confusion_matrix(y_test, y_pred_optimal)
    recall_unconscious = recall_score(y_test, y_pred_optimal, pos_label=0, zero_division=0)
    recall_conscious = recall_score(y_test, y_pred_optimal, pos_label=1, zero_division=0)

    print(f"Test subjects: {test_subjects}")
    print(f"Balanced Accuracy: {best_balanced_accuracy:.3f} (threshold {best_threshold:.2f})")
    print(f"Recall - Unconscious: {recall_unconscious:.3f}, Conscious: {recall_conscious:.3f}")
    print(f"Confusion: [[{confusion_matrix_result[0,0]}, {confusion_matrix_result[0,1]}], [{confusion_matrix_result[1,0]}, {confusion_matrix_result[1,1]}]]")

    holdout_test_passed = best_balanced_accuracy > HOLDOUT_ACCURACY_THRESHOLD
    print(f"{'✓ PASS' if holdout_test_passed else '⚠ FAIL'}: {'Good' if holdout_test_passed else 'Low'} generalization\n")

    # CHECK 2: Feature importance
    print("CHECK 2: FEATURE IMPORTANCE")
    print("-" * 70)
    importances = classifier.feature_importances_
    engineered_importance = importances[:ENGINEERED_FEATURES_COUNT].sum()
    pca_importance = importances[ENGINEERED_FEATURES_COUNT:].sum()

    print(f"Engineered features: {engineered_importance:.3f} ({engineered_importance/(engineered_importance+pca_importance)*100:.0f}%)")
    print(f"PCA connectivity:    {pca_importance:.3f} ({pca_importance/(engineered_importance+pca_importance)*100:.0f}%)")

    feature_importance_test_passed = pca_importance > PCA_IMPORTANCE_THRESHOLD
    print(f"{'✓ PASS' if feature_importance_test_passed else '⚠ FAIL'}: PCA features {'are' if feature_importance_test_passed else 'not'} meaningful\n")

    # CHECK 3: CV stability (configurable number of subjects)
    print("CHECK 3: CV STABILITY")
    print("-" * 70)
    cross_validation_scores = []
    for test_subject in unique_subjects[:CV_STABILITY_SUBJECTS]:
        test_mask = subject_ids == test_subject
        X_train_cv, y_train_cv = X_combined[~test_mask], y[~test_mask]
        X_test_cv, y_test_cv = X_combined[test_mask], y[test_mask]

        _, prediction_probabilities_cv, _ = train_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
        y_pred_cv = (prediction_probabilities_cv >= best_threshold).astype(int)
        cross_validation_scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv))

    mean_cross_validation_score = np.mean(cross_validation_scores)
    std_cross_validation_score = np.std(cross_validation_scores)
    coefficient_of_variation = std_cross_validation_score / mean_cross_validation_score

    print(f"Balanced accuracy: {mean_cross_validation_score:.3f} ± {std_cross_validation_score:.3f}")
    print(f"Coefficient of variation: {coefficient_of_variation:.3f}")

    cv_stability_test_passed = coefficient_of_variation < CV_VARIANCE_THRESHOLD
    print(f"{'✓ PASS' if cv_stability_test_passed else '⚠ FAIL'}: {'Low' if cv_stability_test_passed else 'High'} variance across folds\n")

    # CHECK 4: Permutation test
    print("CHECK 4: PERMUTATION TEST")
    print("-" * 70)
    y_permuted = np.random.permutation(y_train)
    _classifier_permuted, prediction_probabilities_permuted, _ = train_model(X_train, y_permuted, X_test, y_test)
    y_pred_permuted = (prediction_probabilities_permuted >= best_threshold).astype(int)
    permuted_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_permuted)

    print(f"Real labels:     {best_balanced_accuracy:.3f}")
    print(f"Permuted labels: {permuted_balanced_accuracy:.3f}")
    print(f"Difference:      {best_balanced_accuracy - permuted_balanced_accuracy:.3f}")

    permutation_test_passed = best_balanced_accuracy > permuted_balanced_accuracy + PERMUTATION_MARGIN
    print(f"{'✓ PASS' if permutation_test_passed else '⚠ FAIL'}: Real model {'significantly' if permutation_test_passed else 'barely'} outperforms chance\n")

    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    checks_passed = sum([holdout_test_passed, feature_importance_test_passed, cv_stability_test_passed, permutation_test_passed])
    print(f"Checks passed: {checks_passed}/4")
    print(f"✓ Holdout test:       {'PASS' if holdout_test_passed else 'FAIL'}")
    print(f"✓ Feature importance: {'PASS' if feature_importance_test_passed else 'FAIL'}")
    print(f"✓ CV stability:       {'PASS' if cv_stability_test_passed else 'FAIL'}")
    print(f"✓ Permutation test:   {'PASS' if permutation_test_passed else 'FAIL'}")

    if checks_passed >= 3:
        print(f"\n✓✓ MODEL VALIDATED - Bal Acc: {best_balanced_accuracy:.1%}, Stable: {mean_cross_validation_score:.3f}±{std_cross_validation_score:.3f}")
    else:
        print("\n⚠ VALIDATION CONCERNS - Investigate further")
    print("="*70)


if __name__ == "__main__":
    main()
