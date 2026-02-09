"""
ML models for consciousness detection.

This module implements classifiers to predict conscious vs unconscious states
from fMRI functional connectivity data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class ConsciousnessClassifier:
    """
    Binary classifier for conscious vs unconscious states.
    
    Uses Leave-One-Subject-Out (LOSO) cross-validation to evaluate
    generalization to new subjects.
    """
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Args:
            model_type: One of 'logistic', 'random_forest', 'svm'
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = StandardScaler()
        
    def _create_model(self, model_type: str):
        """Create sklearn model based on type."""
        if model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'ConsciousnessClassifier':
        """
        Train model on features and labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,) - 0=unconscious, 1=conscious
            
        Returns:
            self
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict consciousness state.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict consciousness probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples, 2) - [:, 1] is P(conscious)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: True labels (n_samples,)
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }


def leave_one_subject_out_cv(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    model_type: str = 'logistic'
) -> Dict[str, any]:
    """
    Leave-One-Subject-Out cross-validation.
    
    This is the gold standard for neuroimaging: train on N-1 subjects,
    test on the held-out subject. Repeat for all subjects.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Binary labels (n_samples,)
        subject_ids: Subject identifier for each sample (n_samples,)
        model_type: Type of classifier to use
        
    Returns:
        Dictionary with:
            - all_preds: All predictions
            - all_labels: All true labels
            - all_probas: All probabilities
            - metrics: Overall metrics
            - per_subject_metrics: Metrics for each subject
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    all_preds = []
    all_labels = []
    all_probas = []
    per_subject_metrics = {}
    
    print(f"Running LOSO CV with {n_subjects} subjects...")
    
    for i, test_subject in enumerate(unique_subjects):
        # Split into train and test
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        # Train model
        clf = ConsciousnessClassifier(model_type=model_type)
        clf.train(X_train, y_train)
        
        # Predict on test subject
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Store results
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        all_probas.extend(y_proba)
        
        # Per-subject metrics
        per_subject_metrics[test_subject] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'n_samples': len(y_test),
            'n_correct': np.sum(y_test == y_pred)
        }
        
        print(f"  [{i+1}/{n_subjects}] {test_subject}: "
              f"{per_subject_metrics[test_subject]['accuracy']:.3f} "
              f"({per_subject_metrics[test_subject]['n_correct']}/{per_subject_metrics[test_subject]['n_samples']})")
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probas),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    print(f"\nOverall LOSO CV Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    return {
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probas': all_probas,
        'metrics': metrics,
        'per_subject_metrics': per_subject_metrics
    }


def compare_models(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    model_types: List[str] = ['logistic', 'random_forest', 'svm']
) -> Dict[str, Dict]:
    """
    Compare multiple model types using LOSO CV.
    
    Args:
        features: Feature matrix
        labels: Binary labels
        subject_ids: Subject identifiers
        model_types: List of model types to compare
        
    Returns:
        Dictionary mapping model_type -> LOSO CV results
    """
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print('='*60)
        
        results[model_type] = leave_one_subject_out_cv(
            features, labels, subject_ids, model_type
        )
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    print('-'*60)
    
    for model_type in model_types:
        metrics = results[model_type]['metrics']
        print(f"{model_type:<20} "
              f"{metrics['accuracy']:<12.3f} "
              f"{metrics['f1']:<12.3f} "
              f"{metrics['roc_auc']:<12.3f}")
    
    return results
