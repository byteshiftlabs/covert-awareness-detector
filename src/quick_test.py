#!/usr/bin/env python3
"""
Quick test: Train on first 5 subjects to verify everything works.
"""

import numpy as np
from data_loader import load_subject_all_conditions
from features import extract_all_features
from models import leave_one_subject_out_cv
from config import SUBJECTS, CONSCIOUS_CONDITIONS

print("="*70)
print("QUICK TEST: Train on 5 subjects")
print("="*70)

# Use first 5 subjects for quick test
test_subjects = SUBJECTS[:5]
print(f"Subjects: {test_subjects}\n")

# Load data
print("Loading data...")
all_features = []

for subject in test_subjects:
    print(f"  Loading {subject}...")
    all_fc = load_subject_all_conditions(subject)

    for cond_idx in range(7):
        fc = all_fc[cond_idx]
        features = extract_all_features(fc)

        features['subject'] = subject
        features['condition'] = cond_idx
        features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0

        all_features.append(features)

print(f"✓ Loaded {len(all_features)} samples\n")

# Prepare ML data
X = np.array(
    [[f['isd'], f['efficiency'], f['clustering']]
     for f in all_features]
)
y = np.array([f['label'] for f in all_features])
subject_ids = np.array([f['subject'] for f in all_features])

print(f"Features: {X.shape}")
print(f"Labels: {y.shape}")
print(f"  Conscious: {np.sum(y==1)}")
print(f"  Unconscious: {np.sum(y==0)}\n")

# Train with LOSO CV
print("Training Logistic Regression with LOSO CV...\n")
results = leave_one_subject_out_cv(
    features=X,
    labels=y,
    subject_ids=subject_ids,
    model_type='logistic'
)

print("\n" + "="*70)
print("QUICK TEST COMPLETE")
print("="*70)
print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
print(f"F1 Score: {results['metrics']['f1']:.3f}")
print(f"ROC-AUC:  {results['metrics']['roc_auc']:.3f}")
