#!/usr/bin/env python3
"""
Train consciousness detection models on OpenNeuro ds006623 dataset.

Usage:
    python train.py --model logistic
    python train.py --model all --save-results
"""

import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from data_loader import load_all_subjects
from features import extract_all_features
from models import compare_models
from config import SUBJECTS, CONSCIOUS_CONDITIONS


def main():
    parser = argparse.ArgumentParser(
        description='Train consciousness detection models'
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['logistic', 'random_forest', 'svm', 'all'],
                        help='Model type to train')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Directory to save results')

    args = parser.parse_args()

    print("="*70)
    print("CONSCIOUSNESS DETECTION MODEL TRAINING")
    print("="*70)
    print("Dataset: OpenNeuro ds006623")
    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Model: {args.model}")
    print()

    # Step 1: Load data
    print("Step 1/4: Loading data...")
    print("  Computing connectivity matrices for 25 subjects × 7 conditions")
    print("  This may take 3-5 minutes...")

    all_fc, _subjects = load_all_subjects()
    print(f"  ✓ Loaded: {all_fc.shape}")

    # Step 2: Extract features
    print("\nStep 2/4: Extracting features...")
    all_features = []

    for subj_idx, subject in enumerate(SUBJECTS):
        if (subj_idx + 1) % 5 == 0:
            print(f"  Processing subject {subj_idx + 1}/{len(SUBJECTS)}...")

        for cond_idx in range(7):
            fc = all_fc[subj_idx, cond_idx]
            features = extract_all_features(fc)

            features['subject'] = subject
            features['condition'] = cond_idx
            features['label'] = 1 if cond_idx in CONSCIOUS_CONDITIONS else 0

            all_features.append(features)

    print(f"  ✓ Extracted features from {len(all_features)} samples")

    # Step 3: Prepare ML data
    print("\nStep 3/4: Preparing ML dataset...")
    X = np.array(
        [[f['isd'], f['efficiency'], f['clustering']]
         for f in all_features]
    )
    y = np.array([f['label'] for f in all_features])
    subject_ids = np.array([f['subject'] for f in all_features])

    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Class 0 (unconscious): {np.sum(y==0)} samples")
    print(f"  Class 1 (conscious):   {np.sum(y==1)} samples")

    # Step 4: Train models
    print("\nStep 4/4: Training models with LOSO cross-validation...")
    print()

    if args.model == 'all':
        model_types = ['logistic', 'random_forest', 'svm']
    else:
        model_types = [args.model]

    results = compare_models(
        features=X,
        labels=y,
        subject_ids=subject_ids,
        model_types=model_types
    )

    # Save results
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"results_{timestamp}.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for model_type, result in results.items():
            # Convert per-subject metrics (may contain numpy types)
            per_subj = {}
            for subj, metrics_dict in result['per_subject_metrics'].items():
                per_subj[subj] = {
                    k: (int(v)
                        if isinstance(v, (np.integer,))
                        else float(v)
                        if isinstance(v, (np.floating, float))
                        else v)
                    for k, v in metrics_dict.items()
                }

            json_results[model_type] = {
                'metrics': {
                    k: (float(v)
                        if isinstance(v, (np.floating, float))
                        else v.tolist())
                    for k, v in result['metrics'].items()
                    if k != 'confusion_matrix'
                },
                'confusion_matrix': (
                    result['metrics']['confusion_matrix']
                    .tolist()
                ),
                'per_subject_metrics': per_subj
            }

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
