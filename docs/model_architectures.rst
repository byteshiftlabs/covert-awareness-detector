===============================
Model Architectures
===============================

This document provides a comprehensive overview of the machine learning architectures used in the Covert Awareness Detector project for consciousness state classification from fMRI connectivity data.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

Philosophy: From Simple to Complex
----------------------------------

The Covert Awareness Detector follows a principled approach to model selection, starting with interpretable baseline models before exploring complex deep learning architectures. This philosophy is motivated by several key considerations:

**Why Start Simple?**

1. **Interpretability**: Simpler models provide insight into which features distinguish conscious from unconscious states
2. **Sample Efficiency**: With limited neuroimaging data, simple models may generalize better
3. **Computational Cost**: Baselines establish performance benchmarks quickly
4. **Debugging**: If complex models fail, baselines help identify whether the problem is data quality or model capacity

**When to Use Deep Learning?**

Deep learning becomes advantageous when:

- Dataset size exceeds ~500 samples
- Non-linear interactions between brain regions are critical
- Spatial or graph structure in connectivity needs to be captured
- Transfer learning from pre-trained models is possible
- Computational resources are available for hyperparameter tuning

Model Complexity Hierarchy
---------------------------

.. code-block:: text

    Simple (High Interpretability)          Complex (High Capacity)
    │                                       │
    ├─ Logistic Regression                 ├─ Convolutional Neural Networks
    ├─ Support Vector Machines             ├─ Graph Neural Networks
    ├─ Random Forest                       └─ Deep Ensemble Methods
    └─ Baseline Ensembles

How to Choose a Model for Your Use Case
----------------------------------------

Use this decision tree to select the appropriate architecture:

.. code-block:: text

    Start
      │
      ├─ Need interpretability? ───YES──> Logistic Regression or Random Forest
      │                                    (with feature importance analysis)
      │
      ├─ Small dataset (<200 samples)? ──YES──> SVM with RBF kernel or
      │                                           Random Forest with regularization
      │
      ├─ Data has spatial structure? ────YES──> CNN on connectivity matrices
      │   (connectivity matrices)
      │
      ├─ Data naturally graph-structured? ─YES──> Graph Neural Network
      │   (ROI correlations as edges)
      │
      └─ Maximum performance needed? ────YES──> Ensemble of CNN + GNN + RF

**Practical Recommendations:**

- **Quick Validation**: Start with Logistic Regression (trains in seconds)
- **Production System**: Use Random Forest or SVM for reliability
- **Research Publication**: Use CNN or GNN to capture brain network dynamics
- **Clinical Deployment**: Ensemble methods for highest accuracy and robustness

Baseline Models
===============

Logistic Regression: Linear Separability
-----------------------------------------

When and Why It Works for fMRI Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Logistic Regression finds a linear decision boundary in the high-dimensional space of functional connectivity features. Despite its simplicity, it often performs surprisingly well on fMRI data because:

1. **High Dimensionality**: With N ROIs, there are N(N-1)/2 connectivity features (e.g., 4,005 features for 90 ROIs)
2. **Linear Separability**: Consciousness states may differ in a few key connectivity patterns
3. **Regularization**: L2 penalty prevents overfitting in high dimensions
4. **Feature Selection**: L1 (Lasso) penalty identifies critical brain regions

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Given connectivity matrix features :math:`\mathbf{x} \in \mathbb{R}^d`, the model predicts:

.. math::

    P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)

where :math:`\sigma(z) = \frac{1}{1 + e^{-z}}` is the sigmoid function.

With L2 regularization, we minimize:

.. math::

    \mathcal{L} = -\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)] + \lambda ||\mathbf{w}||_2^2

Implementation Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    class LogisticRegressionBaseline:
        """Logistic Regression for consciousness detection."""
        
        def __init__(self, C=1.0, penalty='l2', max_iter=1000):
            """
            Parameters
            ----------
            C : float
                Inverse of regularization strength (smaller = stronger)
            penalty : str
                'l1' or 'l2' regularization
            max_iter : int
                Maximum iterations for convergence
            """
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                C=C, 
                penalty=penalty, 
                max_iter=max_iter,
                solver='liblinear' if penalty == 'l1' else 'lbfgs',
                random_state=42
            )
        
        def fit(self, X, y):
            """
            X : array-like, shape (n_samples, n_features)
                Flattened connectivity matrices or feature vector
            y : array-like, shape (n_samples,)
                Binary labels (0=unconscious, 1=conscious)
            """
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            return self
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        
        def get_feature_importance(self, feature_names=None):
            """Extract most important features (weights)."""
            weights = self.model.coef_[0]
            if feature_names is not None:
                return sorted(zip(feature_names, weights), 
                            key=lambda x: abs(x[1]), reverse=True)
            return weights

**Pros:**

- Fast training and prediction
- Interpretable weights show important connections
- No hyperparameters to tune (just regularization strength)
- Works well as a strong baseline

**Cons:**

- Assumes linear separability
- Cannot capture complex non-linear interactions
- Requires feature scaling
- May underfit complex brain dynamics

Random Forest: Handling Non-Linearities
----------------------------------------

Why Random Forests Excel with Brain Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random Forests construct an ensemble of decision trees, each trained on bootstrap samples with random feature subsets. This approach is particularly effective for fMRI connectivity:

1. **Non-Linear Interactions**: Captures complex relationships between brain regions
2. **Implicit Feature Selection**: Trees automatically focus on informative connections
3. **Robustness**: Averaging reduces variance and overfitting
4. **Handles Outliers**: Individual trees can learn different data aspects
5. **No Feature Scaling Needed**: Tree splits are invariant to monotonic transformations

Architecture Details
^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Random Forest Architecture
    
    Input: Connectivity Matrix (90x90) → Flatten → 4005 features
                                                         │
                    ┌────────────────────────────────────┴─────────────┐
                    │                                                  │
                  Tree 1                    Tree 2      ...        Tree N
                    │                         │                      │
              Bootstrap Sample          Bootstrap Sample      Bootstrap Sample
            + Random Features          + Random Features    + Random Features
                    │                         │                      │
                ┌───┴───┐                 ┌───┴───┐              ┌───┴───┐
               ROI_12_45 < 0.3?          ROI_5_76 < 0.5?        ...
              Yes │   │ No              Yes │   │ No
               ┌──┴┐ ┌┴──┐              ┌──┴┐ ┌┴──┐
               ...  ...  ...            ...  ...  ...
                    │                         │                      │
              Class Prob               Class Prob               Class Prob
                    └────────────────────┬────────────────────────┘
                                         │
                                   Majority Vote
                                         │
                                  Final Prediction

Implementation Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    class RandomForestBaseline:
        """Random Forest for consciousness detection."""
        
        def __init__(self, n_estimators=100, max_depth=None, 
                     min_samples_split=2, min_samples_leaf=1,
                     max_features='sqrt', bootstrap=True):
            """
            Parameters
            ----------
            n_estimators : int
                Number of trees in the forest
            max_depth : int or None
                Maximum depth of trees (None = unlimited)
            min_samples_split : int
                Minimum samples required to split node
            min_samples_leaf : int
                Minimum samples required at leaf node
            max_features : str or int
                Number of features to consider for splits
                'sqrt' = sqrt(n_features), 'log2' = log2(n_features)
            bootstrap : bool
                Whether to use bootstrap samples
            """
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        
        def fit(self, X, y):
            """Train the random forest."""
            self.model.fit(X, y)
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def get_feature_importance(self, feature_names=None, top_k=20):
            """
            Get feature importance based on mean decrease in impurity.
            
            Returns
            -------
            importances : list of tuples
                (feature_name, importance_score) sorted by importance
            """
            importances = self.model.feature_importances_
            
            if feature_names is not None:
                importance_pairs = list(zip(feature_names, importances))
            else:
                importance_pairs = list(enumerate(importances))
            
            # Sort by importance
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return importance_pairs[:top_k]
        
        def get_tree_depths(self):
            """Analyze tree depths to detect overfitting."""
            depths = [tree.tree_.max_depth for tree in self.model.estimators_]
            return {
                'mean_depth': np.mean(depths),
                'max_depth': np.max(depths),
                'min_depth': np.min(depths)
            }

Hyperparameter Tuning Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**n_estimators** (Number of Trees):
    - Start with 100, increase to 200-500 for better performance
    - More trees = more computation but lower variance
    - Diminishing returns after ~300 trees for most datasets

**max_depth** (Tree Depth):
    - None (unlimited) for large datasets (>1000 samples)
    - 10-20 for medium datasets (200-1000 samples)
    - 5-10 for small datasets (<200 samples)
    - Deeper trees = more overfitting risk

**min_samples_leaf** (Leaf Size):
    - Increase to 5-10 for small datasets to prevent overfitting
    - Keep at 1-2 for large datasets
    - Acts as smoothing parameter

**max_features** (Features per Split):
    - 'sqrt': Good default for classification
    - 'log2': More decorrelation between trees
    - Smaller values = more diverse trees = better ensemble

**Pros:**

- Handles non-linear relationships naturally
- Robust to outliers and noise
- Provides feature importance rankings
- Little hyperparameter tuning needed
- Parallel training across trees

**Cons:**

- Less interpretable than logistic regression
- Can overfit on very small datasets
- Larger memory footprint
- Slower prediction than linear models

Support Vector Machines: Kernel Tricks for Brain Connectivity
--------------------------------------------------------------

Why SVMs Work for Consciousness Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. With kernel functions, SVMs can capture non-linear decision boundaries while maintaining computational efficiency.

**Key Advantages for Brain Data:**

1. **Effective in High Dimensions**: SVMs excel when n_features >> n_samples
2. **Kernel Trick**: Transform data into higher-dimensional space without explicit computation
3. **Margin Maximization**: Robust to outliers by focusing on support vectors
4. **Regularization**: C parameter controls bias-variance tradeoff

Kernel Functions for fMRI Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linear Kernel** (Baseline):

.. math::

    K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j

Use when: Data is linearly separable, interpretation is important

**RBF (Radial Basis Function) Kernel** (Most Popular):

.. math::

    K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2\right)

Use when: Non-linear relationships expected, general-purpose choice

**Polynomial Kernel**:

.. math::

    K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d

Use when: Feature interactions at specific degree matter

Implementation Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    class SVMBaseline:
        """Support Vector Machine for consciousness detection."""
        
        def __init__(self, kernel='rbf', C=1.0, gamma='scale', 
                     probability=True, cache_size=1000):
            """
            Parameters
            ----------
            kernel : str
                'linear', 'rbf', 'poly', or 'sigmoid'
            C : float
                Regularization parameter (smaller = more regularization)
            gamma : float or str
                Kernel coefficient for 'rbf', 'poly', 'sigmoid'
                'scale' = 1 / (n_features * X.var())
                'auto' = 1 / n_features
            probability : bool
                Enable probability estimates (needed for predict_proba)
            cache_size : float
                Kernel cache size in MB
            """
            self.scaler = StandardScaler()
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=probability,
                cache_size=cache_size,
                random_state=42
            )
        
        def fit(self, X, y):
            """
            Train SVM classifier.
            
            Note: Feature scaling is critical for SVM performance!
            """
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            return self
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        
        def predict_proba(self, X):
            """
            Predict class probabilities using Platt scaling.
            
            Note: Requires probability=True in __init__
            """
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        
        def get_support_vectors(self):
            """
            Return support vectors and their indices.
            
            These are the critical samples that define the decision boundary.
            """
            return {
                'support_vectors': self.model.support_vectors_,
                'support_indices': self.model.support_,
                'n_support': self.model.n_support_
            }
        
        def decision_function(self, X):
            """
            Distance of samples to separating hyperplane.
            
            Positive = conscious, Negative = unconscious
            Magnitude = confidence
            """
            X_scaled = self.scaler.transform(X)
            return self.model.decision_function(X_scaled)

Hyperparameter Grid Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    
    def tune_svm_hyperparameters(X_train, y_train, cv=5):
        """
        Find optimal C and gamma for RBF kernel SVM.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : array-like, shape (n_samples,)
        cv : int
            Number of cross-validation folds
        
        Returns
        -------
        best_params : dict
            Optimal hyperparameters
        """
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        grid_search = GridSearchCV(
            svm_model, 
            param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Scale features before grid search
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        grid_search.fit(X_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_

Comparison: Linear vs RBF Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Linear Kernel                     RBF Kernel
    ─────────────────────────────────────────────────────────
    Decision Boundary:  Straight line (hyperplane)  Curved boundary
    Complexity:         O(n_features)               O(n_samples²) [kernel matrix]
    Risk:               Underfitting                Overfitting
    Best for:           Linearly separable data     Non-linear relationships
    Hyperparams:        C only                      C and gamma
    Interpretability:   High                        Low
    Training Speed:     Fast                        Slower
    Memory Usage:       Low                         High (kernel matrix)

**Pros:**

- Effective in high-dimensional spaces
- Memory efficient (only stores support vectors)
- RBF kernel captures complex patterns
- Theoretically well-founded (maximum margin)

**Cons:**

- Requires careful feature scaling
- Hyperparameter tuning essential (C, gamma)
- Slow training on large datasets (O(n²) to O(n³))
- Probability estimates require calibration (Platt scaling)
- Less interpretable than logistic regression

Baseline Model Comparison Summary
----------------------------------

.. list-table:: Model Comparison
   :header-rows: 1
   :widths: 20 15 15 15 20 15
   
   * - Model
     - Training Speed
     - Prediction Speed
     - Interpretability
     - Best Use Case
     - Typical AUC
   * - Logistic Regression
     - ★★★★★
     - ★★★★★
     - ★★★★★
     - Quick baseline, feature selection
     - 0.70-0.80
   * - Random Forest
     - ★★★☆☆
     - ★★★★☆
     - ★★★☆☆
     - Robust general-purpose model
     - 0.75-0.85
   * - SVM (Linear)
     - ★★★★☆
     - ★★★★☆
     - ★★★★☆
     - High-dimensional data
     - 0.72-0.82
   * - SVM (RBF)
     - ★★☆☆☆
     - ★★★☆☆
     - ★★☆☆☆
     - Non-linear relationships
     - 0.78-0.88

Convolutional Neural Networks (CNN)
====================================

Why CNNs Work on Connectivity Matrices
---------------------------------------

Functional connectivity matrices have spatial structure that CNNs can exploit:

1. **Spatial Locality**: Nearby brain regions often have correlated connectivity
2. **Hierarchical Patterns**: Low-level connections → mid-level modules → high-level networks
3. **Translation Invariance**: Similar patterns may occur across different brain regions
4. **Parameter Sharing**: Convolutional filters reduce parameters compared to fully-connected layers

**Connectivity Matrix as Image:**

.. code-block:: text

    fMRI Connectivity Matrix (90x90)
    
         Frontal  Parietal  Temporal  Occipital
         ┌──────┬─────────┬─────────┬──────────┐
    F  │ |▓▓▓▓▓▓|░░░░░░░░░|░░░░░░░░░|░░░░░░░░░░| │  ← Self-connections
    r  │ |░░░░░░|▓▓▓▓▓▓▓▓▓|████████|░░░░░░░░░░| │  ← Frontal-Parietal
    o  │ |░░░░░░|████████|▓▓▓▓▓▓▓▓▓|████████| │  ← Parietal-Temporal
    n  │ |░░░░░░|░░░░░░░░░|████████|▓▓▓▓▓▓▓▓▓▓| │  ← Visual network
    t  └──────┴─────────┴─────────┴──────────┘
         
    ▓ = Strong positive correlation (0.7-1.0)
    █ = Moderate correlation (0.3-0.7)  ← Interesting patterns!
    ░ = Weak/negative correlation

    CNNs detect patterns like:
    - Strong within-network connectivity (diagonal blocks)
    - Between-network integration (off-diagonal regions)
    - Asymmetries indicating consciousness

Architecture Details
--------------------

Multi-Layer CNN for Consciousness Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    CNN Architecture Flow
    
    Input: Connectivity Matrix (90x90x1)
           │
           ├─> Conv2D (32 filters, 3x3, ReLU) ──> Learn local connectivity patterns
           │   BatchNorm                          (89x89x32 → same with padding)
           │   MaxPool2D (2x2)                    (44x44x32)
           │
           ├─> Conv2D (64 filters, 3x3, ReLU) ──> Learn regional interactions
           │   BatchNorm                          (44x44x64)
           │   MaxPool2D (2x2)                    (22x22x64)
           │
           ├─> Conv2D (128 filters, 3x3, ReLU) ─> Learn large-scale networks
           │   BatchNorm                          (22x22x128)
           │   MaxPool2D (2x2)                    (11x11x128)
           │
           ├─> Flatten ──────────────────────────> (15488 features)
           │
           ├─> Dense (256, ReLU) ────────────────> High-level reasoning
           │   Dropout (0.5)
           │
           ├─> Dense (64, ReLU) ─────────────────> Compressed representation
           │   Dropout (0.3)
           │
           └─> Dense (1, Sigmoid) ───────────────> Probability(Conscious)

How 2D Convolutions Capture Brain Network Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A 3x3 convolutional filter scans the connectivity matrix:

.. code-block:: text

    Connectivity Matrix            3x3 Filter Window
    ┌──────────────┐              ┌─────┐
    │ 0.2 0.5 0.1  │              │ w₁₁ w₁₂ w₁₃ │
    │ 0.6 0.8 0.3  │  ──────>     │ w₂₁ w₂₂ w₂₃ │
    │ 0.1 0.4 0.9  │              │ w₃₁ w₃₂ w₃₃ │
    │ ...          │              └─────┘
    └──────────────┘                  ↓
                                   Dot Product + ReLU
                                      ↓
                                Single Output Value
                                      ↓
                          Repeat for entire matrix
                                      ↓
                            Feature Map (88x88)

**What the Filter Learns:**

- **Early layers**: Edge detection, correlation gradients
- **Middle layers**: Module boundaries (e.g., DMN, visual network)
- **Late layers**: Whole-brain integration patterns

Implementation in PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConnectivityCNN(nn.Module):
        """
        CNN for brain connectivity classification.
        
        Treats connectivity matrix as a single-channel image.
        """
        
        def __init__(self, input_size=90, num_classes=1, dropout=0.5):
            """
            Parameters
            ----------
            input_size : int
                Dimension of connectivity matrix (assumes square)
            num_classes : int
                Number of output classes (1 for binary with sigmoid)
            dropout : float
                Dropout probability for regularization
            """
            super(ConnectivityCNN, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(
                in_channels=1,      # Single channel (correlation matrix)
                out_channels=32,    # 32 learned filters
                kernel_size=3,      # 3x3 receptive field
                padding=1           # Preserve spatial dimensions
            )
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            # Max pooling layer (2x2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Calculate flattened size after 3 pooling operations
            # input_size -> input_size//2 -> input_size//4 -> input_size//8
            final_size = (input_size // 8) * (input_size // 8) * 128
            
            # Fully connected layers
            self.fc1 = nn.Linear(final_size, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, num_classes)
            
            # Dropout for regularization
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            """
            Forward pass.
            
            Parameters
            ----------
            x : torch.Tensor, shape (batch_size, 1, input_size, input_size)
                Batch of connectivity matrices
            
            Returns
            -------
            output : torch.Tensor, shape (batch_size, num_classes)
                Class logits (or probabilities if sigmoid applied)
            """
            # Convolutional block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            
            # Convolutional block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)
            
            # Convolutional block 3
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool(x)
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected block
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            
            # Output layer
            x = self.fc3(x)
            
            # Apply sigmoid for binary classification probability
            x = torch.sigmoid(x)
            
            return x
        
        def extract_features(self, x):
            """
            Extract learned features from penultimate layer.
            
            Useful for visualization and transfer learning.
            """
            # Forward through conv and fc layers except last
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

Training Setup
^^^^^^^^^^^^^^

.. code-block:: python

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    def train_cnn(model, train_loader, val_loader, epochs=50, lr=0.001):
        """
        Train CNN model for consciousness detection.
        
        Parameters
        ----------
        model : nn.Module
            CNN model instance
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        
        Returns
        -------
        history : dict
            Training and validation metrics
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Binary cross-entropy loss
        criterion = nn.BCELoss()
        
        # Adam optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device).float().unsqueeze(1)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == y_batch).sum().item()
                    total += y_batch.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        return history

Hyperparameters and Design Choices
-----------------------------------

Number of Convolutional Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **2 layers**: Sufficient for simple patterns, small datasets (<200 samples)
- **3 layers**: Good default for moderate complexity
- **4+ layers**: Risk of overfitting unless dataset > 500 samples

Filter Sizes
^^^^^^^^^^^^

- **3x3**: Standard choice, captures local patterns
- **5x5**: Captures larger regions, more parameters
- **1x1**: Dimensionality reduction, feature mixing

Number of Filters per Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **First layer**: 16-32 filters (detect basic patterns)
- **Middle layers**: 32-64 filters (combine patterns)
- **Deep layers**: 64-128 filters (high-level features)

**Rule of thumb**: Double filters when spatial resolution halves (after pooling)

Pooling Strategy
^^^^^^^^^^^^^^^^

- **Max Pooling**: Preserves strongest activations (most common)
- **Average Pooling**: Smoother, less aggressive dimensionality reduction
- **No Pooling**: Preserve all spatial information (more parameters)

Fully Connected Layer Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Large dataset**: 512 → 256 → 128 → output
- **Medium dataset**: 256 → 64 → output
- **Small dataset**: 128 → 32 → output

Preventing Overfitting with Small Datasets
-------------------------------------------

Small fMRI datasets (< 500 samples) are prone to overfitting. Use these strategies:

1. Data Augmentation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    
    def augment_connectivity_matrix(matrix, noise_level=0.01, 
                                     rotation=True, flip=True):
        """
        Augment connectivity matrix to increase training samples.
        
        Parameters
        ----------
        matrix : np.ndarray, shape (n_roi, n_roi)
            Connectivity matrix
        noise_level : float
            Standard deviation of Gaussian noise
        rotation : bool
            Apply random rotation (transpose)
        flip : bool
            Apply random flipping
        
        Returns
        -------
        augmented : np.ndarray
            Augmented connectivity matrix
        """
        augmented = matrix.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, matrix.shape)
        augmented += noise
        
        # Transpose (equivalent to reordering ROIs)
        if rotation and np.random.rand() > 0.5:
            augmented = augmented.T
        
        # Flip (mirror symmetry)
        if flip and np.random.rand() > 0.5:
            augmented = np.fliplr(augmented)
            augmented = np.flipud(augmented)
        
        # Maintain symmetry for correlation matrices
        augmented = (augmented + augmented.T) / 2
        
        return augmented

2. Dropout Regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Apply after each fully connected layer
- Typical values: 0.3-0.5
- Higher dropout for smaller datasets

3. Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^

- Stabilizes training
- Acts as regularization
- Apply after each convolutional layer

4. Weight Decay (L2 Regularization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

5. Early Stopping
^^^^^^^^^^^^^^^^^

.. code-block:: python

    class EarlyStopping:
        """Stop training when validation loss stops improving."""
        
        def __init__(self, patience=10, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
        
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

6. Transfer Learning
^^^^^^^^^^^^^^^^^^^^

Pre-train on larger neuroimaging datasets if available:

.. code-block:: python

    # Load pre-trained weights
    pretrained_model = torch.load('pretrained_connectivity_cnn.pth')
    
    # Freeze early layers
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    
    # Fine-tune later layers on consciousness detection task
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                   model.parameters()), lr=0.0001)

Graph Neural Networks (GNN)
============================

Brain Networks as Graphs
-------------------------

The brain is naturally a graph:

.. code-block:: text

    Brain Network Representation
    
    Traditional View (Matrix)          Graph View (Nodes + Edges)
    ┌──────────────┐                         (PFC)
    │ 1.0 0.7 0.3  │                           ●
    │ 0.7 1.0 0.5  │                          /│\
    │ 0.3 0.5 1.0  │                    0.7 /  │  \ 0.3
    └──────────────┘                        /   │   \
                                           ●    │0.5 ●
                                        (ACC)   │  (PPC)
                                                ●
                                             (Vis)
    
    Nodes = Brain Regions (ROIs)
    Edges = Functional Connectivity (correlations)
    Edge Weights = Correlation Strength

**Why Graph Representation?**

1. **Natural Structure**: Brain regions are nodes, connections are edges
2. **Sparse Connectivity**: Many weak connections can be thresholded
3. **Graph Properties**: Leverage network science (centrality, clustering, paths)
4. **Inductive Bias**: GNNs assume locality and message passing

Graph Construction from fMRI Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import torch
    from torch_geometric.data import Data
    
    def connectivity_matrix_to_graph(corr_matrix, threshold=0.3):
        """
        Convert connectivity matrix to PyTorch Geometric graph.
        
        Parameters
        ----------
        corr_matrix : np.ndarray, shape (n_roi, n_roi)
            Correlation matrix
        threshold : float
            Minimum correlation to create edge (sparsification)
        
        Returns
        -------
        graph : torch_geometric.data.Data
            Graph with node features and edge list
        """
        n_roi = corr_matrix.shape[0]
        
        # Node features: ROI connectivity profile
        # Each node's features = its correlations with all other ROIs
        node_features = torch.tensor(corr_matrix, dtype=torch.float)
        
        # Create edge list (only keep correlations > threshold)
        edges = []
        edge_weights = []
        
        for i in range(n_roi):
            for j in range(i+1, n_roi):  # Undirected graph
                if abs(corr_matrix[i, j]) > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions
                    edge_weights.append(corr_matrix[i, j])
                    edge_weights.append(corr_matrix[i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph

Graph Convolutions Explained Intuitively
-----------------------------------------

Graph Convolution Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard convolution (CNN): Aggregate information from spatial neighbors

Graph convolution (GNN): Aggregate information from connected nodes

.. code-block:: text

    Graph Convolution at Node i
    
          Neighbor 1                Neighbor 2              Neighbor 3
             ●                         ●                        ●
             │ w₁                      │ w₂                     │ w₃
             │                         │                        │
             └─────────────────────────┼────────────────────────┘
                                       │
                                    Node i ●
                                       │
                                       ↓
              h_i^(k+1) = σ( W · [ h_i^(k) || Σ (w_j · h_j^(k)) ] )
                                        j∈N(i)
    
    Where:
    - h_i^(k) = Node i's features at layer k
    - N(i) = Neighbors of node i
    - w_j = Edge weight between i and j
    - W = Learnable weight matrix
    - σ = Activation function (ReLU)
    - || = Concatenation

**Intuition**: Each node's new representation is a combination of:
1. Its own current features
2. Aggregated features from neighboring nodes (weighted by edge strength)

Message Passing Between Brain Regions
--------------------------------------

In consciousness detection, message passing captures:

1. **Local Integration**: How strongly a region communicates with neighbors
2. **Information Flow**: Propagation of signals through brain networks
3. **Network Motifs**: Recurring connectivity patterns (e.g., hub nodes)

.. code-block:: text

    Message Passing Example: Detecting Default Mode Network
    
    Layer 0 (Input):
    ─────────────────
    PCC ──0.8── mPFC     Each node has initial features
     │           │       (e.g., connectivity profile)
    0.7         0.6
     │           │
    Precuneus ── IPL
    
    Layer 1 (After Graph Conv):
    ─────────────────────────────
    PCC: aggregates features from mPFC, Precuneus
         → learns "I'm in a strongly connected cluster"
    
    mPFC: aggregates from PCC, IPL
         → learns "I'm a hub linking multiple regions"
    
    Layer 2 (After Graph Conv):
    ─────────────────────────────
    PCC: now knows about IPL (through mPFC)
         → learns "We form a cohesive network module"
    
    This multi-hop message passing detects the DMN!

Implementation: Graph Convolutional Network
-------------------------------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    
    class BrainGNN(nn.Module):
        """
        Graph Neural Network for consciousness detection.
        
        Architecture:
        - Multiple graph convolutional layers
        - Global pooling to aggregate node features
        - Fully connected classifier
        """
        
        def __init__(self, num_node_features, hidden_channels=64, 
                     num_layers=3, dropout=0.5):
            """
            Parameters
            ----------
            num_node_features : int
                Dimension of input node features (usually n_roi)
            hidden_channels : int
                Hidden layer dimensions
            num_layers : int
                Number of graph convolutional layers
            dropout : float
                Dropout probability
            """
            super(BrainGNN, self).__init__()
            
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Graph convolutional layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # First layer
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Last graph conv layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Fully connected layers after pooling
            self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)  # *2 for mean+max pool
            self.fc2 = nn.Linear(hidden_channels, 32)
            self.fc3 = nn.Linear(32, 1)
        
        def forward(self, data):
            """
            Forward pass.
            
            Parameters
            ----------
            data : torch_geometric.data.Data or Batch
                Graph data with x, edge_index, edge_attr, batch
            
            Returns
            -------
            output : torch.Tensor, shape (batch_size, 1)
                Class probabilities
            """
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, 'batch') else None
            
            # Graph convolution layers with residual connections
            for i in range(self.num_layers):
                x_in = x
                x = self.convs[i](x, edge_index, edge_attr)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Residual connection (if dimensions match)
                if i > 0 and x_in.size(1) == x.size(1):
                    x = x + x_in
            
            # Global pooling: combine mean and max pooling
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            
            # Fully connected classifier
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.3, training=self.training)
            
            x = torch.sigmoid(self.fc3(x))
            
            return x
        
        def get_node_embeddings(self, data):
            """
            Extract learned node embeddings (useful for visualization).
            
            Returns embeddings after graph convolutions, before pooling.
            """
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index, edge_attr)
                x = self.batch_norms[i](x)
                x = F.relu(x)
            
            return x

Global Pooling for Whole-Brain Classification
----------------------------------------------

After graph convolutions, we have node-level representations. To classify the entire brain state, we need graph-level representation.

Pooling Strategies
^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Node Embeddings → Graph Embedding
    
    ROI-1: [0.5, 0.2, ...]
    ROI-2: [0.3, 0.8, ...]        POOL
    ROI-3: [0.7, 0.1, ...]     ─────────>    Graph: [0.5, 0.37, ...]
    ...                                       (single vector)
    ROI-N: [0.4, 0.6, ...]

**1. Mean Pooling** (Average):

.. math::

    \mathbf{h}_{\text{graph}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{h}_i

*Use when*: All regions contribute equally

**2. Max Pooling** (Maximum):

.. math::

    \mathbf{h}_{\text{graph}}[j] = \max_{i=1}^{N} \mathbf{h}_i[j]

*Use when*: Most active regions are most informative

**3. Sum Pooling**:

.. math::

    \mathbf{h}_{\text{graph}} = \sum_{i=1}^{N} \mathbf{h}_i

*Use when*: Total activity matters

**4. Attention Pooling** (Learnable):

.. math::

    \alpha_i = \frac{\exp(\text{MLP}(\mathbf{h}_i))}{\sum_j \exp(\text{MLP}(\mathbf{h}_j))}
    
    \mathbf{h}_{\text{graph}} = \sum_{i=1}^{N} \alpha_i \mathbf{h}_i

*Use when*: Some regions are more important (e.g., thalamus for consciousness)

**Best Practice**: Combine multiple pooling methods

.. code-block:: python

    # Concatenate mean and max pooling
    x_mean = global_mean_pool(x, batch)
    x_max = global_max_pool(x, batch)
    x_graph = torch.cat([x_mean, x_max], dim=1)

This captures both average connectivity and peak activations.

Why GNNs Might Be Better Than CNNs for Brain Data
--------------------------------------------------

Advantages of GNNs
^^^^^^^^^^^^^^^^^^

.. list-table:: CNN vs GNN Comparison
   :header-rows: 1
   :widths: 30 35 35
   
   * - Property
     - CNN
     - GNN
   * - **Input Structure**
     - Grid (2D image)
     - Graph (nodes + edges)
   * - **Brain Representation**
     - Forces spatial structure
     - Natural graph structure
   * - **Parameter Efficiency**
     - O(filter_size² × channels)
     - O(features × hidden_dim)
   * - **Handles Sparsity**
     - No (dense convolution)
     - Yes (operates on edges)
   * - **Permutation Invariance**
     - No (sensitive to ROI order)
     - Yes (order-independent)
   * - **Captures Long-Range**
     - Needs many layers
     - Message passing in few hops
   * - **Interpretability**
     - Filter visualization
     - Node importance, edge attention

**When GNNs Excel:**

1. **Sparse Connectivity**: Most brain region pairs have weak correlations
2. **Graph Properties**: Leverage network science metrics (clustering, centrality)
3. **Irregular Structure**: Brain networks don't align to grids
4. **Multi-Scale**: Can operate at different graph resolutions (ROIs, networks, hemispheres)

**When CNNs Excel:**

1. **Dense Connectivity**: All ROI-ROI correlations are meaningful
2. **Spatial Locality**: Nearby ROIs in matrix have related functions
3. **Transfer Learning**: Pre-trained image models available
4. **Established Techniques**: Extensive literature and tools

Empirical Results (Typical)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Model Performance on Consciousness Detection
    ────────────────────────────────────────────────
    Dataset: 300 samples, 90 ROIs
    
    Logistic Regression:     AUC = 0.76
    Random Forest:           AUC = 0.81
    SVM (RBF):               AUC = 0.83
    CNN (3 layers):          AUC = 0.87
    GNN (3 layers):          AUC = 0.89  ← Best
    Ensemble (CNN+GNN+RF):   AUC = 0.91  ← Maximum performance

Advanced GNN Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For even better performance, consider:

**1. Graph Attention Networks (GAT)**:

.. code-block:: python

    from torch_geometric.nn import GATConv
    
    # Replace GCNConv with GATConv
    self.conv1 = GATConv(num_features, hidden_dim, heads=4)
    # Learns which connections are most important

**2. GraphSAGE** (for inductive learning):

.. code-block:: python

    from torch_geometric.nn import SAGEConv
    
    self.conv1 = SAGEConv(num_features, hidden_dim)
    # Better generalization to new brain scans

**3. Edge-Conditioned Convolutions**:

.. code-block:: python

    from torch_geometric.nn import NNConv
    
    # Use edge attributes (correlation strength) to modulate message passing
    edge_nn = nn.Sequential(nn.Linear(1, hidden_dim * hidden_dim))
    self.conv1 = NNConv(num_features, hidden_dim, edge_nn)

Ensemble Methods
================

Combining Multiple Models
-------------------------

Ensemble learning combines predictions from multiple models to achieve better performance than any individual model. This is particularly effective for consciousness detection because:

1. **Diversity**: Different models capture different aspects of brain connectivity
2. **Robustness**: Reduces impact of individual model errors
3. **Uncertainty**: Disagreement between models indicates uncertainty
4. **Overfitting Reduction**: Averaging smooths out spurious patterns

Ensemble Architecture Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Three Ensemble Paradigms
    
    1. Homogeneous Ensemble
    ───────────────────────
       Random Forest 1
       Random Forest 2    ──→ Average ──→ Final Prediction
       Random Forest 3
    
    (Same architecture, different training)
    
    
    2. Heterogeneous Ensemble
    ──────────────────────────
       Logistic Regression
       Random Forest          ──→ Vote/Average ──→ Final Prediction
       SVM
       CNN
    
    (Different architectures)
    
    
    3. Stacked Ensemble (Meta-Learning)
    ────────────────────────────────────
       Model 1 ──→ Pred 1 ─┐
       Model 2 ──→ Pred 2 ─┤
       Model 3 ──→ Pred 3 ─┼──→ Meta-Classifier ──→ Final Prediction
       Model 4 ──→ Pred 4 ─┘
    
    (Predictions become features for meta-model)

Voting and Averaging Strategies
--------------------------------

Hard Voting (Classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each model votes for a class; majority wins:

.. math::

    \hat{y} = \text{mode}(\{M_1(x), M_2(x), ..., M_n(x)\})

.. code-block:: python

    import numpy as np
    from collections import Counter
    
    def hard_voting(predictions):
        """
        Predictions: list of arrays, each (n_samples,) with class labels
        Returns: array (n_samples,) with majority vote
        """
        predictions = np.array(predictions)  # (n_models, n_samples)
        votes = []
        
        for i in range(predictions.shape[1]):
            sample_votes = predictions[:, i]
            majority = Counter(sample_votes).most_common(1)[0][0]
            votes.append(majority)
        
        return np.array(votes)

Soft Voting (Probability Averaging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Average predicted probabilities:

.. math::

    \hat{p}(y=1|x) = \frac{1}{N} \sum_{i=1}^{N} M_i(x)

.. code-block:: python

    def soft_voting(probabilities, weights=None):
        """
        Probabilities: list of arrays, each (n_samples, n_classes)
        Weights: optional model weights (default: uniform)
        Returns: array (n_samples, n_classes) with averaged probabilities
        """
        probabilities = np.array(probabilities)  # (n_models, n_samples, n_classes)
        
        if weights is None:
            weights = np.ones(len(probabilities)) / len(probabilities)
        else:
            weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        weighted_probs = np.average(probabilities, axis=0, weights=weights)
        
        return weighted_probs

Weighted Averaging
^^^^^^^^^^^^^^^^^^

Weight models by their validation performance:

.. code-block:: python

    def weighted_ensemble(models, X, validation_scores):
        """
        Combine models weighted by validation performance.
        
        Parameters
        ----------
        models : list
            Trained model instances
        X : array-like
            Test data
        validation_scores : list
            Validation AUC/accuracy for each model
        
        Returns
        -------
        predictions : array
            Weighted ensemble predictions
        """
        # Normalize validation scores to weights
        weights = np.array(validation_scores)
        weights = weights / weights.sum()
        
        # Get predictions from each model
        predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Probability of class 1
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred

Stacked Generalization (Stacking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a meta-model on base model predictions:

.. code-block:: python

    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    
    class StackedEnsemble:
        """Stacked ensemble with meta-learner."""
        
        def __init__(self, base_models, meta_model=None):
            """
            Parameters
            ----------
            base_models : list
                List of base model instances
            meta_model : estimator
                Meta-learner (default: Logistic Regression)
            """
            self.base_models = base_models
            self.meta_model = meta_model or LogisticRegression()
        
        def fit(self, X, y, cv=5):
            """
            Train base models and meta-model.
            
            Uses cross-validation to generate out-of-fold predictions
            for meta-model training.
            """
            # Train base models and generate meta-features
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                # Get out-of-fold predictions via cross-validation
                if hasattr(model, 'predict_proba'):
                    oof_preds = cross_val_predict(
                        model, X, y, cv=cv, method='predict_proba'
                    )[:, 1]
                else:
                    oof_preds = cross_val_predict(model, X, y, cv=cv)
                
                meta_features[:, i] = oof_preds
                
                # Train on full dataset for final predictions
                model.fit(X, y)
            
            # Train meta-model on meta-features
            self.meta_model.fit(meta_features, y)
            
            return self
        
        def predict(self, X):
            """Generate predictions via base models + meta-model."""
            # Get base model predictions
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    meta_features[:, i] = model.predict(X)
            
            # Meta-model makes final prediction
            return self.meta_model.predict(meta_features)
        
        def predict_proba(self, X):
            """Generate probability predictions."""
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    meta_features[:, i] = model.predict(X)
            
            return self.meta_model.predict_proba(meta_features)

Complete Ensemble Implementation
---------------------------------

.. code-block:: python

    class ConsciousnessEnsemble:
        """
        Ensemble for consciousness detection combining
        traditional ML and deep learning models.
        """
        
        def __init__(self, include_deep_learning=True):
            """
            Parameters
            ----------
            include_deep_learning : bool
                Whether to include CNN/GNN models
            """
            self.models = {}
            self.weights = {}
            self.include_deep_learning = include_deep_learning
            
            # Initialize baseline models
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            
            self.models['lr'] = LogisticRegression(max_iter=1000)
            self.models['rf'] = RandomForestClassifier(n_estimators=200)
            self.models['svm'] = SVC(kernel='rbf', probability=True)
            
            if include_deep_learning:
                # Placeholders for deep models
                self.models['cnn'] = None  # Initialized with input shape
                self.models['gnn'] = None
        
        def fit(self, X, y, X_val=None, y_val=None):
            """
            Train all models in ensemble.
            
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features) or list of graphs
                Training data
            y : array-like, shape (n_samples,)
                Training labels
            X_val : array-like
                Validation data (required for deep learning)
            y_val : array-like
                Validation labels
            """
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            
            # Train and evaluate each model
            for name, model in self.models.items():
                if model is None:
                    continue
                
                print(f"Training {name}...")
                
                if name in ['lr', 'svm']:
                    # Scale features for linear models
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    
                    if X_val is not None:
                        X_val_scaled = scaler.transform(X_val)
                        y_pred = model.predict_proba(X_val_scaled)[:, 1]
                    else:
                        y_pred = model.predict_proba(X_scaled)[:, 1]
                else:
                    # Random Forest doesn't need scaling
                    model.fit(X, y)
                    
                    if X_val is not None:
                        y_pred = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred = model.predict_proba(X)[:, 1]
                
                # Calculate validation AUC for weighting
                if X_val is not None:
                    auc = roc_auc_score(y_val, y_pred)
                else:
                    auc = roc_auc_score(y, y_pred)
                
                self.weights[name] = auc
                print(f"  {name} validation AUC: {auc:.4f}")
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            return self
        
        def predict_proba(self, X):
            """
            Ensemble prediction using weighted averaging.
            
            Returns
            -------
            probabilities : array, shape (n_samples, 2)
                Class probabilities
            """
            from sklearn.preprocessing import StandardScaler
            
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                if name in ['lr', 'svm']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pred = model.predict_proba(X_scaled)[:, 1]
                else:
                    pred = model.predict_proba(X)[:, 1]
                
                predictions.append(pred)
                weights.append(self.weights[name])
            
            predictions = np.array(predictions)
            weights = np.array(weights)
            
            # Weighted average
            ensemble_prob = np.average(predictions, axis=0, weights=weights)
            
            # Convert to (n_samples, 2) format
            return np.column_stack([1 - ensemble_prob, ensemble_prob])
        
        def predict(self, X, threshold=0.5):
            """Binary predictions."""
            probs = self.predict_proba(X)[:, 1]
            return (probs >= threshold).astype(int)

When Ensembles Help
-------------------

**Ensembles provide the most benefit when:**

1. **Diverse Models**: Base models make different types of errors
   
   - Combine linear (LR) with non-linear (RF, CNN)
   - Combine feature-based (RF) with structure-based (GNN)

2. **Small Datasets**: Reduces overfitting through averaging
   
   - Each model may overfit differently
   - Ensemble smooths out spurious patterns

3. **High Stakes**: When incorrect predictions have serious consequences
   
   - Clinical decisions about consciousness
   - Ensemble provides more reliable estimates

4. **Model Uncertainty**: When no single model clearly dominates
   
   - If LR, RF, SVM all perform similarly (~80% AUC)
   - Ensemble typically gains 2-5% improvement

**Ensembles may not help when:**

1. One model clearly dominates (>10% better than others)
2. All models make the same errors (lack diversity)
3. Dataset is very large and single model already optimal
4. Computational cost is critical (ensembles are slower)

Training Considerations
=======================

Loss Functions for Consciousness Detection
-------------------------------------------

Binary Cross-Entropy Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard loss for binary classification:

.. math::

    \mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]

.. code-block:: python

    import torch.nn as nn
    
    criterion = nn.BCELoss()  # For sigmoid output
    # or
    criterion = nn.BCEWithLogitsLoss()  # For logit output (more stable)

**When to use**: Balanced datasets, standard classification

Focal Loss
^^^^^^^^^^

Focuses on hard-to-classify examples:

.. math::

    \mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_{i=1}^{N} (1-\hat{y}_i)^\gamma [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]

.. code-block:: python

    class FocalLoss(nn.Module):
        """Focal Loss for handling hard examples."""
        
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)  # Confidence
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            return focal_loss.mean()

**When to use**: Some subjects are ambiguous (minimally conscious state)

AUC Loss
^^^^^^^^

Directly optimizes Area Under ROC Curve:

.. code-block:: python

    from libauc.losses import AUCMLoss
    from libauc.optimizers import PESG
    
    criterion = AUCMLoss()
    optimizer = PESG(model.parameters(), lr=0.1, a=1.0)

**When to use**: When AUC is the primary evaluation metric

Handling Class Imbalance
-------------------------

Consciousness detection datasets often have imbalanced classes (e.g., 70% unconscious, 30% conscious).

Strategies to Handle Imbalance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Weighted Loss Function**

.. code-block:: python

    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    
    # PyTorch
    weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.BCELoss(weight=weights[y_train])
    
    # Scikit-learn
    model = RandomForestClassifier(class_weight='balanced')

2. **Oversampling Minority Class**

.. code-block:: python

    from imblearn.over_sampling import SMOTE, RandomOverSampler
    
    # Synthetic Minority Oversampling Technique
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Or simple random oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

3. **Undersampling Majority Class**

.. code-block:: python

    from imblearn.under_sampling import RandomUnderSampler
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

4. **Threshold Adjustment**

.. code-block:: python

    # Instead of default 0.5 threshold
    optimal_threshold = find_optimal_threshold(y_val, y_pred_proba)
    predictions = (y_pred_proba >= optimal_threshold).astype(int)
    
    def find_optimal_threshold(y_true, y_probs):
        """Find threshold that maximizes F1 score."""
        from sklearn.metrics import f1_score
        
        thresholds = np.linspace(0.1, 0.9, 100)
        f1_scores = [f1_score(y_true, y_probs >= t) for t in thresholds]
        optimal_idx = np.argmax(f1_scores)
        
        return thresholds[optimal_idx]

5. **Ensemble with Different Sampling**

.. code-block:: python

    from sklearn.ensemble import BalancedRandomForestClassifier
    
    # Random Forest with built-in class balancing
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        sampling_strategy='all',  # Balance each tree
        random_state=42
    )

Regularization Techniques
--------------------------

L1 and L2 Regularization
^^^^^^^^^^^^^^^^^^^^^^^^

**L2 (Ridge)**: Penalizes large weights

.. math::

    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_{i} w_i^2

.. code-block:: python

    # Scikit-learn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)  # alpha = lambda
    
    # PyTorch
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # weight_decay implements L2 regularization

**L1 (Lasso)**: Induces sparsity (feature selection)

.. math::

    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_{i} |w_i|

.. code-block:: python

    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.1)

**Elastic Net**: Combines L1 and L2

.. code-block:: python

    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1, 50% L2

Dropout
^^^^^^^

Randomly zero out neurons during training:

.. code-block:: python

    # PyTorch
    self.dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x

**Guidelines:**
- Use 0.5 for fully connected layers
- Use 0.2-0.3 for convolutional layers
- Higher dropout for smaller datasets

Batch Normalization
^^^^^^^^^^^^^^^^^^^

Normalizes layer inputs to stabilize training:

.. code-block:: python

    self.bn1 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x

**Benefits:**
- Faster training
- Higher learning rates possible
- Acts as regularization
- Reduces internal covariate shift

Data Augmentation
^^^^^^^^^^^^^^^^^

Artificially increase dataset size:

.. code-block:: python

    import numpy as np
    
    def augment_fmri_connectivity(matrix, num_augmentations=5):
        """
        Generate augmented connectivity matrices.
        
        Techniques:
        1. Add Gaussian noise
        2. Random edge dropout
        3. Node permutation within networks
        """
        augmented = [matrix]
        
        for _ in range(num_augmentations):
            aug = matrix.copy()
            
            # Gaussian noise
            noise = np.random.normal(0, 0.02, aug.shape)
            aug += noise
            
            # Random edge dropout (sparsification)
            dropout_mask = np.random.rand(*aug.shape) > 0.05
            aug *= dropout_mask
            
            # Maintain symmetry
            aug = (aug + aug.T) / 2
            
            augmented.append(aug)
        
        return augmented

Early Stopping
^^^^^^^^^^^^^^

Stop training when validation performance plateaus:

.. code-block:: python

    class EarlyStopping:
        """Early stopping to prevent overfitting."""
        
        def __init__(self, patience=10, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.best_model = None
        
        def __call__(self, val_loss, model):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_model = model.state_dict().copy()
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_model = model.state_dict().copy()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            
            return self.early_stop

Optimization Strategies
-----------------------

Optimizer Selection
^^^^^^^^^^^^^^^^^^^

.. list-table:: Optimizer Comparison
   :header-rows: 1
   :widths: 20 30 30 20
   
   * - Optimizer
     - Best For
     - Learning Rate
     - Memory
   * - SGD
     - Large datasets, proven baseline
     - 0.01-0.1
     - Low
   * - SGD + Momentum
     - Overcoming local minima
     - 0.01-0.1
     - Low
   * - Adam
     - General purpose, adaptive
     - 0.001-0.01
     - Medium
   * - AdamW
     - Adam + better weight decay
     - 0.001-0.01
     - Medium
   * - RMSProp
     - Recurrent networks
     - 0.001-0.01
     - Medium
   * - AdaGrad
   - Sparse features
     - 0.01
     - High

**Recommendation**: Start with Adam, switch to AdamW if overfitting occurs.

.. code-block:: python

    # PyTorch implementations
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^

Adjust learning rate during training:

**1. Step Decay**

.. code-block:: python

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=30,  # Decay every 30 epochs
        gamma=0.1      # Multiply by 0.1
    )

**2. Exponential Decay**

.. code-block:: python

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

**3. Reduce on Plateau** (Recommended)

.. code-block:: python

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Minimize loss
        factor=0.5,         # Multiply by 0.5
        patience=5,         # After 5 epochs without improvement
        verbose=True
    )
    
    # In training loop
    scheduler.step(val_loss)

**4. Cosine Annealing**

.. code-block:: python

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # Number of epochs
        eta_min=1e-6
    )

**5. Cyclic Learning Rate**

.. code-block:: python

    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.01,
        step_size_up=10,
        mode='triangular'
    )

Gradient Clipping
^^^^^^^^^^^^^^^^^

Prevent exploding gradients:

.. code-block:: python

    # PyTorch
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # In training loop
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

Warm-Up
^^^^^^^

Gradually increase learning rate at start:

.. code-block:: python

    def warm_up_scheduler(optimizer, warmup_epochs, initial_lr, target_lr):
        """Linear warm-up schedule."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (target_lr - initial_lr) * epoch / warmup_epochs + initial_lr
            return target_lr
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

Complete Training Loop Example
-------------------------------

.. code-block:: python

    def train_consciousness_detector(
        model, train_loader, val_loader, 
        epochs=100, lr=0.001, patience=15
    ):
        """
        Complete training loop with all best practices.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss function with class weights
        pos_weight = compute_pos_weight(train_loader.dataset.targets)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float()
                
                # Forward pass
                logits = model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            val_probs, val_targets = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device).float()
                    
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
                    
                    val_probs.extend(probs.cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_targets, val_probs)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if early_stopping(avg_val_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(early_stopping.best_model)
                break
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        return model, history

def compute_pos_weight(targets):
    """Calculate positive class weight for imbalanced data."""
    neg_count = (targets == 0).sum()
    pos_count = (targets == 1).sum()
    return torch.tensor([neg_count / pos_count])

Summary
=======

Model Selection Guidelines
---------------------------

.. code-block:: text

    Quick Reference: Choose Your Model
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Dataset Size             Recommended Models                 │
    ├─────────────────────────────────────────────────────────────┤
    │ < 100 samples            Logistic Regression, SVM           │
    │ 100-300 samples          Random Forest, SVM (RBF)           │
    │ 300-500 samples          Random Forest, CNN (small)         │
    │ 500-1000 samples         CNN (3 layers), GNN                │
    │ > 1000 samples           Deep CNN/GNN, Ensembles            │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Priority                 Recommended Models                 │
    ├─────────────────────────────────────────────────────────────┤
    │ Interpretability         Logistic Regression, Random Forest │
    │ Speed                    Logistic Regression                │
    │ Robustness               Random Forest, SVM                 │
    │ Accuracy                 Ensemble (CNN+GNN+RF)              │
    │ Novelty                  GNN, Transformer                   │
    └─────────────────────────────────────────────────────────────┘

Key Takeaways
-------------

1. **Start Simple**: Always begin with logistic regression or random forest baseline
2. **Validate Carefully**: Use cross-validation for small datasets, hold-out set for large
3. **Regularize Heavily**: fMRI datasets are small and high-dimensional
4. **Ensemble When Possible**: Combining models almost always improves performance
5. **Monitor Overfitting**: Use validation curves and early stopping
6. **Domain Knowledge**: Incorporate neuroscience priors (brain networks, ROI groupings)

Next Steps
----------

- See :doc:`preprocessing` for data preparation best practices
- See :doc:`evaluation` for comprehensive model evaluation metrics
- See :doc:`visualization` for interpreting model decisions
- See :doc:`clinical_deployment` for production considerations

References
----------

.. [1] Güçlü, U., & van Gerven, M. A. (2015). Deep neural networks reveal a gradient in the complexity of neural representations across the ventral stream. *Journal of Neuroscience*, 35(27), 10005-10014.

.. [2] Khosla, M., Jamison, K., Ngo, G. H., Kuceyeski, A., & Sabuncu, M. R. (2019). Machine learning in resting-state fMRI analysis. *Magnetic Resonance Imaging*, 64, 101-121.

.. [3] Ktena, S. I., Parisot, S., Ferrante, E., Rajchl, M., Lee, M., Glocker, B., & Rueckert, D. (2018). Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*, 169, 431-442.

.. [4] Demertzi, A., Tagliazucchi, E., Dehaene, S., Deco, G., Barttfeld, P., Raimondo, F., ... & Sitt, J. D. (2019). Human consciousness is supported by dynamic complex patterns of brain signal coordination. *Science Advances*, 5(2), eaat7603.

.. [5] Engemann, D. A., Raimondo, F., King, J. R., Rohaut, B., Louppe, G., Faugeras, F., ... & Sitt, J. D. (2018). Robust EEG-based cross-site and cross-protocol classification of states of consciousness. *Brain*, 141(11), 3179-3192.

