Feature Extraction from fMRI Data
===================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------

Why Feature Extraction Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature extraction is the critical bridge between raw fMRI signals and machine learning models that detect consciousness. Think of fMRI data as a high-resolution video of brain activity—but instead of pixels showing colors, each "voxel" (3D pixel) shows blood oxygen levels that indicate neural activity. With tens of thousands of voxels recorded every few seconds, we need to extract meaningful patterns that capture the brain's functional organization.

For consciousness detection, we're particularly interested in features that represent:

1. **Communication patterns** between brain regions (connectivity)
2. **Network organization** of the brain (graph theory metrics)
3. **Dynamic changes** in brain states over time (temporal features)

Raw fMRI data is too high-dimensional and noisy for direct classification. A single scan might have 50,000+ voxels × 200+ timepoints = 10 million raw values. Feature extraction transforms this into a manageable set of interpretable measurements (typically hundreds to thousands) that capture the essence of brain function.

Types of Features Used in This Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our feature extraction pipeline generates three complementary feature families:

**1. Connectivity Features**
   Measure how brain regions communicate and coordinate their activity. These capture the functional integration of the brain—a key marker of consciousness.

**2. Graph Theory Features**
   Treat the brain as a network of nodes (regions) connected by edges (functional connections), then compute metrics that describe the network's topology and efficiency.

**3. Temporal Features**
   Capture how brain activity and connectivity patterns evolve over time, revealing dynamic state transitions characteristic of conscious processing.

Each feature type provides unique information about brain organization, and combining them gives us a comprehensive "fingerprint" of consciousness.

Connectivity Features
---------------------

The brain doesn't work through isolated regions—it functions as an integrated system where regions continuously communicate. Connectivity features measure these functional relationships.

Pearson Correlation Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What They Represent**

Pearson correlation quantifies how synchronous two brain regions' activities are. If regions A and B consistently activate together (correlation ≈ +1) or show opposite patterns (correlation ≈ -1), they're functionally connected. Think of it like measuring how two musicians in an orchestra synchronize their playing.

**How They're Computed**

For each pair of brain regions (or ROIs—Regions of Interest):

1. Extract the BOLD signal timeseries for both regions
2. Compute the Pearson correlation coefficient:

.. math::

   r_{ij} = \frac{\sum_{t=1}^{T}(x_{i}(t) - \bar{x}_i)(x_{j}(t) - \bar{x}_j)}{\sqrt{\sum_{t=1}^{T}(x_{i}(t) - \bar{x}_i)^2} \sqrt{\sum_{t=1}^{T}(x_{j}(t) - \bar{x}_j)^2}}

where :math:`x_i(t)` is the signal in region *i* at time *t*, and :math:`\bar{x}_i` is the mean signal across time.

For *N* brain regions, this produces an *N* × *N* symmetric connectivity matrix where each element represents the correlation between two regions.

**Interpretation**

- **Values near +1**: Strong positive correlation (regions activate together)
- **Values near 0**: No linear relationship (regions operate independently)
- **Values near -1**: Strong negative correlation (anti-correlated activity)

**Why It Matters for Consciousness**

Conscious states show characteristic patterns of long-range correlations between frontal and parietal regions. Unconscious states (anesthesia, vegetative state) show breakdown of these correlations, with more isolated, local processing.

Partial Correlation
~~~~~~~~~~~~~~~~~~~

**The Problem with Pearson Correlation**

Pearson correlation can be misleading. Imagine three regions: A→B→C, where A influences B, and B influences C. Pearson correlation might show A and C are correlated, but this is an *indirect* connection through B, not a direct relationship.

**What Partial Correlation Does**

Partial correlation removes indirect connections by controlling for all other regions. It estimates the *direct* functional connection between two regions while "partialing out" the influence of all others.

.. math::

   r_{ij \cdot \text{rest}} = \text{correlation}(x_i, x_j \mid x_k \text{ for all } k \neq i,j)

**Computation Method**

The most efficient approach uses the precision matrix (inverse covariance matrix):

.. math::

   P = \Sigma^{-1}

where :math:`\Sigma` is the covariance matrix. The partial correlation between regions *i* and *j* is:

.. math::

   r_{ij \cdot \text{rest}} = -\frac{P_{ij}}{\sqrt{P_{ii} P_{jj}}}

**Why It Matters**

Partial correlation reveals the brain's true "wiring diagram" by showing only direct functional connections. This is crucial because consciousness involves specific direct pathways, not just general correlations. It helps distinguish between:

- **Conscious processing**: Rich direct connections in frontoparietal networks
- **Unconscious processing**: Simpler, more local direct connectivity

Covariance Matrices
~~~~~~~~~~~~~~~~~~~

**What They Represent**

While correlation standardizes signals (ranging from -1 to +1), covariance preserves the scale of signal fluctuations:

.. math::

   \text{Cov}(x_i, x_j) = \frac{1}{T-1}\sum_{t=1}^{T}(x_i(t) - \bar{x}_i)(x_j(t) - \bar{x}_j)

Covariance captures both the direction of the relationship *and* the magnitude of co-variation. Two regions might correlate strongly but with small amplitude fluctuations (low covariance) or moderate correlation with large amplitude fluctuations (high covariance).

**When to Use Each Type**

.. list-table:: Connectivity Feature Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Feature Type
     - Best For
     - Limitations
   * - **Pearson Correlation**
     - Quick functional connectivity assessment; standard in fMRI research
     - Includes indirect connections; sensitive to preprocessing
   * - **Partial Correlation**
     - Identifying direct connections; network structure analysis
     - Requires more data; can be unstable with high dimensionality
   * - **Covariance**
     - When signal amplitude matters; multivariate analyses
     - Scale-dependent; harder to interpret across subjects

**Practical Advice**

For consciousness detection, use **all three** as complementary features:

- Correlation matrices capture overall functional integration
- Partial correlation reveals direct network architecture
- Covariance adds information about signal strength

Graph Theory Features
---------------------

Brain as a Network
~~~~~~~~~~~~~~~~~~

Graph theory provides a powerful framework for understanding brain organization. We represent the brain as a **graph**:

- **Nodes**: Brain regions (e.g., 90 regions from an anatomical atlas)
- **Edges**: Functional connections between regions (from connectivity matrices)
- **Weights**: Connection strength (e.g., correlation values)

This abstraction lets us apply network science concepts to quantify brain organization. Think of social networks (Facebook), transportation networks (airline routes), or the internet—the same mathematical tools that analyze these can analyze brain networks.

**Why This Matters**

Conscious brains show specific network properties: they're both *segregated* (specialized regions forming modules) and *integrated* (efficient long-range communication). Graph metrics quantify these properties.

Global Metrics
~~~~~~~~~~~~~~

These metrics characterize the entire brain network's organization.

**Modularity**

Modularity measures how well the network divides into communities or modules—groups of nodes that are densely connected internally but sparsely connected externally.

.. math::

   Q = \frac{1}{2m}\sum_{ij}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)

where:
- :math:`A_{ij}` is the connection weight between nodes *i* and *j*
- :math:`k_i` is the degree (sum of connections) of node *i*
- *m* is the total number of edges
- :math:`\delta(c_i, c_j) = 1` if nodes *i* and *j* are in the same community, 0 otherwise

**Intuition**: High modularity means the brain is organized into specialized subsystems (visual, motor, language modules). Low modularity means a more homogeneous network.

**For Consciousness**: Conscious states maintain modular organization—specialized systems remain distinct while still communicating. Loss of consciousness often reduces modularity as the network becomes more random.

**Global Efficiency**

Global efficiency measures how efficiently information can spread across the entire network—the average inverse shortest path length between all node pairs.

.. math::

   E_{\text{global}} = \frac{1}{N(N-1)}\sum_{i \neq j}\frac{1}{d_{ij}}

where :math:`d_{ij}` is the shortest path length (minimum number of steps) between nodes *i* and *j*.

**Intuition**: Think of air travel—high global efficiency means you can get anywhere with few connections. In the brain, high efficiency means information can reach any region quickly.

**For Consciousness**: Conscious states require high global efficiency for widespread information integration. Anesthesia and deep sleep reduce global efficiency, fragmenting the network into isolated islands.

**Characteristic Path Length**

The average shortest path length across all node pairs:

.. math::

   L = \frac{1}{N(N-1)}\sum_{i \neq j}d_{ij}

**Intuition**: This is the flip side of global efficiency—how many "hops" on average to get from one region to another. Lower values indicate tighter integration.

**For Consciousness**: Conscious brains maintain short path lengths between distant regions. Unconscious states show longer paths, requiring more intermediary steps for communication.

Nodal Metrics
~~~~~~~~~~~~~

These metrics characterize individual nodes' roles in the network.

**Degree Centrality**

The simplest centrality measure—how many connections a node has.

.. math::

   k_i = \sum_{j}A_{ij}

For weighted networks (strength):

.. math::

   s_i = \sum_{j}w_{ij}

where :math:`w_{ij}` is the connection weight.

**Intuition**: High-degree nodes are "hubs"—highly connected regions that facilitate communication, like major airports in a flight network.

**For Consciousness**: Specific hubs (posterior cingulate cortex, precuneus, frontal regions) are critical for consciousness. Their disruption disproportionately affects conscious processing.

**Betweenness Centrality**

Measures how often a node lies on the shortest path between other nodes:

.. math::

   b_i = \sum_{j \neq k}\frac{\sigma_{jk}(i)}{\sigma_{jk}}

where :math:`\sigma_{jk}` is the total number of shortest paths from *j* to *k*, and :math:`\sigma_{jk}(i)` is the number that pass through *i*.

**Intuition**: High-betweenness nodes are "bridges" or "connectors" that link different communities. Removing them fragments the network.

**For Consciousness**: Connector hubs in frontal and parietal regions show high betweenness in conscious states. These regions mediate communication between specialized modules.

**Clustering Coefficient**

Measures how much a node's neighbors are also connected to each other (forming triangles):

.. math::

   C_i = \frac{2t_i}{k_i(k_i-1)}

where :math:`t_i` is the number of triangles around node *i*.

**Intuition**: High clustering means a node's neighbors form a tight-knit group. Think of your friend group—high clustering means your friends are also friends with each other.

**For Consciousness**: High clustering supports local information processing and specialization. Conscious states balance high clustering (local processing) with long-range connections (integration).

Network-Level Metrics
~~~~~~~~~~~~~~~~~~~~~~

These metrics characterize overall network topology.

**Small-Worldness**

A small-world network combines high clustering (like regular lattices) with short path lengths (like random networks)—the "six degrees of separation" property.

.. math::

   \sigma = \frac{C/C_{\text{random}}}{L/L_{\text{random}}}

A network is small-world if :math:`\sigma > 1` (typically :math:`\sigma \gg 1`).

**Intuition**: Most people have local friend groups (high clustering) but can reach anyone globally through few connections (short paths). Brain networks show this property—specialized regions (high clustering) can rapidly communicate (short paths).

**For Consciousness**: Healthy conscious brains are small-world networks, optimizing both specialized processing and global integration. This topology supports both the segregation and integration required for consciousness.

**Rich Club Organization**

Rich club organization occurs when high-degree hubs are densely interconnected, forming a "club" of highly connected nodes.

.. math::

   \phi(k) = \frac{2E_{>k}}{N_{>k}(N_{>k}-1)}

where :math:`E_{>k}` is the number of connections among nodes with degree > *k*, and :math:`N_{>k}` is the number of such nodes.

**Intuition**: Like wealthy individuals often knowing each other and forming exclusive networks, brain hubs preferentially connect to other hubs, creating a high-capacity "backbone" for information flow.

**For Consciousness**: The rich club backbone (connecting hubs in prefrontal, parietal, and cingulate cortices) is essential for conscious integration. Damage to these connections has disproportionate effects on consciousness.

Temporal Features
-----------------

Static connectivity measures (computed over entire scans) miss a crucial aspect: the brain is dynamic. Neural activity and connectivity patterns constantly evolve as the brain processes information and transitions between states.

Dynamic Connectivity
~~~~~~~~~~~~~~~~~~~~

**The Sliding Window Approach**

Instead of computing one connectivity matrix for an entire scan, we compute multiple matrices using a sliding window:

1. Choose a window length (e.g., 30-60 seconds, or 20-40 TRs)
2. Compute connectivity within that window
3. Slide the window forward (e.g., by 1-2 TRs)
4. Repeat for the entire timeseries

This produces a sequence of connectivity matrices: :math:`C(t_1), C(t_2), ..., C(t_n)`.

**What This Captures**

Dynamic connectivity reveals:

- How strongly regions connect at different moments
- Which connections are stable vs. variable
- Temporal patterns and periodicities in connectivity

**Metrics from Dynamic Connectivity**

From the sequence of connectivity matrices, we extract:

- **Mean connectivity**: Average over time (similar to static connectivity)
- **Variance**: How much each connection fluctuates
- **Temporal correlation**: How one connection's strength relates to another over time

.. math::

   \text{Variance}(C_{ij}) = \frac{1}{T}\sum_{t=1}^{T}(C_{ij}(t) - \bar{C}_{ij})^2

Capturing Brain State Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**State Transitions**

The brain doesn't maintain constant connectivity—it transitions between discrete "states" with characteristic connectivity patterns. Think of mental states (focused attention, mind-wandering, sensory processing) as having distinct connectivity signatures.

**Approach**:

1. Compute dynamic connectivity matrices
2. Apply clustering (e.g., k-means) to identify recurring connectivity patterns
3. Characterize state properties:
   - Number of states
   - State occurrence frequency
   - State transition probabilities
   - Dwell time in each state

**For Consciousness**: Conscious states show:
- More distinct states (greater repertoire)
- More frequent state transitions (flexibility)
- Specific states involving frontoparietal integration
- Orderly transition patterns

Unconscious states show:
- Fewer distinct states (reduced repertoire)
- Longer dwell times (less flexibility)
- Loss of integrated states
- More random transitions

**Brain Network Dynamics**

Beyond identifying states, we can quantify dynamics directly:

**Flexibility**: How often a node's community assignment changes

.. math::

   F_i = 1 - \frac{1}{T-1}\sum_{t=1}^{T-1}\delta(c_i(t), c_i(t+1))

where :math:`c_i(t)` is node *i*'s community at time *t*.

**Integration**: How well a node integrates information across communities over time

**Recruitment**: How consistently nodes within a module connect over time

Frequency Domain Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

BOLD signals contain oscillations at different frequencies. Frequency analysis decomposes signals into these components.

**Power Spectral Density**

Quantifies signal power at each frequency:

.. math::

   P(f) = |F\{x(t)\}|^2

where :math:`F\{\cdot\}` is the Fourier transform.

**Frequency Bands in fMRI**

Due to slow hemodynamic response, fMRI captures low frequencies:

- **Slow-5** (0.01-0.027 Hz): Long-range integration
- **Slow-4** (0.027-0.073 Hz): Default mode network activity
- **Slow-3** (0.073-0.198 Hz): Sensorimotor networks

**Amplitude of Low-Frequency Fluctuations (ALFF)**

.. math::

   \text{ALFF} = \sum_{f=0.01}^{0.08}P(f)

Conscious states show characteristic patterns in low-frequency power, particularly in the default mode network.

**Coherence**

Frequency-domain connectivity—how synchronized two regions' oscillations are at specific frequencies:

.. math::

   \text{Coh}_{ij}(f) = \frac{|P_{ij}(f)|^2}{P_{ii}(f)P_{jj}(f)}

Different consciousness states may differ in coherence at specific frequencies rather than broadband connectivity.

Feature Selection and Dimensionality
-------------------------------------

Dealing with High-Dimensional Connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Curse of Dimensionality**

A connectivity matrix for 90 brain regions contains 90 × 90 = 8,100 values. With symmetry, that's still 4,005 unique connections. Add graph metrics, temporal features, and multiple connectivity types—you quickly have 10,000+ features for datasets with perhaps only 100-500 subjects.

**Problems This Creates**:

1. **Overfitting**: Models learn noise instead of true patterns
2. **Computational cost**: Training becomes slow and memory-intensive
3. **Interpretability**: Hard to understand which features matter
4. **Multicollinearity**: Many features are highly correlated

**The ML Practitioner's Dilemma**

You need enough features to capture consciousness signatures, but too many features cause models to fail. The solution: strategic feature selection and dimensionality reduction.

Feature Selection Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Filter Methods**

Select features based on statistical properties before training.

1. **Variance Threshold**: Remove low-variance features (they don't vary across samples, so they can't discriminate)

   .. code-block:: python
   
      from sklearn.feature_selection import VarianceThreshold
      selector = VarianceThreshold(threshold=0.01)
      X_filtered = selector.fit_transform(X)

2. **Correlation with Target**: Select features most correlated with consciousness labels

   .. math::
   
      r = \text{corr}(X_i, y)

   Keep top-k features with highest |r|.

3. **Statistical Tests**: Use t-tests or ANOVA to identify features that differ significantly between consciousness states

**Wrapper Methods**

Use the ML model itself to select features.

1. **Recursive Feature Elimination (RFE)**: Iteratively remove least important features

   .. code-block:: python
   
      from sklearn.feature_selection import RFE
      from sklearn.svm import SVC
      
      estimator = SVC(kernel="linear")
      selector = RFE(estimator, n_features_to_select=100)
      X_selected = selector.fit_transform(X, y)

2. **Sequential Selection**: Forward or backward stepwise addition/removal of features

**Embedded Methods**

Feature selection happens during model training.

1. **L1 Regularization (Lasso)**: Penalizes model complexity, driving some feature weights to zero

   .. math::
   
      \text{Loss} = \text{MSE} + \lambda\sum_{i}|w_i|

2. **Tree-Based Feature Importance**: Random Forests and Gradient Boosting compute feature importance scores

   .. code-block:: python
   
      from sklearn.ensemble import RandomForestClassifier
      
      clf = RandomForestClassifier(n_estimators=100)
      clf.fit(X, y)
      importances = clf.feature_importances_

**Recommended Approach for fMRI**

Use a combination:

1. **Pre-filter**: Remove zero-variance and highly correlated features (r > 0.95)
2. **Statistical filter**: Select features with significant group differences (p < 0.05, corrected)
3. **Model-based**: Use Random Forest feature importance or L1 regularization for final selection
4. **Validate**: Ensure selected features generalize to held-out test data

PCA and Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principal Component Analysis (PCA)**

PCA rotates feature space to find orthogonal directions of maximum variance.

.. math::

   X_{\text{PCA}} = XW

where W contains eigenvectors of the covariance matrix :math:`X^TX`.

**How It Helps**:

- Reduces thousands of correlated features to hundreds of uncorrelated components
- First few components capture most variance
- Removes redundancy and noise

**For Connectivity Matrices**:

Connectivity matrices are symmetric with redundant information. PCA can reduce 4,005 connections to ~50-100 principal components capturing 80-90% of variance.

.. code-block:: python

   from sklearn.decomposition import PCA
   
   pca = PCA(n_components=0.9)  # Keep 90% of variance
   X_reduced = pca.fit_transform(X)
   print(f"Reduced to {pca.n_components_} components")

**Limitations**:

- Loss of interpretability (components are linear combinations)
- Assumes linear relationships
- Sensitive to scaling

**Alternative: Sparse PCA**

Encourages sparse component loadings (fewer non-zero weights), improving interpretability:

.. math::

   \min_{W, H} ||X - WH||^2 + \lambda||W||_1

**Independent Component Analysis (ICA)**

ICA finds statistically independent components rather than orthogonal ones. Useful when features represent mixtures of independent sources.

**Autoencoders**

Neural network-based dimensionality reduction that can capture nonlinear relationships:

- **Encoder**: Maps high-dimensional features to low-dimensional latent space
- **Decoder**: Reconstructs original features
- **Latent space**: Provides compressed representation

.. code-block:: python

   from tensorflow.keras import layers, Model
   
   # Encoder
   input_layer = layers.Input(shape=(4005,))
   encoded = layers.Dense(512, activation='relu')(input_layer)
   encoded = layers.Dense(128, activation='relu')(encoded)
   latent = layers.Dense(50, activation='relu')(encoded)
   
   # Decoder
   decoded = layers.Dense(128, activation='relu')(latent)
   decoded = layers.Dense(512, activation='relu')(decoded)
   output_layer = layers.Dense(4005, activation='linear')(decoded)
   
   autoencoder = Model(input_layer, output_layer)
   autoencoder.compile(optimizer='adam', loss='mse')

**Graph-Based Dimensionality Reduction**

Methods like **t-SNE** and **UMAP** preserve local structure—useful for visualization and discovering clusters in feature space.

Practical Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Start Simple, Add Complexity**

1. **Baseline**: Start with mean connectivity matrices → PCA → simple classifier
2. **Add graph metrics**: Include global efficiency, modularity, rich club
3. **Add dynamics**: Include temporal variance of connectivity
4. **Feature selection**: Use Random Forest importance to identify key features
5. **Refinement**: Try more sophisticated dimensionality reduction if needed

**Cross-Validation is Critical**

Always perform feature selection *within* cross-validation folds:

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import Pipeline
   
   pipeline = Pipeline([
       ('variance_filter', VarianceThreshold()),
       ('scaler', StandardScaler()),
       ('feature_selection', SelectKBest(k=100)),
       ('pca', PCA(n_components=0.9)),
       ('classifier', SVC())
   ])
   
   scores = cross_val_score(pipeline, X, y, cv=5)

This prevents data leakage—selecting features based on the entire dataset biases results.

**Feature Engineering > Feature Selection**

Creating meaningful derived features often beats simply selecting from raw features:

- **Ratios**: E.g., within-module / between-module connectivity
- **Differences**: E.g., connectivity in state A vs. state B
- **Summary statistics**: Mean, variance, skewness of connectivity distributions
- **Interaction terms**: Products of important features

**Domain Knowledge Matters**

Don't rely purely on automated selection. Neuroscience literature identifies specific networks (default mode, frontoparietal, salience) critical for consciousness. Prioritize features from these networks.

Summary
-------

Feature extraction transforms raw fMRI timeseries into interpretable measurements that machine learning models can use to detect consciousness:

1. **Connectivity features** quantify functional communication between brain regions
2. **Graph theory features** characterize network organization and topology
3. **Temporal features** capture dynamic changes and state transitions
4. **Dimensionality reduction** makes high-dimensional features tractable

The key is combining complementary feature types while managing dimensionality through principled selection and reduction. This creates a rich, interpretable feature space that captures the multifaceted nature of conscious brain activity.

With these features, machine learning models can learn to distinguish conscious from unconscious states, potentially revolutionizing clinical assessment of patients with disorders of consciousness.
