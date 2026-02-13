========
Glossary
========

A comprehensive guide to neuroscience and neuroimaging terms used throughout
this documentation. Definitions are based on peer-reviewed neuroscience
literature and encyclopedic sources.

.. contents:: Table of Contents
   :depth: 2
   :local:


Neuroscience Fundamentals
=========================

Core Concepts
-------------

Behavioral Response
^^^^^^^^^^^^^^^^^^^
**Definition**: An observable action or reaction to a stimulus, such as moving a hand, speaking, or squeezing fingers when commanded.

**Context in this project**: Traditional consciousness assessment relies on behavioral responses.

Consciousness
^^^^^^^^^^^^^
**Definition**: The state of being aware of and able to think about oneself, one's surroundings, and internal mental states. Consciousness involves both arousal (wakefulness) and awareness (content of experience).

**Context in this project**: We detect consciousness by analyzing brain network patterns that distinguish conscious awareness from unconscious states during anesthesia.

**References**:

* Laureys, S. (2005). The neural correlate of (un)awareness: lessons from the vegetative state. *Trends in Cognitive Sciences*, 9(12), 556-559.
* Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200-227.

Covert Consciousness
^^^^^^^^^^^^^^^^^^^^^
**Definition**: A dissociation between behavioral responsiveness and neural markers of awareness, where patients show brain activity consistent with conscious processing (e.g., following mental imagery commands) but exhibit no observable behavioral responses.

**Also known as**: Covert awareness, hidden consciousness, cognitive motor dissociation.

**Context in this project**: The primary phenomenon we aim to detect - patients who appear unconscious but show neural signatures of consciousness.

**References**:

* Huang, Z., et al. (2018). Brain imaging reveals covert consciousness during behavioral unresponsiveness induced by propofol. *Scientific Reports*, 8, 13195.
* Schiff, N. D. (2015). Cognitive motor dissociation following severe brain injuries. *JAMA Neurology*, 72(12), 1413-1415.

Neural
^^^^^^
**Definition**: Relating to neurons (nerve cells) or the nervous system. Neural activity refers to electrical and chemical signaling between neurons.

**Context in this project**: We analyze neural signatures (brain activity patterns) that indicate conscious awareness.


Neuroimaging
============

Data Acquisition
----------------

BOLD Signal (Blood-Oxygen-Level-Dependent)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: The MRI signal change that reflects changes in blood oxygenation. When neurons are active, they consume oxygen, triggering increased blood flow to that region. This changes the magnetic properties of blood, detectable by MRI scanners.

**Key principle**: Active brain regions show increased blood oxygenation → stronger BOLD signal.

**Important note**: BOLD signal is an indirect measure of neural activity with hemodynamic delay.

**References**:

* Ogawa, S., & Lee, T. M. (1990). Magnetic resonance imaging of blood vessels at high fields: in vivo and in vitro measurements and image simulation. *Magnetic Resonance in Medicine*, 16(1), 9-18.
* Buxton, R. B., et al. (2004). Modeling the hemodynamic response to brain activation. *NeuroImage*, 23(Suppl 1), S220-S233.

fMRI (Functional Magnetic Resonance Imaging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A neuroimaging technique that measures brain activity by detecting changes in blood flow. Active brain regions require more oxygen, leading to localized changes in blood oxygenation that fMRI can detect.

**Technical details**:

* Non-invasive (no radiation)
* Measures BOLD signal as a proxy for neural activity

**Context in this project**: The primary data source for consciousness detection.

**References**:

* Ogawa, S., et al. (1990). Brain magnetic resonance imaging with contrast dependent on blood oxygenation. *Proceedings of the National Academy of Sciences*, 87(24), 9868-9872.
* Logothetis, N. K. (2008). What we can do and what we cannot do with fMRI. *Nature*, 453(7197), 869-878.

Resting-State fMRI
^^^^^^^^^^^^^^^^^^^
**Definition**: fMRI data collected while participants rest quietly without performing any explicit task. Spontaneous fluctuations in brain activity reveal intrinsic functional networks.

**Context in this project**: Used to assess baseline brain organization during different sedation levels.

**References**:

* Biswal, B., et al. (1995). Functional connectivity in the motor cortex of resting human brain using echo-planar MRI. *Magnetic Resonance in Medicine*, 34(4), 537-541.
* Fox, M. D., & Raichle, M. E. (2007). Spontaneous fluctuations in brain activity observed with functional magnetic resonance imaging. *Nature Reviews Neuroscience*, 8(9), 700-711.

Task-Based fMRI
^^^^^^^^^^^^^^^
**Definition**: fMRI collected while participants perform specific cognitive tasks. Brain regions involved in the task show increased BOLD signal compared to baseline.

**Context in this project**: Mental imagery tasks (imagining playing tennis, spatial navigation) are used to probe consciousness during sedation.

**References**:

* Haxby, J. V., et al. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science*, 293(5539), 2425-2430.
* Owen, A. M., et al. (2006). Detecting awareness in the vegetative state. *Science*, 313(5792), 1402.

TR (Repetition Time)
^^^^^^^^^^^^^^^^^^^^^
**Definition**: The time between successive MRI measurements of the same slice or volume. In fMRI, it represents how frequently the entire brain is scanned.

**In this project**: TR = 3.0 seconds, meaning we capture one complete brain volume every 3 seconds.

Voxel
^^^^^
**Definition**: A volume element (3D pixel) representing the smallest distinguishable box-shaped part of a 3D image. In brain imaging, each voxel contains signal from neural populations.

Preprocessing & Pipelines
--------------------------

fMRIPrep
^^^^^^^^
**Definition**: An automated preprocessing pipeline for fMRI data. It takes raw scanner images and applies motion correction, spatial normalisation, confound estimation, and other standard steps to produce analysis-ready data.

**Context in this project**: The dataset authors used fMRIPrep to preprocess the raw fMRI scans before publishing them on OpenNeuro.

**References**:

* Esteban, O., et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods*, 16(1), 111-116.

Framewise Displacement (FD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A scalar measure of head movement between consecutive fMRI timepoints. It combines translational and rotational motion into a single number (in millimetres) that indicates how much the head moved.

**Context in this project**: Timepoints where FD ≥ 0.8 mm are excluded (censored) before computing connectivity, because excessive head motion corrupts the signal.

**References**:

* Power, J. D., et al. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142-2154.

Global Signal Removal
^^^^^^^^^^^^^^^^^^^^^
**Definition**: A preprocessing step that removes the strongest shared pattern of activity across all brain regions. This isolates region-specific connectivity from the overall background signal.

**Context in this project**: Applied before computing the ISD feature so that the efficiency and clustering values reflect specific inter-regional connections rather than global fluctuations.

Parcellation
^^^^^^^^^^^^
**Definition**: The process of dividing the brain into discrete regions (parcels) based on anatomical, functional, or connectivity criteria. Each parcel is treated as a single unit of analysis.

**Context in this project**: The 4S456Parcels atlas parcellates the brain into 456 regions (446 used).

Preprocessing
^^^^^^^^^^^^^
**Definition**: A series of computational steps applied to raw neuroimaging data before analysis to remove artifacts, normalize anatomy, and improve signal-to-noise ratio.

**Common steps**:

* **Motion correction**: Realigns brain volumes to correct for head movement
* **Slice timing correction**: Adjusts for different acquisition times of brain slices
* **Spatial normalization**: Warps individual brains to a standard template space
* **Smoothing**: Spatial filtering to reduce noise
* **Denoising**: Removes physiological noise (cardiac, respiratory), scanner drift

**References**:

* Lindquist, M. A. (2008). The statistical analysis of fMRI data. *Statistical Science*, 23(4), 439-464.
* Poldrack, R. A., et al. (2011). *Handbook of Functional MRI Data Analysis*. Cambridge University Press.

XCP-D
^^^^^
**Definition**: A postprocessing pipeline that takes fMRIPrep outputs and performs additional steps such as confound regression, bandpass filtering, and parcellation (extracting region-level timeseries from an atlas). It produces connectivity-ready derivatives.

**Context in this project**: The dataset's XCP-D output directory contains the timeseries and motion files that our code loads directly.

**References**:

* Ciric, R., et al. (2018). Mitigating head motion artifact in functional connectivity MRI. *Nature Protocols*, 13, 2801-2826.

Data Standards & Platforms
--------------------------

BIDS (Brain Imaging Data Structure)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A community standard for organising and describing neuroimaging data. It specifies folder structures, file naming conventions, and metadata formats so that datasets are interoperable across tools and labs.

**Context in this project**: The Michigan dataset follows the BIDS standard, which is why tools like fMRIPrep and XCP-D can process it automatically.

**References**:

* Gorgolewski, K. J., et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044.

OpenNeuro
^^^^^^^^^
**Definition**: A free, open-access platform for sharing neuroimaging datasets. Datasets are stored in BIDS format and assigned a persistent identifier (DOI).

**Context in this project**: The Michigan dataset is hosted on OpenNeuro with ID ds006623.

**Website**: https://openneuro.org/

ROI (Region of Interest)
^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A defined anatomical or functional brain region used as a unit of analysis. ROIs are typically derived from brain atlases that parcellate (divide) the brain into distinct regions based on anatomy, function, or connectivity patterns.

**In this project**: We use 446 ROIs from the Schaefer-Gordon atlas to represent the whole brain as discrete regions.

**References**:

* Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.
* Tzourio-Mazoyer, N., et al. (2002). Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. *NeuroImage*, 15(1), 273-289.


Brain Connectivity
==================

Connectivity Concepts
---------------------

Brain Network
^^^^^^^^^^^^^
**Definition**: A set of brain regions (nodes) and their connections (edges), conceptualized as a graph. Networks can be structural (anatomical connections) or functional (statistical associations).

**Major brain networks**:

* **Default Mode Network (DMN)**: Active during rest, self-referential thought
* **Frontoparietal Network**: Executive control, attention
* **Salience Network**: Detecting behaviorally relevant stimuli
* **Sensorimotor Network**: Motor control and sensory processing

**Context in this project**: Consciousness involves integration across multiple networks; anesthesia disrupts network connectivity.

**References**:

* Raichle, M. E. (2015). The brain's default mode network. *Annual Review of Neuroscience*, 38, 433-447.
* Yeo, B. T., et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.

Connectivity Matrix
^^^^^^^^^^^^^^^^^^^
**Definition**: An N × N matrix where N is the number of brain regions (ROIs), and each element (i,j) represents the functional connectivity strength between regions i and j.

**Properties**:

* Symmetric: connectivity from A to B equals B to A
* Diagonal is typically 1 (perfect correlation with itself) or removed
* Values typically range from -1 (anti-correlated) to +1 (perfectly correlated)

**In this project**: 446 × 446 matrices (99,235 unique connections) representing whole-brain connectivity.

Functional Connectivity
^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: Statistical dependence between the time series of activity in different brain regions. High functional connectivity indicates regions that consistently activate together over time.

**Measurement**: Typically computed as Pearson correlation between ROI time series.

**Important distinction**: Functional connectivity does not imply direct anatomical connections or causal relationships - it measures statistical association.

**Context in this project**: The primary feature for consciousness detection; conscious states show specific patterns of long-range connectivity.

**References**:

* Friston, K. J. (1994). Functional and effective connectivity in neuroimaging: A synthesis. *Human Brain Mapping*, 2(1-2), 56-78.
* Biswal, B., et al. (1995). Functional connectivity in the motor cortex of resting human brain using echo-planar MRI. *Magnetic Resonance in Medicine*, 34(4), 537-541.

ISD (Integration-Segregation Difference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A single number that captures the balance between integration and segregation in a brain network. It is computed as:

::

   ISD = efficiency - clustering

where *efficiency* measures how easily information can travel between any two brain regions, and *clustering* measures how much regions form tight local groups.

**Context in this project**: The central feature used for consciousness classification. Higher ISD is associated with consciousness; loss of consciousness reduces ISD.

Network Topology
^^^^^^^^^^^^^^^^
**Definition**: The pattern of connections in a network, independent of spatial layout. Describes organizational principles like clustering, modularity, and hub structure.

**Key concepts**:

* **Small-world**: High local clustering + short path lengths (efficient communication)
* **Hubs**: Highly connected nodes that integrate information across the network
* **Modules**: Densely connected subnetworks with sparse connections between modules

**References**:

* Sporns, O. (2013). Network attributes for segregation and integration in the human brain. *Current Opinion in Neurobiology*, 23(2), 162-171.
* Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186-198.

Graph Theory Metrics
--------------------

Clustering Coefficient
^^^^^^^^^^^^^^^^^^^^^^
**Definition**: Measures local connectivity - the extent to which a node's neighbors are also connected to each other. Represents local efficiency and segregation.

**Intuition**: In social networks, it's the probability that your friends are also friends with each other.

**Formula**: For node i with k neighbors, clustering C_i = (number of connections between neighbors) / (k(k-1)/2).

**In brain networks**: High clustering indicates functionally specialized modules.

**References**:

* Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
* Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059-1069.

Degree
^^^^^^
**Definition**: The number of connections (edges) a node has. Nodes with high degree are called **hubs** and play critical roles in network communication.

**Formula**: For node i, degree k_i = sum of all connections to other nodes.

**Interpretation**: High-degree nodes integrate information across the network.

**References**:

* van den Heuvel, M. P., & Sporns, O. (2013). Network hubs in the human brain. *Trends in Cognitive Sciences*, 17(12), 683-696.

Edge
^^^^
**Definition**: A connection between two nodes in a graph. In functional brain networks, edges represent functional connectivity (correlation strength) between regions.

**Edge weight**: The strength of connection, typically the correlation coefficient.

Global Efficiency
^^^^^^^^^^^^^^^^^
**Definition**: A graph metric that measures how efficiently information can travel across the entire network. It is calculated as the average of the inverse shortest path lengths between all pairs of nodes. Higher values mean the network is more integrated.

**Context in this project**: Used as part of the ISD computation. Combined across multiple thresholds (multilevel efficiency).

**References**:

* Latora, V., & Marchiori, M. (2001). Efficient behavior of small-world networks. *Physical Review Letters*, 87(19), 198701.

Hub
^^^
**Definition**: A node with disproportionately high connectivity or importance to network function. Hubs integrate information across the network and are critical for efficient communication.

**Types**:

* **Connector hubs**: Link different modules
* **Provincial hubs**: Highly connected within their module

**References**:

* van den Heuvel, M. P., & Sporns, O. (2013). Network hubs in the human brain. *Trends in Cognitive Sciences*, 17(12), 683-696.

Modularity
^^^^^^^^^^
**Definition**: The degree to which a network can be subdivided into distinct communities (modules) with dense internal connections but sparse connections between modules.

**In brain networks**: Modules often correspond to functional systems (visual, motor, cognitive control).

**References**:

* Newman, M. E. (2006). Modularity and community structure in networks. *Proceedings of the National Academy of Sciences*, 103(23), 8577-8582.
* Sporns, O., & Betzel, R. F. (2016). Modular brain networks. *Annual Review of Psychology*, 67, 613-640.

Node
^^^^
**Definition**: In brain network analysis, a node represents a brain region (ROI). The collection of all nodes forms the network.

**In this project**: 446 nodes corresponding to cortical and subcortical brain regions.

Path Length
^^^^^^^^^^^
**Definition**: The minimum number of edges to traverse from one node to another. Average path length measures network integration - how efficiently information can travel globally.

**Shortest path**: The most direct route between two nodes.

**In brain networks**: Short path length indicates efficient global communication.

**References**:

* Achard, S., & Bullmore, E. (2007). Efficiency and cost of economical brain functional networks. *PLoS Computational Biology*, 3(2), e17.


Clinical Context
================

Anesthesia & Sedation
---------------------

Anesthesia
^^^^^^^^^^
**Definition**: A medically induced state of reversible loss of consciousness, sensation, and reflexes to enable surgical procedures. Includes general anesthesia (complete unconsciousness) and sedation (reduced consciousness).

**Mechanisms**: Anesthetic drugs enhance inhibitory neurotransmission (GABA) and reduce excitatory transmission, disrupting communication between brain regions.

**Context in this project**: Propofol-induced anesthesia is used as a controlled model to study consciousness.

**References**:

* Brown, E. N., et al. (2010). General anesthesia, sleep, and coma. *New England Journal of Medicine*, 363(27), 2638-2650.

Loss of Responsiveness (LOR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: The point during sedation at which a subject stops responding to external commands (e.g., hand squeezing on auditory cue). Defined as a TR (timepoint) index in the fMRI data.

**Context in this project**: LOR marks the boundary between conscious (pre-LOR) and unconscious (post-LOR) segments of each subject's scan. A 375-TR buffer is applied around the transition to exclude ambiguous periods.

Propofol
^^^^^^^^
**Definition**: A widely used intravenous anesthetic agent (2,6-diisopropylphenol) that produces rapid onset of unconsciousness by enhancing GABA_A receptor activity.

**Properties**:

* Rapid onset (30-60 seconds)
* Short duration (5-10 minutes without continuous infusion)
* Dose-dependent effects: low doses → sedation, high doses → general anesthesia
* Reversible and relatively safe

**References**:

* Franks, N. P. (2008). General anaesthesia: from molecular targets to neuronal pathways of sleep and arousal. *Nature Reviews Neuroscience*, 9(5), 370-386.
* Voss, L. J., & Sleigh, J. W. (2007). Monitoring consciousness: the current status of EEG-based depth of anaesthesia monitors. *Best Practice & Research Clinical Anaesthesiology*, 21(3), 313-325.

Return of Responsiveness (ROR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: The point during recovery from sedation at which a subject resumes responding to external commands. Defined as a TR index.

**Context in this project**: ROR marks the boundary between the unconscious segment and the recovery segments. Subject sub-29 has no post-ROR data.

Sedation Levels
^^^^^^^^^^^^^^^
**Definition**: Gradations of consciousness depression from full awareness to general anesthesia.

**Standard levels**:

1. **Minimal sedation (anxiolysis)**: Normal response to verbal stimulation
2. **Moderate sedation**: Purposeful response to verbal/tactile stimulation
3. **Deep sedation**: Purposeful response following repeated/painful stimulation
4. **General anesthesia**: No response even to painful stimulation

Disorders of Consciousness
---------------------------

Disorders of Consciousness
^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A spectrum of conditions characterized by impaired arousal and/or awareness following severe brain injury.

**Main categories**:

1. **Coma**: No eye opening, no arousal
2. **Unresponsive Wakefulness Syndrome (UWS)**: Eyes open, but no awareness (formerly "vegetative state")
3. **Minimally Conscious State (MCS)**: Inconsistent but reproducible signs of awareness
4. **Emerged from MCS**: Functional communication or object use

**Context in this project**: Research application in disorders of consciousness.

**References**:

* Giacino, J. T., et al. (2002). The minimally conscious state: definition and diagnostic criteria. *Neurology*, 58(3), 349-353.
* Laureys, S., et al. (2010). Unresponsive wakefulness syndrome: a new name for the vegetative state or apallic syndrome. *BMC Medicine*, 8(1), 68.

Intraoperative Awareness
^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: The unintended occurrence of consciousness during general anesthesia, where patients experience awareness during surgery despite appearing adequately anesthetized.

**Context in this project**: Research application in anesthesia monitoring.

**References**:

* Mashour, G. A., et al. (2012). Prevention of intraoperative awareness with explicit recall in an unselected surgical population. *Anesthesiology*, 117(4), 717-725.

Minimally Conscious State (MCS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A condition of severely altered consciousness where patients show inconsistent but reproducible signs of awareness, such as following simple commands, yes/no responses, or intelligible verbalization.

**Distinguishing features from UWS**:

* Purposeful behavior (even if inconsistent)
* Command following
* Visual pursuit
* Appropriate emotional responses

**References**:

* Giacino, J. T., et al. (2002). The minimally conscious state: definition and diagnostic criteria. *Neurology*, 58(3), 349-353.

Vegetative State
^^^^^^^^^^^^^^^^
**Definition**: A condition of wakefulness without awareness. Patients show sleep-wake cycles and may exhibit reflexive movements, but demonstrate no purposeful responses or signs of consciousness.

**Modern term**: Unresponsive Wakefulness Syndrome (UWS) to avoid pejorative connotations.

**Key features**:

* Eyes open spontaneously
* No purposeful behavior
* No language comprehension or expression
* No sustained visual pursuit
* Preserved autonomic functions (breathing, circulation)

**Diagnostic challenge**: Clinical diagnosis relies on behavioral assessment which has limitations.

**References**:

* The Multi-Society Task Force on PVS (1994). Medical aspects of the persistent vegetative state. *New England Journal of Medicine*, 330(21), 1499-1508.
* Monti, M. M., et al. (2010). Willful modulation of brain activity in disorders of consciousness. *New England Journal of Medicine*, 362(7), 579-589.

Experimental Paradigm
---------------------

Mental Imagery
^^^^^^^^^^^^^^
**Definition**: The deliberate generation of sensory experiences in the mind without external stimuli — for example, imagining playing tennis or navigating through a familiar house.

**Context in this project**: Subjects performed mental imagery tasks during fMRI scanning. The key finding from Huang et al. (2018) is that some subjects showed brain activation during imagery tasks despite being behaviourally unresponsive, indicating covert consciousness.

**References**:

* Owen, A. M., et al. (2006). Detecting awareness in the vegetative state. *Science*, 313(5792), 1402.


Machine Learning
================

Core Concepts
-------------

Balanced Class Weights
^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A training strategy that adjusts the importance of each class in proportion to how rare it is. This prevents the model from being biased towards the more frequent class.

**Context in this project**: All three classifiers use balanced class weights because there are more conscious than unconscious samples.

Classification
^^^^^^^^^^^^^^
**Definition**: A supervised learning task where the goal is to predict discrete category labels (classes) from input features.

**In this project**: Binary classification (conscious vs. unconscious) or multi-class classification (sedation levels).

Cross-Subject Generalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: The ability of a model trained on some subjects to accurately predict outcomes for new, previously unseen subjects. Critical for clinical deployment.

**Challenge**: Inter-subject variability in brain anatomy and function makes this difficult in neuroimaging.

**Our approach**: Leave-One-Subject-Out Cross-Validation (LOSO-CV) to rigorously test generalization.

Feature
^^^^^^^
**Definition**: A measurable property or characteristic extracted from raw data and used as input to machine learning models. Good features capture relevant information while reducing dimensionality.

**In this project**: Features include connectivity matrices, graph metrics, and temporal dynamics extracted from fMRI data.

Ground Truth
^^^^^^^^^^^^
**Definition**: The true, known labels used to train and evaluate supervised learning models. In this project, ground truth comes from behavioral assessments of responsiveness.

**Challenge**: Ground truth based on behavioral assessment.

LOSO-CV (Leave-One-Subject-Out Cross-Validation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A validation strategy where the model is trained on N-1 subjects and tested on the held-out subject, repeating for each subject. Provides unbiased estimate of generalization to new individuals.

**Why critical for neuroimaging**: Ensures the model learns subject-independent patterns, not individual-specific idiosyncrasies.

Classifiers
-----------

Logistic Regression
^^^^^^^^^^^^^^^^^^^
**Definition**: A linear classifier that draws a straight boundary (in high-dimensional feature space) to separate two classes. Each feature receives a weight; the model sums the weighted features and applies a threshold to produce a probability.

**Context in this project**: One of the three classifiers used for consciousness detection. Its weights are directly interpretable — they indicate which brain features are most associated with each state.

Random Forest
^^^^^^^^^^^^^
**Definition**: An ensemble classifier that builds many decision trees, each trained on a random subset of the data and features. The final prediction is the majority vote across all trees.

**Context in this project**: One of the three classifiers. Provides feature importance scores showing which brain connections contribute most to classification.

Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A classifier that finds the boundary between classes with the widest possible margin. It can draw curved boundaries using kernel functions (e.g., RBF kernel).

**Context in this project**: One of the three classifiers. Uses the RBF kernel to capture non-linear relationships in the data.


Brain Anatomy & Atlases
=======================

Key Brain Regions
-----------------

Anterior Insula
^^^^^^^^^^^^^^^
**Definition**: A region of the cerebral cortex located deep within the lateral sulcus (Sylvian fissure), involved in interoception, emotion, and salience detection.

**Role in consciousness**: The anterior insula is a critical hub in the "salience network" that switches between internal and external attention. Huang et al. (2018) found insula connectivity predicts covert consciousness.

**References**:

* Craig, A. D. (2009). How do you feel—now? The anterior insula and human awareness. *Nature Reviews Neuroscience*, 10(1), 59-70.
* Huang, Z., et al. (2021). Anterior insula regulates brain network transitions that gate conscious access. *Cell Reports*, 35(5), 109081.

Default Mode Network (DMN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A large-scale brain network including medial prefrontal cortex, posterior cingulate cortex, and lateral parietal regions that is more active during rest than during externally-focused tasks.

**Functions**: Self-referential thought, memory, future planning, mind-wandering.

**References**:

* Raichle, M. E., et al. (2001). A default mode of brain function. *Proceedings of the National Academy of Sciences*, 98(2), 676-682.
* Buckner, R. L., et al. (2008). The brain's default network: anatomy, function, and relevance to disease. *Annals of the New York Academy of Sciences*, 1124(1), 1-38.

Atlases
-------

4S456Parcels Atlas
^^^^^^^^^^^^^^^^^^
**Definition**: A whole-brain atlas that divides the brain into 456 regions by combining two established parcellations: 400 cortical regions from the Schaefer atlas and 56 subcortical regions from the Tian atlas.

**Context in this project**: The code uses the first 446 of the 456 parcels, following the approach of the reference paper.

Schaefer Atlas
^^^^^^^^^^^^^^
**Definition**: A cortical parcellation that divides the cerebral cortex into regions based on functional connectivity patterns observed in resting-state fMRI data. Available in multiple resolutions (100, 200, 400, 600, 800, 1000 parcels).

**Context in this project**: The 400-parcel version is used as the cortical component of the 4S456Parcels atlas.

**References**:

* Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.

Tian Atlas
^^^^^^^^^^
**Definition**: A subcortical parcellation that divides subcortical structures (thalamus, hippocampus, amygdala, striatum, etc.) into functionally distinct regions based on resting-state fMRI connectivity.

**Context in this project**: The 56-region version provides the subcortical component of the 4S456Parcels atlas.

**References**:

* Tian, Y., et al. (2020). Topographic organization of the human subcortex unveiled with functional connectivity gradients. *Nature Neuroscience*, 23(11), 1421-1432.


Statistical Measures
====================

Performance Metrics
-------------------

AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Definition**: A performance metric that measures the model's ability to discriminate between classes across all classification thresholds. Ranges from 0.5 (random) to 1.0 (perfect).

False Negative Rate
^^^^^^^^^^^^^^^^^^^
**Definition**: The proportion of positive instances incorrectly classified as negative.

False Positive Rate
^^^^^^^^^^^^^^^^^^^
**Definition**: The proportion of negative instances incorrectly classified as positive.

Sensitivity (Recall)
^^^^^^^^^^^^^^^^^^^^
**Definition**: The proportion of actual positive cases correctly identified. Formally: TP / (TP + FN), where TP = true positives, FN = false negatives.

Specificity
^^^^^^^^^^^
**Definition**: The proportion of actual negative cases correctly identified. Formally: TN / (TN + FP), where TN = true negatives, FP = false positives.

Correlation & Association
-------------------------

Covariance
^^^^^^^^^^
**Definition**: A measure of how two variables change together. Positive covariance means both variables tend to increase together; negative means one increases while the other decreases. Related to correlation but not normalised by the variables' spread.

**In connectivity analysis**: Forms the basis for computing correlation matrices.

Pearson Correlation
^^^^^^^^^^^^^^^^^^^
**Definition**: A measure of linear association between two variables, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).

**In neuroimaging**: Measures how synchronously two brain regions' activity fluctuates over time. Values range from -1 (perfectly anti-correlated) through 0 (no relationship) to +1 (perfectly correlated).


Acronyms Quick Reference
=========================

.. glossary::

   BIDS
      Brain Imaging Data Structure

   BOLD
      Blood-Oxygen-Level-Dependent (imaging signal)

   DMN
      Default Mode Network

   DoC
      Disorders of Consciousness

   FD
      Framewise Displacement

   fMRI
      Functional Magnetic Resonance Imaging

   ISD
      Integration-Segregation Difference

   LOR
      Loss of Responsiveness

   LOSO-CV
      Leave-One-Subject-Out Cross-Validation

   MCS
      Minimally Conscious State

   MRI
      Magnetic Resonance Imaging

   ROI
      Region of Interest

   ROR
      Return of Responsiveness

   SVM
      Support Vector Machine

   TR
      Repetition Time

   UWS
      Unresponsive Wakefulness Syndrome (formerly vegetative state)


Additional Resources
====================

**Comprehensive References:**

1. **Neuroscience Reference Works:**

   * Squire, L. R., et al. (Eds.). (2013). *Fundamental Neuroscience* (4th ed.). Academic Press.
   * Kandel, E. R., et al. (2013). *Principles of Neural Science* (5th ed.). McGraw-Hill.

2. **Network Neuroscience:**

   * Sporns, O. (2016). *Networks of the Brain*. MIT Press.
   * Bullmore, E., & Sporns, O. (2012). The economy of brain network organization. *Nature Reviews Neuroscience*, 13(5), 336-349.

3. **Consciousness Research:**

   * Laureys, S., & Tononi, G. (Eds.). (2009). *The Neurology of Consciousness*. Academic Press.

4. **Neuroimaging Analysis:**

   * Poldrack, R. A., et al. (2011). *Handbook of Functional MRI Data Analysis*. Cambridge University Press.
   * Huettel, S. A., et al. (2014). *Functional Magnetic Resonance Imaging* (3rd ed.). Sinauer Associates.

**Online Resources:**

* Brain Imaging Data Structure (BIDS): https://bids.neuroimaging.io/
* NeuroVault: https://neurovault.org/ (brain maps repository)
* OpenNeuro: https://openneuro.org/ (open neuroimaging datasets)
* Human Connectome Project: https://www.humanconnectome.org/

.. note::
   All definitions in this glossary are based on peer-reviewed scientific literature. For the most current understanding of these concepts, consult recent publications in journals such as *Nature Neuroscience*, *Neuron*, *Science*, and *Trends in Cognitive Sciences*.
