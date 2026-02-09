============
Introduction
============

What is Covert Consciousness?
==============================

**Covert consciousness** (also called *covert awareness* or *hidden consciousness*) is a phenomenon where patients show neural signatures of awareness and can follow mental commands, yet exhibit **no behavioral response** - appearing completely unresponsive to external observers.

.. image:: _static/covert_consciousness_diagram.png
   :alt: Diagram showing covert consciousness
   :align: center
   :width: 600px

*(Illustration: A patient appears behaviorally unresponsive but shows brain activation during mental imagery tasks)*

Traditional Assessment Challenge
---------------------------------

Consider a patient under sedation:

* **Doctor asks**: "If you can hear me, squeeze my hand"
* **Patient's response**: No hand movement, no behavioral sign
* **Standard conclusion**: Patient is unconscious

But what if their brain is actually processing the command and they're consciously following mental imagery instructions, just unable to produce a motor response?

This dissociation between **neural awareness** and **behavioral responsiveness** is covert consciousness - first systematically characterized by Huang, Hudetz, Mashour and colleagues at the University of Michigan (2018).

.. note::
   **Key Insight from Huang et al. (2018)**: Out of 26 healthy volunteers under propofol sedation, **5 subjects (19%)** showed clear brain activation patterns during mental imagery tasks (imagining playing tennis, spatial navigation) despite showing **zero behavioral response** when asked to squeeze a hand.

Why This Matters
----------------

**Clinical Significance:**

1. **Anesthesia Awareness**: Detecting hidden consciousness during surgery when patients appear adequately sedated
2. **Disorders of Consciousness**: Identifying cognitive function in patients with severe brain injury who cannot communicate
3. **Minimally Conscious State**: Distinguishing between vegetative state and minimal consciousness
4. **Personalized Medicine**: Tailoring sedation levels based on neural rather than purely behavioral indicators

**Real-World Impact:**

* **Preventing intraoperative awareness** - a traumatic experience where patients recall surgery
* **Improving prognosis** - correctly identifying consciousness in non-responsive patients
* **Ethical decision-making** - informing care decisions for patients with disorders of consciousness
* **Optimizing recovery** - adjusting treatments based on neural markers


The Problem This Project Solves
================================

Current Limitations
-------------------

**Manual Analysis Challenges:**

The groundbreaking research by Huang et al. established the scientific foundation through careful manual analysis of fMRI data:

* **Time-intensive**: Each patient requires hours of expert analysis
* **Subjective**: Requires trained neuroscientists to interpret connectivity patterns
* **Not real-time**: Post-hoc analysis, not suitable for clinical monitoring
* **Limited scalability**: Cannot be deployed across hospitals at scale

**Clinical Gap:**

While the neuroscience demonstrates that covert consciousness can be detected, there's no **automated, deployable system** that:

1. Works across different subjects (generalization)
2. Operates in real-time or near-real-time
3. Provides interpretable predictions for clinicians
4. Requires minimal expert supervision

.. warning::
   This is a critical gap: We have the science, but lack the engineering to translate it into clinical practice.


Our Solution
------------

This project implements **deep learning models** to automate consciousness detection from fMRI data, building on the scientific foundations laid by the Michigan team.

**What We Provide:**

* **Automated classification** of consciousness states
* **Cross-subject generalization** using neural networks
* **Interpretable predictions** with attention mechanisms
* **Production-ready code** for deployment
* **Extensible framework** for new models and analyses


Approach Overview
=================

The Pipeline
------------

Our machine learning pipeline transforms raw fMRI data into consciousness predictions:

.. code-block:: text

   ┌─────────────────┐
   │  fMRI BOLD Data │  (4D: x,y,z,time)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────┐
   │  Preprocessing          │
   │  • Motion correction    │
   │  • Spatial normalization│
   │  • Denoising           │
   └────────┬────────────────┘
            │
            ▼
   ┌─────────────────────────┐
   │  Feature Extraction     │
   │  • Connectivity matrices│
   │  • ROI time-series     │
   │  • Network metrics     │
   └────────┬────────────────┘
            │
            ▼
   ┌─────────────────────────┐
   │  ML Models             │
   │  • Random Forest       │
   │  • CNNs                │
   │  • Graph Neural Nets   │
   └────────┬────────────────┘
            │
            ▼
   ┌─────────────────────────┐
   │  Consciousness State   │
   │  • Conscious           │
   │  • Unconscious         │
   │  • Covert Awareness    │
   └─────────────────────────┘

Input Data: fMRI Scans
----------------------

**What we analyze:**

* **Resting-state fMRI**: Brain activity without external task
* **Task-based fMRI**: Mental imagery during sedation (tennis, navigation)
* **Temporal resolution**: ~2 seconds per brain volume
* **Spatial resolution**: 2-3mm voxels
* **Duration**: 5-10 minutes per recording

**Data source**: Michigan Human Anesthesia fMRI Dataset (OpenNeuro ds006623)

* 26 healthy volunteers
* Graded propofol sedation (Awake → Mild → Moderate → Deep → Recovery)
* Mental imagery tasks at each sedation level
* Behavioral assessments (responsiveness tests)


Feature Extraction
------------------

We extract multiple representations from fMRI data:

**1. Functional Connectivity Matrices**

Correlation between brain regions across time:

.. math::

   C_{ij} = \text{corr}(\text{ROI}_i(t), \text{ROI}_j(t))

* Results in N×N matrix (N = number of brain regions)
* Captures network-level organization
* Basis for graph neural network approaches

**2. ROI Time-Series**

Raw temporal dynamics from brain regions:

* Direct input to recurrent neural networks (LSTMs, GRUs)
* Preserves temporal information
* Enables sequence modeling

**3. Graph-Theoretic Metrics**

Network neuroscience features:

* **Clustering coefficient**: Local network organization
* **Path length**: Global network integration  
* **Modularity**: Community structure
* **Hub detection**: Critical network nodes

**4. Spectral Features**

Frequency-domain properties:

* Power in different frequency bands
* Coherence between regions
* Non-linear dynamics


Machine Learning Models
-----------------------

We implement and compare multiple architectures:

**Baseline Models:**

* **Logistic Regression**: Linear baseline on connectivity features
* **Random Forest**: Non-linear baseline with feature importance
* **SVM**: Support vector machines on engineered features

**Deep Learning:**

* **CNN (Convolutional Neural Networks)**: 
  * Treat connectivity matrices as images
  * Learn hierarchical spatial patterns
  * Rotation/reflection invariant features

* **RNN (Recurrent Neural Networks)**:
  * Process time-series sequences
  * Capture temporal dynamics
  * LSTM/GRU architectures for long-term dependencies

* **GNN (Graph Neural Networks)**:
  * Directly model brain network structure
  * Message passing between ROIs
  * Preserve graph topology

* **Hybrid Architectures**:
  * Combine spatial + temporal + graph information
  * Attention mechanisms for interpretability
  * Multi-modal fusion


Output: Consciousness Classification
-------------------------------------

**Primary Task**: Binary Classification

* **Class 0**: Unconscious (behaviorally unresponsive)
* **Class 1**: Conscious (behaviorally responsive)

**Secondary Tasks**:

* **Multi-class**: Awake / Mild / Moderate / Deep / Recovery
* **Covert detection**: Identifying hidden consciousness
* **Regression**: Predicting sedation depth (propofol concentration)


Relation to Original Research
==============================

Scientific Foundation
---------------------

This project **builds upon** the neuroscience established by:

**Huang, Hudetz, Mashour et al.** at University of Michigan

Key Publications:

1. **Huang et al. (2018)** - *Scientific Reports*
   
   *"Brain imaging reveals covert consciousness during behavioral unresponsiveness induced by propofol"*
   
   * Discovered covert consciousness in 19% of subjects
   * Identified anterior insula as key region
   * Established mental imagery paradigm

2. **Huang et al. (2021)** - *Cell Reports*
   
   *"Anterior insula regulates brain network transitions that gate conscious access"*
   
   * Insula connectivity predicts conscious access
   * Dynamic network transitions during sedation
   * Mechanistic understanding of consciousness gating

3. **Huang et al. (2021)** - *NeuroImage*
   
   *"Asymmetric neural dynamics characterize loss and recovery of consciousness"*
   
   * Neural hysteresis in consciousness transitions
   * Different pathways for loss vs. recovery
   * Time-varying network dynamics

.. note::
   **Credit**: All scientific discoveries, experimental design, and dataset creation belong to the University of Michigan team. Their MATLAB analysis code is available at: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation


Our Contribution
----------------

This project provides the **machine learning engineering** to translate their findings into deployable systems:

**What's Original Here:**

* **Python reimplementation** using modern ML frameworks (PyTorch/TensorFlow)
* **Deep learning models** beyond traditional connectivity analysis
* **Cross-subject generalization** for clinical deployment
* **Automated detection** without manual analysis
* **Production-ready code** with testing and documentation

**What's Not Original:**

* The neuroscientific discoveries (credit: Huang et al.)
* The dataset and experimental design (credit: Michigan team)
* The theoretical framework for covert consciousness (neuroscience literature)

.. important::
   **Philosophy**: We stand on the shoulders of giants. This project aims to **engineer solutions** based on **established neuroscience**, not to claim credit for scientific discoveries made by domain experts.


Target Audience
===============

This project is designed for multiple audiences:

Machine Learning Practitioners
------------------------------

**What you'll find:**

* End-to-end ML pipeline from data loading to deployment
* Multiple model architectures to learn from
* Comparison of classical vs. deep learning approaches
* Best practices for neuroimaging ML

**Background needed:**

* Python programming
* Basic ML/DL concepts (CNNs, RNNs, training loops)
* Familiarity with PyTorch or TensorFlow
* No neuroscience background required - we explain!

.. tip::
   If you're an ML engineer curious about applying your skills to neuroscience, this is a great entry point. The code is well-documented and follows standard ML practices.


Neuroscience Researchers
-------------------------

**What you'll find:**

* Automated analysis of fMRI connectivity data
* Replication of Huang et al.'s manual analysis with ML
* Extended analyses using modern neural networks
* Tools for your own consciousness research

**Background needed:**

* Understanding of fMRI and connectivity analysis
* Basic Python programming
* Willingness to learn ML concepts (we provide tutorials)

.. tip::
   We provide bridges between neuroscience concepts and ML terminology. The models implement well-established neuroscience principles in a computational framework.


Clinical Researchers
--------------------

**What you'll find:**

* Potential tools for consciousness assessment
* Automated detection pipelines
* Interpretable predictions for clinical decision-support
* Validation on published dataset

**Background needed:**

* Clinical experience with disorders of consciousness or anesthesia
* Understanding of neuroimaging
* Basic data analysis skills

.. warning::
   **Important**: This is a research tool, not a medical device. All models require extensive validation before any clinical use. This code is for research purposes only.


Data Scientists & AI Researchers
---------------------------------

**What you'll find:**

* Interesting domain for graph neural networks
* Time-series classification challenges
* Small dataset learning techniques
* Interpretable AI in safety-critical domain

**Background needed:**

* Strong ML/DL foundations
* Experience with neural networks
* Interest in challenging real-world applications


Students & Educators
--------------------

**What you'll find:**

* Complete example of applied ML project
* Clean code structure following best practices
* Educational documentation and tutorials
* Opportunity to contribute to neuroscience

**Background needed:**

* Undergraduate-level ML knowledge
* Python programming
* Curiosity and willingness to learn


Next Steps
==========

Now that you understand what covert consciousness is and what this project does:

1. **Install the software**: See :doc:`installation` for setup instructions
2. **Run your first model**: Follow the :doc:`quickstart` tutorial
3. **Explore the API**: Check :doc:`api` for detailed documentation
4. **Contribute**: See :doc:`contributing` to help improve the project


.. seealso::
   
   **Additional Resources:**
   
   * :doc:`architecture` - Detailed model architectures
   * :doc:`dataset` - Deep dive into the fMRI data
   * :doc:`evaluation` - How we measure performance
   * :doc:`deployment` - Using models in production


Questions?
----------

* **GitHub Issues**: https://github.com/yourusername/consciousness_detector/issues
* **Email**: Contact the maintainer
* **Discussions**: Join our community forum

.. note::
   This is an open-source research project. We welcome questions, contributions, and collaboration!
