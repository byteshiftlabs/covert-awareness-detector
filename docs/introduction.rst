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

Our machine learning pipeline transforms fMRI brain scans into automated consciousness predictions:

1. **Input**: fMRI data from the Michigan Human Anesthesia Dataset (26 subjects under propofol sedation)
2. **Feature Extraction**: Extract connectivity patterns and network metrics from brain activity
3. **Classification**: Train machine learning models to predict conscious vs. unconscious states
4. **Output**: Automated consciousness detection with interpretability

.. seealso::
   
   * :doc:`dataset` - Detailed information about the fMRI data
   * :doc:`feature_extraction` - Complete feature extraction methods
   * :doc:`model_architectures` - All model architectures and comparisons


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

This project is designed for:

**Machine Learning Practitioners**
   Apply deep learning to neuroscience problems. No neuroscience background required - we explain the domain concepts.

**Neuroscience Researchers**
   Automated analysis of fMRI connectivity data using modern ML techniques. Replicate and extend published consciousness research.

**Clinical Researchers**
   Research tools for consciousness assessment. Potential applications in anesthesia monitoring and disorders of consciousness.

**Data Scientists & Students**
   Complete example of applied ML on challenging real-world data with graph neural networks and time-series analysis.

.. danger::
   **CRITICAL DISCLAIMER**: This is a research tool, NOT a medical device. This software:
   
   * Requires extensive clinical validation before any medical use
   * Is provided for research and educational purposes only
   * Should NEVER be used for patient diagnosis or treatment decisions
   * Comes with NO WARRANTY of any kind
   
   **The authors and contributors accept NO RESPONSIBILITY OR LIABILITY for any misuse, harm, or adverse outcomes resulting from the use of this software. Use at your own risk.**


Next Steps
==========

Ready to get started?

1. :doc:`installation` - Set up the software
2. :doc:`dataset` - Understand the fMRI data
3. :doc:`model_architectures` - Explore the models

.. note::
   This is an open-source research project. Questions and contributions welcome via GitHub!
