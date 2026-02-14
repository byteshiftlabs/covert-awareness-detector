============
Introduction
============

What is Covert Consciousness?
==============================

**Covert consciousness** (also called *covert awareness* or *hidden consciousness*) is a phenomenon where patients show neural signatures of awareness and can follow mental commands, yet exhibit **no behavioral response** - appearing completely unresponsive to external observers.


Traditional Assessment Challenge
---------------------------------

Consider a patient under sedation:

* **Doctor asks**: "If you can hear me, squeeze my hand"
* **Patient's response**: No hand movement, no behavioral sign
* **Standard conclusion**: Patient is unconscious

But what if their brain is actually processing the command and they're consciously following mental imagery instructions, just unable to produce a motor response?

This dissociation between **neural awareness** and **behavioral responsiveness** is covert consciousness.


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
4. **Output**: Automated consciousness detection

.. seealso::
   
   * :doc:`dataset` — dataset details
   * :doc:`feature_extraction` — feature methods
   * :doc:`model_architecture` — model details


Relation to Original Research
==============================

Scientific Foundation
---------------------

This project **builds upon** the neuroscience established by:

**Huang, Hudetz, Mashour et al.** at University of Michigan

Key publication:

- **Huang et al. (2018)** - *Scientific Reports*

  *"Brain imaging reveals covert consciousness during behavioral unresponsiveness induced by propofol"*

  * Discovered covert consciousness in 19% of subjects
  * Identified anterior insula as key region
  * Established mental imagery paradigm

.. note::
   **Credit**: All scientific discoveries, experimental design, and dataset creation belong to the University of Michigan team. Their MATLAB analysis code is available at: https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation


Our Contribution
----------------

A concise, production‑ready Python pipeline that automates consciousness detection from fMRI (feature extraction, ML training, LOSO evaluation) with cross‑subject validation. The underlying neuroscience, experimental design, and dataset are credited to Huang et al. (2018).

.. important::
   **Philosophy**: This project aims to **engineer solutions** based on **established neuroscience**, not to claim credit for scientific discoveries made by domain experts.


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
3. :doc:`model_architecture` - Explore the models

.. note::
   This is an open-source research project. Questions and contributions welcome via GitHub!
