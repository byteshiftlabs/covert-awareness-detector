============
Introduction
============

What is Covert Consciousness?
==============================

**Covert consciousness** (also called *covert awareness* or *hidden consciousness*) is the presence of subjective experience in the absence of behavioral response (Huang et al., 2026).

   Huang, Z., Tarnal, V., Fotiadis, P., Vlisides, P. E., Janke, E. L., Puglia, M., McKinney, A. M., Jang, H., Dai, R., Picton, P., Mashour, G. A., & Hudetz, A. G. (2026). An open fMRI resource for studying human brain function and covert consciousness under anesthesia. *Scientific Data*, *13*\(1), Article 127. https://doi.org/10.1038/s41597-025-06442-2

Traditional Assessment Challenge
---------------------------------

Consider a patient under sedation:

* **Doctor asks**: "If you can hear me, squeeze my hand"
* **Patient's response**: No hand movement, no behavioral sign
* **Standard conclusion**: Patient is unconscious

But what if their brain is actually processing the command and they're consciously following mental imagery instructions, just unable to produce a motor response?

This separation between **neural awareness** and **behavioral responsiveness** is covert consciousness.


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

This project implements a **machine learning pipeline** to automate consciousness detection from fMRI data, building on the scientific foundations laid by the Michigan team.

**What We Provide:**

* **Automated classification** of consciousness states
* **Cross-subject generalization** using ensemble learning (XGBoost)
* **Production-ready code** for research and analysis
* **Extensible framework** for new models and analyses


Approach Overview
=================

Our machine learning pipeline transforms fMRI brain scans into automated consciousness predictions:

1. **Input**: fMRI data from the Michigan Human Anesthesia Dataset (25 subjects under propofol sedation)
2. **Feature Extraction**: Extract connectivity patterns and network metrics from brain activity
3. **Classification**: Train ensemble classifier (XGBoost) to predict conscious vs. unconscious states
4. **Output**: Automated consciousness detection

.. seealso::
   
   * :doc:`dataset` - dataset details
   * :doc:`feature_extraction` - feature methods
   * :doc:`model_architecture` - model details


Our Contribution
================

A production‑ready Python pipeline that automates consciousness detection from fMRI (feature extraction, ML training, LOSO evaluation) with cross‑subject validation. The underlying neuroscience, experimental design, and dataset are credited to Huang et al. (2026).

.. important::
   **Philosophy**: This project aims to **engineer solutions** based on **established neuroscience**, not to claim credit for scientific discoveries made by domain experts.


Target Audience
===============

This is a research implementation of fMRI-based consciousness detection using machine learning.

**Machine Learning Practitioners**
   Machine learning pipeline for fMRI data classification.

**Neuroscience Researchers**
   Analysis tools for fMRI connectivity data and consciousness research.

**Clinical Researchers**
   Research tool for studying consciousness assessment methods.

**Data Scientists & Students**
   Applied ML implementation with functional connectivity and network analysis.

.. danger::
   **CRITICAL DISCLAIMER**: This is a research tool, NOT a medical device. This software:
   
   * Requires extensive clinical validation before any medical use
   * Is provided for research and educational purposes only
   * Should not be used for patient diagnosis or treatment decisions unless validated and certified by appropriate regulatory authorities
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
