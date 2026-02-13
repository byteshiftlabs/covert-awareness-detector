============
Installation
============

This guide will help you install the Covert Awareness Detector and all its dependencies. Follow the steps carefully to ensure a smooth setup.

.. contents:: Table of Contents
   :local:
   :depth: 2


Tested Environment
==================

This software has been tested on **Ubuntu 22.04 LTS on WSL (Windows Subsystem for Linux)**.


Installation Steps
==================

Step 1: Install WSL Ubuntu
---------------------------

1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Ubuntu 22.04 LTS from Microsoft Store
3. Open Ubuntu terminal


Step 2: Install Python
----------------------

.. code-block:: bash

   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip
   python3.10 --version


Step 3: Clone the Repository
-----------------------------

.. code-block:: bash

   sudo apt install git
   git clone https://github.com/yourusername/consciousness_detector.git
   cd consciousness_detector


Step 4: Create Virtual Environment
-----------------------------------

.. code-block:: bash

   python3.10 -m venv venv
   source venv/bin/activate


Step 5: Upgrade pip
-------------------

.. code-block:: bash

   pip install --upgrade pip setuptools wheel


Step 6: Install Dependencies
-----------------------------

.. code-block:: bash

   pip install -r requirements.txt


Step 7: Verify Installation
----------------------------

.. code-block:: bash

   python -c "import numpy, scipy, pandas, nibabel, nilearn, sklearn, torch; print('✓ Installation successful')"


Troubleshooting
===============

If you encounter issues:

.. code-block:: bash

   # Make sure virtual environment is activated
   which python  # Should point to venv/bin/python
   
   # Reinstall dependencies if needed
   pip install --force-reinstall -r requirements.txt





Next Steps
==========

See :doc:`quickstart` for dataset download and model training.
