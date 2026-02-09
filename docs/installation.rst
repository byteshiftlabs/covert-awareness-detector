============
Installation
============

This guide will help you install the Covert Awareness Detector and all its dependencies. Follow the steps carefully to ensure a smooth setup.

.. contents:: Table of Contents
   :local:
   :depth: 2


System Requirements
===================

Minimum Requirements
--------------------

**Hardware:**

* **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
* **RAM**: 16GB minimum, 32GB recommended
* **Storage**: 50GB free space (20GB for dataset, 30GB for models/results)
* **GPU**: Optional but highly recommended (NVIDIA with CUDA support)

**Software:**

* **Python**: 3.8, 3.9, 3.10, or 3.11
* **pip**: 20.0 or higher
* **git**: For cloning the repository

**Operating Systems:**

* **Linux**: Ubuntu 20.04+, Debian 10+, CentOS 8+, Arch Linux (recommended)
* **macOS**: 11.0 (Big Sur) or later (M1/M2 supported)
* **Windows**: 10/11 with WSL2 (Windows Subsystem for Linux)

.. warning::
   **Windows Users**: We strongly recommend using WSL2 (Ubuntu 20.04+) rather than native Windows. Many neuroimaging tools work better on Linux.


Recommended System
------------------

For optimal performance:

* **CPU**: 8+ cores (Intel i7/i9, AMD Ryzen 7/9)
* **RAM**: 32GB or more
* **GPU**: NVIDIA RTX 3060 or better with 12GB+ VRAM
* **Storage**: 100GB+ SSD
* **OS**: Ubuntu 22.04 LTS

GPU Acceleration
----------------

Deep learning models train **significantly faster** with GPU support:

* **Without GPU**: ~2-4 hours per model
* **With GPU**: ~15-30 minutes per model

**NVIDIA GPU Setup:**

.. code-block:: bash

   # Check if you have an NVIDIA GPU
   lspci | grep -i nvidia
   
   # Check CUDA version (if installed)
   nvidia-smi

If you have an NVIDIA GPU, make sure CUDA is installed (CUDA 11.7+ recommended for PyTorch 2.0+).

**AMD/Apple Silicon:**

While possible, GPU acceleration is less mature. The project will work on CPU for these systems.


Installation Steps
==================

Step 1: Install Python
-----------------------

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   # Update package list
   sudo apt update
   
   # Install Python 3.10 (recommended)
   sudo apt install python3.10 python3.10-venv python3.10-dev
   
   # Install pip
   sudo apt install python3-pip
   
   # Verify installation
   python3.10 --version
   pip3 --version

**Linux (Arch):**

.. code-block:: bash

   sudo pacman -S python python-pip
   python --version

**macOS:**

.. code-block:: bash

   # Using Homebrew (install from https://brew.sh if needed)
   brew install python@3.10
   python3.10 --version

**Windows (WSL2):**

1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Ubuntu 20.04+ from Microsoft Store
3. Open Ubuntu terminal and follow Linux instructions above


Step 2: Clone the Repository
-----------------------------

.. code-block:: bash

   # Navigate to your projects directory
   cd ~/Projects  # or wherever you keep code
   
   # Clone the repository
   git clone https://github.com/yourusername/consciousness_detector.git
   
   # Enter the project directory
   cd consciousness_detector
   
   # Verify you're in the right place
   ls -la  # Should see README.md, requirements.txt, src/, etc.

.. note::
   If you don't have git installed:
   
   * **Linux**: ``sudo apt install git``
   * **macOS**: ``brew install git``


Step 3: Create Virtual Environment
-----------------------------------

Virtual environments isolate project dependencies from your system Python.

**Using venv (recommended):**

.. code-block:: bash

   # Create virtual environment
   python3.10 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # Linux/macOS
   
   # Windows WSL2 (same as Linux):
   source venv/bin/activate
   
   # Your prompt should now show (venv)
   # Example: (venv) user@hostname:~/Projects/consciousness_detector$

**Using conda (alternative):**

.. code-block:: bash

   # Create environment with Python 3.10
   conda create -n consciousness python=3.10
   
   # Activate environment
   conda activate consciousness

.. tip::
   **Always activate your virtual environment** before working on the project. If you close your terminal, you'll need to activate it again.


Step 4: Upgrade pip
-------------------

.. code-block:: bash

   # Make sure venv is activated!
   # Upgrade pip to latest version
   pip install --upgrade pip setuptools wheel
   
   # Verify
   pip --version  # Should show 23.0+


Step 5: Install Dependencies
-----------------------------

**Basic Installation (CPU only):**

.. code-block:: bash

   # Install all required packages
   pip install -r requirements.txt
   
   # This will take 5-10 minutes

**With GPU Support (NVIDIA CUDA):**

.. code-block:: bash

   # Install PyTorch with CUDA 11.8 (adjust based on your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # Install remaining dependencies
   pip install -r requirements.txt

.. note::
   Check your CUDA version with ``nvidia-smi`` and match the PyTorch installation. See https://pytorch.org/get-started/locally/ for version compatibility.


**For Developers (additional tools):**

.. code-block:: bash

   # Install development dependencies
   pip install -r requirements-dev.txt  # If this file exists
   
   # Or manually install common dev tools:
   pip install pytest pytest-cov black flake8 mypy jupyter ipykernel

.. important::
   **Installation Size**: The full installation requires approximately:
   
   * PyTorch: ~2GB
   * Other packages: ~1.5GB
   * Total: ~3.5GB


Understanding Dependencies
--------------------------

Key packages and their purposes:

**Core Scientific Computing:**

* ``numpy``: Array operations and numerical computing
* ``scipy``: Statistical functions and optimization
* ``pandas``: Data manipulation and analysis

**Neuroimaging:**

* ``nibabel``: Reading/writing neuroimaging file formats (NIfTI)
* ``nilearn``: fMRI analysis and brain visualization
* ``pybids``: BIDS dataset handling

**Machine Learning:**

* ``scikit-learn``: Classical ML algorithms (Random Forest, SVM)
* ``torch`` / ``torchvision``: Deep learning framework

**Graph Neural Networks:**

* ``torch-geometric``: GNN implementations for brain connectivity
* ``networkx``: Graph theory algorithms
* ``python-igraph``: Fast graph analysis

**Visualization:**

* ``matplotlib`` / ``seaborn``: Static plots
* ``plotly``: Interactive visualizations

**Data Management:**

* ``datalad``: Dataset version control (optional)
* ``h5py``: Efficient data storage


Step 6: Verify Installation
----------------------------

Run the verification script to check everything is installed correctly:

.. code-block:: bash

   # Test imports
   python -c "import numpy, scipy, pandas, nibabel, nilearn, sklearn, torch; print('✓ All core packages imported successfully')"
   
   # Check PyTorch
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   
   # Check GPU (if you have one)
   python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

Expected output (CPU only):

.. code-block:: text

   ✓ All core packages imported successfully
   PyTorch version: 2.0.1
   CUDA available: False
   GPU device: CPU only

Expected output (with GPU):

.. code-block:: text

   ✓ All core packages imported successfully
   PyTorch version: 2.0.1+cu118
   CUDA available: True
   GPU device: NVIDIA GeForce RTX 3080

.. tip::
   If all tests pass, you're ready to go! Proceed to :doc:`quickstart`.


Troubleshooting
===============

Common Issues
-------------

**Issue 1: "No module named 'XXX'" after installation**

**Solution:**

.. code-block:: bash

   # Make sure virtual environment is activated
   which python  # Should point to venv/bin/python
   
   # Reinstall the missing package
   pip install <package-name>
   
   # If that doesn't work, reinstall all dependencies
   pip install --force-reinstall -r requirements.txt


**Issue 2: PyTorch CUDA version mismatch**

.. code-block:: text

   RuntimeError: CUDA error: no kernel image is available for execution on the device

**Solution:**

.. code-block:: bash

   # Check CUDA version
   nvidia-smi  # Look for "CUDA Version: X.X"
   
   # Uninstall existing PyTorch
   pip uninstall torch torchvision
   
   # Install PyTorch matching your CUDA version
   # For CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


**Issue 3: Out of memory errors**

.. code-block:: text

   RuntimeError: CUDA out of memory

**Solution:**

Reduce batch size in configuration:

.. code-block:: bash

   # Edit config file
   nano src/config.py
   
   # Change BATCH_SIZE from 32 to 16 or 8


**Issue 4: Nibabel/Nilearn import errors on macOS**

**Solution:**

.. code-block:: bash

   # Install system dependencies first
   brew install gcc hdf5
   
   # Then reinstall packages
   pip install --no-cache-dir nibabel nilearn


**Issue 5: "Permission denied" when creating directories**

**Solution:**

.. code-block:: bash

   # Make sure you have write permissions
   sudo chown -R $USER:$USER ~/Projects/consciousness_detector
   
   # Or run from your home directory
   cd ~/Projects/consciousness_detector


**Issue 6: Very slow installation**

**Solution:**

.. code-block:: bash

   # Use a faster mirror
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # Or upgrade pip first
   pip install --upgrade pip


**Issue 7: torch-geometric installation fails**

**Solution:**

.. code-block:: bash

   # Install PyTorch first
   pip install torch torchvision
   
   # Then install torch-geometric with wheels
   pip install torch-geometric
   
   # If that fails, install from source
   pip install torch-geometric --no-cache-dir


Platform-Specific Issues
------------------------

**Linux: Missing system libraries**

.. code-block:: bash

   # Install common dependencies
   sudo apt install build-essential python3-dev libhdf5-dev

**macOS: Apple Silicon (M1/M2) compatibility**

.. code-block:: bash

   # Use conda for better ARM support
   conda install pytorch torchvision -c pytorch
   
   # Then install other packages
   pip install -r requirements.txt

**Windows WSL2: Network issues**

.. code-block:: bash

   # If pip install fails with network errors
   # 1. Check WSL2 can access internet
   ping google.com
   
   # 2. Update WSL2
   wsl --update
   
   # 3. Restart WSL2
   wsl --shutdown
   # Then reopen Ubuntu


Getting Help
------------

If you're still stuck:

1. **Check the logs**: Look at error messages carefully
2. **Search issues**: https://github.com/yourusername/consciousness_detector/issues
3. **Ask for help**: Open a new issue with:
   
   * Your OS and Python version
   * Complete error message
   * Steps you've tried
   * Output of ``pip list`` and ``python --version``


Verifying Dataset Access
=========================

After installation, verify you can download test data:

.. code-block:: bash

   # Test dataset download (small 1MB test)
   python download_dataset.py --test
   
   # Expected output:
   # ✓ Connection to OpenNeuro successful
   # ✓ Dataset metadata accessible
   # ✓ Download functionality working

.. note::
   Full dataset download is covered in :doc:`quickstart`. This just tests connectivity.


Optional: Jupyter Notebook Setup
=================================

If you want to use Jupyter notebooks for exploration:

.. code-block:: bash

   # Install Jupyter
   pip install jupyter ipykernel
   
   # Add virtual environment to Jupyter
   python -m ipykernel install --user --name=consciousness --display-name="Consciousness Detector"
   
   # Start Jupyter
   jupyter notebook
   
   # Open browser and navigate to notebooks/ directory

You can now create notebooks using the "Consciousness Detector" kernel.


Next Steps
==========

Installation complete! Now you can:

1. **Download the dataset**: :doc:`quickstart` - Section "Dataset Download"
2. **Train your first model**: :doc:`quickstart` - Section "Running Models"
3. **Explore the code**: :doc:`architecture` for technical details

.. tip::
   We recommend starting with the :doc:`quickstart` tutorial, which walks you through a complete workflow from data download to model training.


Uninstallation
==============

To remove the project and free up space:

.. code-block:: bash

   # Deactivate virtual environment
   deactivate
   
   # Remove virtual environment
   rm -rf venv/
   
   # Remove project (careful!)
   cd ..
   rm -rf consciousness_detector/
   
   # If using conda
   conda env remove -n consciousness

.. warning::
   This will delete all downloaded data and trained models. Make sure to back up any important results first.


Updating
========

To update to the latest version:

.. code-block:: bash

   # Activate environment
   source venv/bin/activate
   
   # Pull latest changes
   git pull origin main
   
   # Update dependencies (in case requirements changed)
   pip install --upgrade -r requirements.txt
   
   # Verify
   python -c "import src; print('✓ Update successful')"

.. note::
   Always check the ``CHANGELOG.md`` for breaking changes before updating.
