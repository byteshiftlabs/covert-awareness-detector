#!/bin/bash
################################################################################
# Quick Reference: Manual Download Commands
################################################################################
#
# This file contains direct download commands for individual files.
# Copy and paste these commands to download specific files manually.
#
# Base URL: https://s3.amazonaws.com/openneuro.org/ds006623
################################################################################

# =============================================================================
# METADATA FILES (Essential)
# =============================================================================

# Participant demographics and info
wget https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/Participant_Info.csv

# Consciousness timing data (LOR/ROR timestamps)
wget https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/LOR_ROR_Timing.csv

# Dataset description
wget https://s3.amazonaws.com/openneuro.org/ds006623/dataset_description.json
wget https://s3.amazonaws.com/openneuro.org/ds006623/README.md


# =============================================================================
# CONNECTIVITY DATA - Example for one subject (sub-02)
# =============================================================================

BASE="https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/xcp_d_without_GSR_bandpass_output"

# Gordon Atlas (333 ROIs)
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Gordon_timeseries.tsv
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Gordon_connectome.tsv

# Schaefer Atlas (417 ROIs)
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Schaefer417_timeseries.tsv
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Schaefer417_connectome.tsv

# Glasser Atlas (360 ROIs)
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Glasser_timeseries.tsv
wget ${BASE}/sub-02/func/sub-02_task-rest_atlas-Glasser_connectome.tsv


# =============================================================================
# DOWNLOAD ALL SUBJECTS (Loop)
# =============================================================================

# Subjects in the dataset
SUBJECTS=(sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16 sub-17 sub-18 sub-19 sub-20 sub-21 sub-22 sub-23 sub-24 sub-25 sub-26 sub-27 sub-28 sub-29 sub-30)

# Download connectivity for all subjects
for subject in "${SUBJECTS[@]}"; do
    echo "Downloading ${subject}..."
    
    # Create directory
    mkdir -p derivatives/xcp_d_without_GSR_bandpass_output/${subject}/func/
    cd derivatives/xcp_d_without_GSR_bandpass_output/${subject}/func/
    
    # Gordon atlas
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Gordon_timeseries.tsv
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Gordon_connectome.tsv
    
    # Schaefer atlas
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Schaefer417_timeseries.tsv
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Schaefer417_connectome.tsv
    
    # Glasser atlas
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Glasser_timeseries.tsv
    wget ${BASE}/${subject}/func/${subject}_task-rest_atlas-Glasser_connectome.tsv
    
    cd ../../../../
done


# =============================================================================
# USING CURL INSTEAD OF WGET
# =============================================================================

# If you prefer curl, replace wget with curl -O:
# wget URL  -->  curl -O URL

# Example:
# curl -O https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/Participant_Info.csv


# =============================================================================
# DOWNLOAD WITH PROGRESS BAR (wget)
# =============================================================================

# Show progress bar
wget --progress=bar:force https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/Participant_Info.csv

# Quiet mode (no output)
wget -q https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/Participant_Info.csv

# Resume interrupted download
wget -c https://s3.amazonaws.com/openneuro.org/ds006623/derivatives/Participant_Info.csv


# =============================================================================
# VERIFY DOWNLOAD
# =============================================================================

# Check if file was downloaded successfully
if [ -f "derivatives/Participant_Info.csv" ]; then
    echo "✓ File exists"
    wc -l derivatives/Participant_Info.csv  # Should show 27 lines (header + 26 subjects)
else
    echo "✗ File not found"
fi


# =============================================================================
# ALTERNATIVE: Use AWS CLI (if installed)
# =============================================================================

# Install AWS CLI:
# pip install awscli

# Download entire directory
# aws s3 sync s3://openneuro.org/ds006623/derivatives/ ./derivatives/ --no-sign-request

# Download specific files
# aws s3 cp s3://openneuro.org/ds006623/derivatives/Participant_Info.csv ./derivatives/ --no-sign-request
