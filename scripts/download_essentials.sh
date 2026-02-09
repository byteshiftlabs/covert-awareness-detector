#!/bin/bash
################################################################################
# Download Essential Files for Consciousness Detector Project
################################################################################
#
# This script downloads only the essential files needed for consciousness 
# detection research from the OpenNeuro ds006623 dataset.
#
# Dataset: Michigan Human Anesthesia fMRI Dataset
# URL: https://openneuro.org/datasets/ds006623
# DOI: 10.18112/openneuro.ds006623.v1.0.0
#
# What it downloads:
#   - Participant metadata (demographics, info)
#   - Consciousness timing data (LOR/ROR timestamps)
#   - Preprocessed connectivity matrices (Gordon, Schaefer, Glasser atlases)
#
# Usage:
#   ./download_essentials.sh [output_directory]
#
# Examples:
#   ./download_essentials.sh                           # Use default location
#   ./download_essentials.sh /path/to/dataset          # Custom location
#
# Requirements:
#   - curl or wget
#   - Basic shell utilities (mkdir, grep, etc.)
#
# Author: Christian
# Date: February 2026
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="https://s3.amazonaws.com/openneuro.org/ds006623"
DEFAULT_OUTPUT_DIR="../datasets/openneuro/ds006623"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

# Dataset subjects
SUBJECTS=(
    sub-02 sub-03 sub-04 sub-05 sub-06 sub-07
    sub-11 sub-12 sub-13 sub-14 sub-15 sub-16
    sub-17 sub-18 sub-19 sub-20 sub-21 sub-22
    sub-23 sub-24 sub-25 sub-26 sub-27 sub-28
    sub-29 sub-30
)

# Statistics
TOTAL_FILES=0
DOWNLOADED_FILES=0
SKIPPED_FILES=0
FAILED_FILES=0

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Detect available download tool
detect_download_tool() {
    if command -v curl &> /dev/null; then
        echo "curl"
    elif command -v wget &> /dev/null; then
        echo "wget"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
}

# Download a single file
download_file() {
    local remote_path="$1"
    local local_path="$2"
    local url="${BASE_URL}/${remote_path}"
    
    ((TOTAL_FILES++))
    
    # Create parent directory
    mkdir -p "$(dirname "$local_path")"
    
    # Check if file exists
    if [ -f "$local_path" ]; then
        print_success "Already exists: $remote_path"
        ((SKIPPED_FILES++))
        return 0
    fi
    
    # Download based on available tool
    if [ "$DOWNLOAD_TOOL" = "curl" ]; then
        if curl -f -L -o "$local_path" "$url" 2>/dev/null; then
            print_success "Downloaded: $remote_path"
            ((DOWNLOADED_FILES++))
            return 0
        else
            print_error "Failed: $remote_path"
            ((FAILED_FILES++))
            rm -f "$local_path"  # Remove partial file
            return 1
        fi
    else
        if wget -q -O "$local_path" "$url" 2>/dev/null; then
            print_success "Downloaded: $remote_path"
            ((DOWNLOADED_FILES++))
            return 0
        else
            print_error "Failed: $remote_path"
            ((FAILED_FILES++))
            rm -f "$local_path"  # Remove partial file
            return 1
        fi
    fi
}

################################################################################
# Download Functions
################################################################################

download_metadata() {
    print_header "STEP 1: Downloading Essential Metadata Files"
    
    local files=(
        "dataset_description.json"
        "README.md"
        "CHANGES"
        "derivatives/Participant_Info.csv"
        "derivatives/LOR_ROR_Timing.csv"
    )
    
    for file in "${files[@]}"; do
        download_file "$file" "${OUTPUT_DIR}/${file}"
    done
}

download_connectivity_subject() {
    local subject="$1"
    local base_path="derivatives/xcp_d_without_GSR_bandpass_output"
    
    echo ""
    print_info "Processing $subject..."
    
    # Connectivity files for each atlas
    local files=(
        "func/${subject}_task-rest_atlas-Gordon_timeseries.tsv"
        "func/${subject}_task-rest_atlas-Gordon_connectome.tsv"
        "func/${subject}_task-rest_atlas-Schaefer417_timeseries.tsv"
        "func/${subject}_task-rest_atlas-Schaefer417_connectome.tsv"
        "func/${subject}_task-rest_atlas-Glasser_timeseries.tsv"
        "func/${subject}_task-rest_atlas-Glasser_connectome.tsv"
    )
    
    for file in "${files[@]}"; do
        local remote_path="${base_path}/${subject}/${file}"
        local local_path="${OUTPUT_DIR}/${remote_path}"
        download_file "$remote_path" "$local_path"
    done
}

download_connectivity_data() {
    print_header "STEP 2: Downloading Preprocessed Connectivity Data"
    echo "Directory: derivatives/xcp_d_without_GSR_bandpass_output/"
    
    for subject in "${SUBJECTS[@]}"; do
        download_connectivity_subject "$subject"
    done
}

print_summary() {
    local duration=$SECONDS
    
    echo ""
    print_header "Download Summary"
    
    echo "Total files processed:       $TOTAL_FILES"
    echo "Successfully downloaded:     $DOWNLOADED_FILES"
    echo "Already existed (skipped):   $SKIPPED_FILES"
    echo "Failed:                      $FAILED_FILES"
    echo "Time elapsed:                ${duration}s"
    
    echo -e "${BLUE}======================================================================${NC}"
    
    if [ $FAILED_FILES -gt 0 ]; then
        echo ""
        print_warning "Some files failed to download. Check errors above."
        return 1
    else
        echo ""
        print_success "All files downloaded successfully!"
        echo ""
        print_info "Dataset location: $(cd "$OUTPUT_DIR" && pwd)"
        return 0
    fi
}

################################################################################
# Main Script
################################################################################

main() {
    # Start timer
    SECONDS=0
    
    # Print header
    print_header "OpenNeuro ds006623 Dataset Downloader"
    echo "Michigan Human Anesthesia fMRI Dataset"
    echo ""
    echo "Output directory: $(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
    
    # Detect download tool
    DOWNLOAD_TOOL=$(detect_download_tool)
    echo "Download tool: $DOWNLOAD_TOOL"
    echo -e "${BLUE}======================================================================${NC}"
    
    # Download files
    download_metadata
    download_connectivity_data
    
    # Print summary
    print_summary
}

# Run main function
main
