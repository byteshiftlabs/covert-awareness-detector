#!/bin/bash
#
# Complete Training Pipeline
# ==========================
# 1. Download dataset (if not already present)
# 2. Train advanced consciousness detection model
#
# Usage: ./run_full_training.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${PROJECT_DIR}/../datasets/openneuro/ds006623"
VENV_DIR="${PROJECT_DIR}/venv"

echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║${NC}   ${CYAN}Covert Consciousness Detection - Full Training Pipeline${NC}   ${PURPLE}║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check/Activate virtual environment
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[1/3] Checking Virtual Environment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo -e "${YELLOW}  Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${GREEN}✓ Activating virtual environment${NC}"
source "$VENV_DIR/bin/activate"

# Check if required packages are installed
echo -e "${YELLOW}  Checking dependencies...${NC}"
if python -c "import numpy, sklearn, xgboost, pandas, imblearn" 2>/dev/null; then
    echo -e "${GREEN}✓ All dependencies present${NC}"
else
    echo -e "${YELLOW}  Missing dependencies - installing into venv...${NC}"
    python -m pip install --no-cache-dir numpy scipy pandas scikit-learn xgboost imbalanced-learn
    
    # Verify installation succeeded
    if python -c "import numpy, sklearn, xgboost, pandas, imblearn" 2>/dev/null; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install or import dependencies${NC}"
        echo -e "${YELLOW}  Attempting to diagnose...${NC}"
        python -c "import numpy; print('✓ numpy')" 2>&1 || echo "✗ numpy failed"
        python -c "import sklearn; print('✓ sklearn')" 2>&1 || echo "✗ sklearn failed"
        python -c "import xgboost; print('✓ xgboost')" 2>&1 || echo "✗ xgboost failed"
        python -c "import pandas; print('✓ pandas')" 2>&1 || echo "✗ pandas failed"
        python -c "import imblearn; print('✓ imblearn')" 2>&1 || echo "✗ imblearn failed"
        exit 1
    fi
fi

echo ""

# Step 2: Download dataset
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[2/3] Dataset Preparation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if dataset exists
if [ -d "$DATASET_DIR/derivatives/xcp_d_without_GSR_bandpass_output" ]; then
    # Count number of subject directories
    SUBJECT_COUNT=$(ls -d "$DATASET_DIR/derivatives/xcp_d_without_GSR_bandpass_output"/sub-* 2>/dev/null | wc -l)
    
    if [ "$SUBJECT_COUNT" -ge 25 ]; then
        echo -e "${GREEN}✓ Dataset already present${NC}"
        echo -e "  Location: ${DATASET_DIR}"
        echo -e "  Subjects: ${SUBJECT_COUNT}"
        
        DATASET_SIZE=$(du -sh "$DATASET_DIR" 2>/dev/null | cut -f1)
        echo -e "  Size: ${DATASET_SIZE}"
    else
        echo -e "${YELLOW}⚠ Dataset incomplete (only ${SUBJECT_COUNT} subjects)${NC}"
        echo -e "${YELLOW}  Re-downloading...${NC}"
        python src/download_dataset.py --output-dir "$DATASET_DIR"
    fi
else
    echo -e "${YELLOW}⚠ Dataset not found${NC}"
    echo -e "${CYAN}  Downloading OpenNeuro ds006623...${NC}"
    echo -e "  ${YELLOW}Note: This will download ~1.8GB of preprocessed data${NC}"
    echo ""
    
    python src/download_dataset.py --output-dir "$DATASET_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Dataset downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Dataset download failed!${NC}"
        exit 1
    fi
fi

echo ""

# Step 3: Train model
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[3/3] Training Advanced Model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Model: XGBoost with PCA + SMOTE"
echo -e "  Validation: Leave-One-Subject-Out CV (25 folds)"
echo -e "  Subjects: All 25 subjects"
echo -e "  ${YELLOW}Estimated time: 10-15 minutes${NC}"
echo ""

# Save output to log file
LOG_FILE="${PROJECT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo -e "${CYAN}  Saving output to: ${LOG_FILE}${NC}"
echo ""

# Run training with tee to show output and save to log
# Note: stderr (progress bars) goes to terminal, stdout (results) goes to both
python src/train.py | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                  ${GREEN}✓ Training Complete!${NC}                         ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}📊 Results saved to: ${LOG_FILE}${NC}"
    
    # Check if results directory has JSON files
    LATEST_RESULT=$(ls -t results/results_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo -e "${CYAN}📈 Metrics saved to: ${LATEST_RESULT}${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  • Review the training log: ${GREEN}less ${LOG_FILE}${NC}"
    echo -e "  • Validate model: ${GREEN}python src/validate_model.py${NC}"
    echo -e "  • Check results: ${GREEN}ls -lh results/${NC}"
    
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}                  ${RED}✗ Training Failed${NC}                            ${RED}║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Check the log file for details: ${LOG_FILE}${NC}"
    exit 1
fi
