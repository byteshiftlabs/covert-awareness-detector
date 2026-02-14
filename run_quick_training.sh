#!/bin/bash
#
# Quick Training Pipeline
# ========================
# Same as full training but uses only 5 subjects for fast testing
#
# Usage: ./run_quick_training.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} Quick Test Training (5 subjects only)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo ""

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

# Check dataset
DATASET_DIR="${PROJECT_DIR}/../datasets/openneuro/ds006623"
if [ ! -d "$DATASET_DIR/derivatives/xcp_d_without_GSR_bandpass_output" ]; then
    echo -e "${YELLOW}Downloading dataset first...${NC}"
    python src/download_dataset.py --output-dir "$DATASET_DIR"
fi

echo -e "${CYAN}Running training on first 5 subjects...${NC}"
echo -e "${YELLOW}Estimated time: 2-3 minutes${NC}"
echo ""

# Run quick training (5 subjects only)
python src/train.py 2>&1 | tee "quick_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo -e "${GREEN}✓ Quick training complete!${NC}"
echo -e "${CYAN}For full training (25 subjects): ${NC}${GREEN}./run_full_training.sh${NC}"
