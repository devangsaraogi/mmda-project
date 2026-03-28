#!/bin/bash
# Hazel Cluster Environment Setup for MMDA Project
# Run this once on the LOGIN NODE from the project root:
#   bash scripts/setup_env.sh

set -e

echo "========================================"
echo "MMDA Project — Environment Setup"
echo "========================================"

# Step 1: Load modules (with fallbacks)
echo -e "\n1. Loading modules..."
module load python/3.11 2>/dev/null || module load python/3.9.6 2>/dev/null || module load python 2>/dev/null || echo "Warning: Python module not found"
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: CUDA module not found"

echo "Loaded modules:"
module list

# Step 2: Create virtual environment
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo -e "\n2. Creating virtual environment in $PROJECT_ROOT/venv ..."
python3 -m venv venv || python -m venv venv

# Step 3: Activate
echo -e "\n3. Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
echo -e "\n4. Upgrading pip..."
pip install --upgrade pip

# Step 5: Install PyTorch with CUDA support
echo -e "\n5. Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 6: Install project dependencies
echo -e "\n6. Installing project dependencies..."
pip install -r requirements.txt
pip install -e .

# Step 7: Test installation + download NLTK data
echo -e "\n7. Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || echo "NLTK punkt_tab download skipped (will retry at runtime)"

# Step 8: Create output directories
echo -e "\n8. Creating directories..."
mkdir -p results/{logs,figures,metrics,embeddings,checkpoints,experiments} logs

# Step 9: Make LSF scripts executable
echo -e "\n9. Making scripts executable..."
chmod +x lsf/*.bsub 2>/dev/null

echo -e "\n========================================"
echo "Setup Complete!"
echo "========================================"
echo -e "\nActivate with:  source venv/bin/activate"
echo "Project root:   $PROJECT_ROOT"
echo -e "\nLSF Commands:"
echo "  Submit job:   cd lsf && bsub < embed_images.bsub"
echo "  Check jobs:   bjobs"
echo "  Cancel job:   bkill <JOB_ID>"
echo "  Monitor:      tail -f logs/*.out"
