#!/bin/bash
# setup_env.sh
# Setup virtual environment and install dependencies
#
# PINNED PACKAGE VERSIONS (tested and working):
#   - PyTorch: 2.1.0+cu118
#   - PyTorch Geometric: 2.7.0
#   - NumPy: 1.26.4 (must be <2.0 for PyTorch 2.1.0)
#   - Pandas: 2.3.3
#   - scikit-learn: 1.7.2
#   - HDBSCAN: 0.8.40
#   - igraph: 0.11.9
#   - leidenalg: 0.10.2
#   - tqdm: 4.67.1
#
# System Requirements:
#   - Python 3.9 or 3.11
#   - CUDA 11.8 compatible GPU
#   - GLIBC 2.28+ (RHEL 8 / Ubuntu 18.04+)
#

set -e  # Exit on error

echo "================================================"
echo "Setting up Netlist Tomography environment"
echo "================================================"

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✓ Found Python 3.11"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo "✓ Found Python 3.9"
else
    echo "✗ Error: Python 3.9 or 3.11 not found"
    echo "  Please install Python 3.9 or 3.11"
    exit 1
fi

# Check if venv exists
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf $VENV_DIR
    else
        echo "Using existing venv..."
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 version)
# Pinned to 2.1.0 for compatibility with PyG extensions
echo "================================================"
echo "Installing PyTorch with CUDA support..."
echo "================================================"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "================================================"
echo "Installing PyTorch Geometric..."
echo "================================================"
pip install torch-geometric==2.7.0
pip install pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install data processing libraries
# CRITICAL: NumPy must be <2.0 for PyTorch 2.1.0 compatibility
echo "================================================"
echo "Installing data processing libraries..."
echo "================================================"
pip install "numpy<2.0" pandas==2.3.3 scikit-learn==1.7.2

# Install clustering libraries
echo "================================================"
echo "Installing clustering libraries..."
echo "================================================"
pip install hdbscan==0.8.40

# Install graph libraries (for Leiden) - OPTIONAL
echo "================================================"
echo "Installing graph libraries (optional)..."
echo "================================================"
echo "Note: Leiden clustering is optional. HDBSCAN is the default."
echo "Attempting to install igraph and leidenalg..."
pip install igraph==0.11.9 leidenalg==0.10.2 || echo "Warning: Leiden libraries failed to install. HDBSCAN will be used for clustering."

# Install utilities
echo "================================================"
echo "Installing utilities..."
echo "================================================"
pip install tqdm==4.67.1

# Verify installation
echo "================================================"
echo "Verifying installation..."
echo "================================================"

python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"

python -c "
import torch_geometric
print(f'✓ PyTorch Geometric {torch_geometric.__version__}')
"

python -c "
import numpy as np
import pandas as pd
import hdbscan
print(f'✓ NumPy {np.__version__}')
print(f'✓ Pandas {pd.__version__}')
print(f'✓ HDBSCAN {hdbscan.__version__}')

# Check optional Leiden libraries
try:
    import igraph as ig
    import leidenalg
    print(f'✓ igraph {ig.__version__}')
    print('✓ leidenalg (Leiden clustering available)')
except ImportError:
    print('⚠ Leiden libraries not available (using HDBSCAN only)')
"

echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python main_pipeline.py --help"
echo ""
