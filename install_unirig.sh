#!/bin/bash
# Installation script for UniRig in separate Python 3.11 environment

set -e  # Exit on error

echo "========================================="
echo "UniRig Installation Script"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Check if UniRig is cloned
if [ ! -d "/workspace/UniRig" ]; then
    echo "Cloning UniRig repository..."
    cd /workspace
    git clone https://github.com/VAST-AI-Research/UniRig.git
    echo "✓ UniRig cloned"
else
    echo "✓ UniRig already exists at /workspace/UniRig"
fi

# Create conda environment
echo ""
echo "Creating Python 3.11 conda environment 'unirig'..."
if conda env list | grep -q "^unirig "; then
    echo "Environment 'unirig' already exists. Skipping creation."
else
    conda create -n unirig python=3.11 -y
    echo "✓ Environment created"
fi

# Activate and install dependencies
echo ""
echo "Installing UniRig dependencies..."
echo "This may take 10-20 minutes..."
echo ""

# Use conda run to execute in unirig environment
cd /workspace/UniRig

# Install PyTorch
echo "Installing PyTorch (CUDA 11.8)..."
conda run -n unirig pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
echo "Installing base requirements..."
conda run -n unirig pip install -r requirements.txt

# Install spconv
echo "Installing spconv..."
conda run -n unirig pip install spconv-cu118

# Install PyG packages
echo "Installing PyG packages (torch_scatter, torch_cluster)..."
conda run -n unirig pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.1+cu118.html --no-cache-dir

# Downgrade numpy
echo "Fixing numpy version..."
conda run -n unirig pip install numpy==1.26.4

# Optional: flash_attn
echo ""
read -p "Install flash_attn? (requires compilation, may fail) [y/N]: " install_flash
if [[ $install_flash =~ ^[Yy]$ ]]; then
    echo "Installing flash_attn..."
    conda run -n unirig pip install flash-attn --no-build-isolation || echo "Warning: flash_attn installation failed (optional)"
fi

echo ""
echo "========================================="
echo "✓ Installation complete!"
echo "========================================="
echo ""
echo "UniRig is installed in conda environment 'unirig'"
echo "The ComfyUI-UniRig nodes will automatically use this environment"
echo ""
echo "To test manually:"
echo "  conda activate unirig"
echo "  cd /workspace/UniRig"
echo "  bash launch/inference/generate_skeleton.sh --input examples/giraffe.glb --output test.fbx"
echo ""
