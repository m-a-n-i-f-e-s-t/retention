#!/bin/bash

set -e

# Default versions
TORCH_VERSION="2.6.0"
PYTHON_VERSION="3.11"
CLEANUP=false

# Parse command line arguments
while getopts "t:p:ch" opt; do
    case $opt in
        t) TORCH_VERSION="$OPTARG" ;;
        p) PYTHON_VERSION="$OPTARG" ;;
        c) CLEANUP=true ;;
        h) echo "Usage: $0 [-t torch_version] [-p python_version] [-c]"
           echo "  -t: Torch version (default: 2.6.0)"
           echo "  -p: Python version (default: 3.11)"
           echo "  -c: Clean up build environment after completion"
           exit 0 ;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    esac
done

echo "Building wheel with Python $PYTHON_VERSION and PyTorch $TORCH_VERSION"

# Create virtual environment
ENV_NAME="build-env-py${PYTHON_VERSION}-torch${TORCH_VERSION}"
echo "Creating virtual environment: $ENV_NAME"
uv venv --python "$PYTHON_VERSION" "build/$ENV_NAME"

# Cleanup function
cleanup() {
    if [[ "$CLEANUP" == true && -d "$ENV_NAME" ]]; then
        echo "Cleaning up environment: $ENV_NAME"
        rm -rf "$ENV_NAME"
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Activate environment and build
source "build/$ENV_NAME/bin/activate"

# Install torch with specified version
echo "Installing PyTorch $TORCH_VERSION..."
uv pip install "torch==$TORCH_VERSION" numpy psutil

# Install build dependencies
echo "Installing build dependencies..."
uv pip install setuptools>=69.5.1 wheel ninja

# Build the wheel
echo "Building wheel..."
python setup.py bdist_wheel

echo "✅ Wheel built successfully!"
echo "Environment: $ENV_NAME"
echo "Wheel location: dist/"

# Deactivate environment
deactivate
