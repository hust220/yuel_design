#!/bin/bash

# YuelDesign Setup Script
# This script clones the yuel_design repository and installs all required dependencies

set -e  # Exit on any error

echo "=== YuelDesign Setup Script ==="
echo "This script will clone the yuel_design repository and install all dependencies."
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a Python package is installed
python_package_exists() {
    python -c "import $1" >/dev/null 2>&1
}

# Function to install Python package
install_python_package() {
    local package=$1
    local install_name=${2:-$1}
    
    echo "Installing $install_name..."
    if $PIP_CMD install "$package"; then
        echo "✓ Successfully installed $install_name"
    else
        echo "✗ Failed to install $install_name"
        return 1
    fi
}

# Check if git is available
if ! command_exists git; then
    echo "Error: Git is not installed. Please install git first."
    echo "On Ubuntu/Debian: sudo apt-get install git"
    echo "On CentOS/RHEL: sudo yum install git"
    echo "On macOS: brew install git"
    exit 1
fi

# Clone the repository
echo "Step 1: Cloning yuel_design repository..."
if [ -d "yuel_design" ]; then
    echo "Directory 'yuel_design' already exists. Removing it..."
    rm -rf yuel_design
fi

if git clone https://bitbucket.org/dokhlab/yuel_design.git; then
    echo "✓ Successfully cloned yuel_design repository"
else
    echo "✗ Failed to clone yuel_design repository"
    exit 1
fi

# Change to the yuel_design directory
cd yuel_design
echo "Changed to yuel_design directory"

# Check if Python is installed
echo ""
echo "Step 2: Checking Python installation..."
if ! command_exists python; then
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3.7 or higher:"
    echo "  - Download from https://www.python.org/downloads/"
    echo "  - Or use your system package manager:"
    echo "    Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "    CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "    macOS: brew install python3"
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $python_version is installed"

# Check if pip is available and set PIP_CMD accordingly
if command_exists pip; then
    PIP_CMD="pip"
elif command_exists pip3; then
    PIP_CMD="pip3"
elif python -m pip --version >/dev/null 2>&1; then
    PIP_CMD="python -m pip"
elif python3 -m pip --version >/dev/null 2>&1; then
    PIP_CMD="python3 -m pip"
else
    echo "Error: pip is not installed. Please install pip:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "✓ pip is available (${PIP_CMD})"

# Upgrade pip to latest version
echo ""
echo "Step 3: Upgrading pip..."
$PIP_CMD install --upgrade pip

# Check PyTorch and PyTorch Lightning installation
echo ""
echo "Step 4: Checking PyTorch and PyTorch Lightning installation..."

# Check if PyTorch is installed
if python_package_exists torch; then
    echo "✓ PyTorch is already installed"
else
    echo "✗ PyTorch is not installed"
    echo "Please install PyTorch manually from: https://pytorch.org/get-started/locally/"
    echo "Choose your OS, package manager, and compute platform to get the correct installation command."
    echo "After installing PyTorch, run this script again."
    exit 1
fi

# Check if PyTorch Lightning is installed
if python_package_exists lightning; then
    echo "✓ PyTorch Lightning is already installed"
else
    echo "Installing PyTorch Lightning..."
    install_python_package lightning
fi

# Install other required packages
echo ""
echo "Step 5: Installing other required packages..."

# List of packages to install
packages=(
    "rdkit"
    "biopython"
    "tqdm"
    "pdb-tools"
    "imageio"
    "networkx"
    "scipy"
    "scikit-learn"
    "wandb"
    "pyyaml"
)

# Install each package
for package in "${packages[@]}"; do
    # Handle special cases for package names
    case $package in
        "rdkit")
            if python_package_exists rdkit; then
                echo "✓ RDKit is already installed"
            else
                install_python_package "rdkit" "RDKit"
            fi
            ;;
        "lightning")
            # Already handled above
            ;;
        *)
            # Extract package name for import check (remove version specifiers)
            import_name=$(echo "$package" | cut -d'=' -f1 | cut -d'<' -f1 | cut -d'>' -f1)
            if python_package_exists "$import_name"; then
                echo "✓ $import_name is already installed"
            else
                install_python_package "$package"
            fi
            ;;
    esac
done

# Verify all installations
echo ""
echo "Step 6: Verifying installations..."

# Test imports
python -c "
import sys
packages_to_test = [
    'torch', 'lightning', 'rdkit', 'Bio', 'tqdm', 
    'pdb_tools', 'imageio', 'networkx', 'scipy', 'sklearn', 'wandb', 'yaml'
]

failed_imports = []
for package in packages_to_test:
    try:
        __import__(package)
        print(f'✓ {package} imported successfully')
    except ImportError as e:
        print(f'✗ Failed to import {package}: {e}')
        failed_imports.append(package)

if failed_imports:
    print(f'\\nWarning: Failed to import: {failed_imports}')
    sys.exit(1)
else:
    print('\\n✓ All packages imported successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Setup Complete! ==="
    echo "✓ yuel_design repository cloned successfully"
    echo "✓ All dependencies installed and verified"
    echo ""
    echo "Next steps:"
    echo "1. Download the models:"
    echo "   mkdir -p models"
    echo "   wget https://zenodo.org/records/15467850/files/moad.ckpt?download=1 -O models/moad.ckpt"
    echo ""
    echo "2. For generation, use:"
    echo "   python yuel_design.py --pocket 2req_pocket.pdb --model models/moad.ckpt --size 15"
    echo ""
    echo "3. For training, download datasets and use:"
    echo "   python train_yuel_design.py --config configs/train_moad.yml"
    echo ""
    echo "For more information, see the README.md file."
else
    echo ""
    echo "✗ Setup completed with warnings. Some packages may not be properly installed."
    echo "Please check the error messages above and install missing packages manually."
    exit 1
fi 