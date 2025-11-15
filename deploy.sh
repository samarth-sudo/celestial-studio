#!/bin/bash

##############################################################################
# Celestial Studio - Modal Deployment Script
##############################################################################

set -e  # Exit on error

echo "🚀 Celestial Studio - Modal Deployment"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check Prerequisites
echo "Step 1: Checking Prerequisites"
echo "========================================"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION installed"
else
    print_error "Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Check Modal CLI
if command -v modal &> /dev/null; then
    MODAL_VERSION=$(modal --version 2>&1 | head -n1)
    print_success "Modal CLI installed: $MODAL_VERSION"
else
    print_warning "Modal CLI not found. Installing..."
    pip install modal
    print_success "Modal CLI installed"
fi

# Check Modal authentication
echo ""
echo "Checking Modal authentication..."
if modal profile list &> /dev/null; then
    PROFILE=$(modal profile list 2>/dev/null | grep "•" | awk '{print $3}')
    print_success "Modal authenticated (workspace: $PROFILE)"
else
    print_error "Modal not authenticated. Please run: modal setup"
    exit 1
fi

# Check secrets
echo ""
echo "Checking Modal secrets..."
if modal secret list 2>&1 | grep -q "nvidia-eula"; then
    print_success "Secret 'nvidia-eula' found"
else
    print_warning "Secret 'nvidia-eula' not found. Creating..."
    modal secret create nvidia-eula ACCEPT_EULA=Y
    print_success "Secret 'nvidia-eula' created"
fi

if modal secret list 2>&1 | grep -q "isaac-config"; then
    print_success "Secret 'isaac-config' found"
else
    print_warning "Secret 'isaac-config' not found. Creating..."
    modal secret create isaac-config \
        ISAAC_LAB_VERSION=v1.0.0 \
        DEFAULT_GPU=A10G
    print_success "Secret 'isaac-config' created"
fi

# Step 2: Verify Project Structure
echo ""
echo "Step 2: Verifying Project Structure"
echo "========================================"

if [ ! -f "modal_app.py" ]; then
    print_error "modal_app.py not found in current directory"
    exit 1
fi
print_success "modal_app.py found"

if [ ! -d "backend" ]; then
    print_error "backend directory not found"
    exit 1
fi
print_success "backend directory found"

# Step 3: Check Volumes
echo ""
echo "Step 3: Checking Modal Volumes"
echo "========================================"

if modal volume list 2>&1 | grep -q "celestial-isaac-models"; then
    print_success "Volume 'celestial-isaac-models' exists"
else
    print_warning "Volume 'celestial-isaac-models' not found. Creating..."
    modal volume create celestial-isaac-models
    print_success "Volume 'celestial-isaac-models' created"
fi

if modal volume list 2>&1 | grep -q "celestial-exports"; then
    print_success "Volume 'celestial-exports' exists"
else
    print_warning "Volume 'celestial-exports' not found. Creating..."
    modal volume create celestial-exports
    print_success "Volume 'celestial-exports' created"
fi

# Step 4: Deploy to Modal
echo ""
echo "Step 4: Deploying to Modal"
echo "========================================"
echo ""
echo "This will deploy:"
echo "  - FastAPI web endpoint"
echo "  - Isaac Lab GPU simulation function"
echo "  - Isaac Lab GPU training function"
echo ""
echo "First deployment will take ~15-20 minutes to build Isaac Sim container."
echo "Subsequent deploys are instant (cached)."
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_warning "Starting deployment..."
    echo ""

    modal deploy modal_app.py

    echo ""
    print_success "Deployment complete!"
    echo ""
    echo "Next steps:"
    echo "  1. View your app: https://modal.com/apps"
    echo "  2. API docs: https://[username]--celestial-studio-web.modal.run/docs"
    echo "  3. Test health: curl https://[username]--celestial-studio-web.modal.run/health"
    echo ""
else
    print_warning "Deployment cancelled"
    exit 0
fi
