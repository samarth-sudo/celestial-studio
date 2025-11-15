#!/bin/bash

##############################################################################
# Celestial Studio - Modal Local Test Script
##############################################################################

set -e  # Exit on error

echo "🧪 Celestial Studio - Modal Local Test"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Modal authentication
if ! modal profile list &> /dev/null; then
    echo "❌ Modal not authenticated. Please run: modal setup"
    exit 1
fi

PROFILE=$(modal profile list 2>/dev/null | grep "•" | awk '{print $3}')
print_success "Modal authenticated (workspace: $PROFILE)"
echo ""

# Run local test
echo "Running Modal functions locally..."
echo "This will test:"
echo "  - Isaac Lab simulation (3 seconds)"
echo "  - Isaac Lab training (5 iterations)"
echo ""

print_warning "This may take 5-10 minutes on first run (downloading containers)..."
echo ""

modal run modal_app.py

echo ""
print_success "Local test complete!"
echo ""
echo "If tests passed, deploy with:"
echo "  ./deploy.sh"
echo ""
echo "Or directly:"
echo "  modal deploy modal_app.py"
