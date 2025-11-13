#!/bin/bash

# Celestial Studio - Logo Setup Script

echo "🎨 Adding Celestial Studio logo..."

# Find the logo on Desktop
LOGO_PATH=$(find ~/Desktop -name "*.png" -type f 2>/dev/null | grep -i "screenshot.*11.37" | head -1)

if [ -z "$LOGO_PATH" ]; then
    echo "❌ Logo not found on Desktop"
    echo "📝 Please manually copy your logo to:"
    echo "   frontend/public/celestial-logo.png"
    exit 1
fi

# Copy to frontend
cp "$LOGO_PATH" frontend/public/celestial-logo.png

if [ $? -eq 0 ]; then
    echo "✅ Logo added successfully!"
    echo "🌐 Refresh http://localhost:5173 to see it"
else
    echo "❌ Failed to copy logo"
    echo "📝 Please manually copy:"
    echo "   From: $LOGO_PATH"
    echo "   To:   frontend/public/celestial-logo.png"
fi
