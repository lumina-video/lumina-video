#!/bin/bash
# Setup script for building lumina-video with FFmpeg support on macOS
#
# This script:
# 1. Installs FFmpeg via Homebrew (if not present)
# 2. Detects SDKROOT for bindgen (fixes errno.h not found)
# 3. Prints instructions - does NOT modify your dotfiles
#
# Usage:
#   ./scripts/setup-macos-ffmpeg.sh

set -euo pipefail

echo "=== macOS FFmpeg Setup for lumina-video ==="
echo ""

# Check if on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# Install FFmpeg if not present
if ! brew list ffmpeg &>/dev/null; then
    echo "Installing FFmpeg via Homebrew..."
    brew install ffmpeg
else
    echo "✓ FFmpeg already installed"
fi

# Get SDKROOT
SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
echo "✓ Found macOS SDK: $SDKROOT"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To build lumina-video, set SDKROOT before running cargo:"
echo ""
echo "  export SDKROOT=\$(xcrun --sdk macosx --show-sdk-path)"
echo "  cargo build"
echo ""
echo "Or as a one-liner:"
echo ""
echo "  SDKROOT=\$(xcrun --sdk macosx --show-sdk-path) cargo build"
echo ""
echo "── Optional: Persist SDKROOT ──────────────────────────────────"
echo ""
echo "If you want SDKROOT set automatically in new terminals, add this"
echo "line to your shell config (~/.zshrc or ~/.bash_profile):"
echo ""
echo "  export SDKROOT=\$(xcrun --sdk macosx --show-sdk-path)"
echo ""
echo "Or use direnv with a .envrc file in this project."
echo ""
