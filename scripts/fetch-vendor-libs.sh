#!/bin/bash
# Fetch vendored GStreamer libraries for Linux
#
# This script downloads pre-built GStreamer libraries from GitHub releases.
# The libraries are built by CI on Ubuntu 24.04 and require glibc 2.39+.
#
# Usage:
#   ./scripts/fetch-vendor-libs.sh linux-x86_64
#
# LGPL Compliance:
#   GStreamer is LGPL-2.1+. Source is available at:
#   https://gstreamer.freedesktop.org/src/
#
# Ubuntu Compatibility:
#   - Ubuntu 24.04+: Supported
#   - Ubuntu 22.04: NOT supported (glibc too old), use Nix or Flatpak instead
#
# macOS Note:
#   macOS uses native AVFoundation for video. For MKV/WebM support via FFmpeg,
#   run ./scripts/setup-macos-ffmpeg.sh instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_ROOT/vendor"

# GStreamer bundle for Linux
GSTREAMER_VERSION="1.24.0-ubuntu24.04-1"
GSTREAMER_URL="https://github.com/lumina-video/lumina-video/releases/download/vendor-gstreamer-${GSTREAMER_VERSION}/gstreamer-vendor-linux-x86_64.tar.gz"

usage() {
    echo "Usage: $0 <platform>"
    echo ""
    echo "Platforms:"
    echo "  linux-x86_64    Download GStreamer for Linux x86_64 (Ubuntu 24.04+)"
    echo ""
    echo "Note: Ubuntu 22.04 is NOT supported due to glibc version requirements."
    echo "      Use Nix or Flatpak for Ubuntu 22.04."
    echo ""
    echo "macOS: Run ./scripts/setup-macos-ffmpeg.sh for MKV/WebM support."
    exit 1
}

check_glibc_version() {
    local glibc_version
    glibc_version=$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "0.0")

    local major minor
    major=$(echo "$glibc_version" | cut -d. -f1)
    minor=$(echo "$glibc_version" | cut -d. -f2)

    # Require glibc 2.39+
    if [[ "$major" -lt 2 ]] || [[ "$major" -eq 2 && "$minor" -lt 39 ]]; then
        echo "Error: glibc $glibc_version detected, but glibc 2.39+ is required."
        echo ""
        echo "The vendored GStreamer bundle is built on Ubuntu 24.04 (glibc 2.39)."
        echo ""
        echo "Options for your system:"
        echo "  1. Use Nix: nix develop github:lumina-video/lumina-video"
        echo "  2. Use Flatpak for distribution"
        echo "  3. Install system GStreamer packages (no zero-copy on GStreamer <1.24)"
        exit 1
    fi

    echo "glibc $glibc_version detected (OK)"
}

fetch_linux_x86_64() {
    local target_dir="$VENDOR_DIR/linux-x86_64"

    echo "Checking system compatibility..."
    check_glibc_version

    echo ""
    echo "Downloading GStreamer vendor bundle ($GSTREAMER_VERSION)..."
    echo "URL: $GSTREAMER_URL"
    echo ""

    # Create target directory
    mkdir -p "$target_dir"

    local tarball="$target_dir/gstreamer-vendor.tar.gz"

    # Download
    if command -v curl &>/dev/null; then
        curl -fSL --progress-bar "$GSTREAMER_URL" -o "$tarball"
    elif command -v wget &>/dev/null; then
        wget --show-progress "$GSTREAMER_URL" -O "$tarball"
    else
        echo "Error: curl or wget required"
        exit 1
    fi

    echo ""
    echo "Extracting..."
    tar -xzf "$tarball" -C "$target_dir"

    # Cleanup
    rm -f "$tarball"

    echo ""
    echo "Done! Vendored GStreamer installed to: $target_dir"
    echo ""

    if [ -f "$target_dir/VERSION" ]; then
        echo "Bundle info:"
        cat "$target_dir/VERSION"
    fi
}

# Main
if [[ $# -lt 1 ]]; then
    usage
fi

case "$1" in
    linux-x86_64)
        fetch_linux_x86_64
        ;;
    *)
        echo "Error: Unknown platform: $1"
        usage
        ;;
esac
