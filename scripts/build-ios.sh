#!/bin/bash
# Cross-compile lumina-video-ios for iOS targets and verify symbols + Swift linkage.
#
# Builds for:
#   - aarch64-apple-ios       (device)
#   - aarch64-apple-ios-sim   (simulator)
#
# Usage:
#   ./scripts/build-ios.sh [--debug]    (default is release)

set -euo pipefail

PROFILE="release"
CARGO_FLAG="--release"
if [[ "${1:-}" == "--debug" ]]; then
    PROFILE="debug"
    CARGO_FLAG=""
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT_DIR/include/LuminaVideo.h"
CRATE="lumina-video-ios"

# =========================================================================
# Platform check
# =========================================================================

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script must be run on macOS."
    exit 1
fi

echo "=== iOS Build Pipeline for lumina-video ==="
echo ""

# =========================================================================
# Prerequisites
# =========================================================================

echo "Checking prerequisites..."

# Xcode
if ! command -v xcrun &>/dev/null; then
    echo "ERROR: xcrun not found. Install Xcode Command Line Tools."
    exit 1
fi
XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -1 || echo "unknown")
echo "  Xcode: $XCODE_VERSION"

# Rust targets
for TARGET in aarch64-apple-ios aarch64-apple-ios-sim; do
    if ! rustup target list --installed | grep -q "$TARGET"; then
        echo "  Installing Rust target: $TARGET"
        rustup target add "$TARGET"
    else
        echo "  Rust target: $TARGET (installed)"
    fi
done

# Header file
if [[ ! -f "$HEADER" ]]; then
    echo "ERROR: Header not found at $HEADER"
    exit 1
fi
echo "  Header: $HEADER"
echo ""

# =========================================================================
# Expected symbols
# =========================================================================

EXPECTED_SYMBOLS=(
    "_lumina_player_create"
    "_lumina_player_destroy"
    "_lumina_player_play"
    "_lumina_player_pause"
    "_lumina_player_seek"
    "_lumina_player_state"
    "_lumina_player_position"
    "_lumina_player_duration"
    "_lumina_player_set_muted"
    "_lumina_player_is_muted"
    "_lumina_player_set_volume"
    "_lumina_player_volume"
    "_lumina_player_poll_frame"
    "_lumina_frame_width"
    "_lumina_frame_height"
    "_lumina_frame_iosurface"
    "_lumina_frame_release"
    "_lumina_diagnostics_snapshot"
    "_lumina_audio_smoke_test"
)

# =========================================================================
# Build function
# =========================================================================

build_target() {
    local TARGET="$1"
    local SDK="$2"
    local SDK_PATH
    SDK_PATH="$(xcrun --sdk "$SDK" --show-sdk-path)"

    echo "Building $CRATE for $TARGET (SDK: $SDK)..."
    SDKROOT="$SDK_PATH" cargo build -p "$CRATE" --target "$TARGET" $CARGO_FLAG 2>&1

    local LIB="$ROOT_DIR/target/$TARGET/$PROFILE/lib${CRATE//-/_}.a"
    if [[ ! -f "$LIB" ]]; then
        echo "ERROR: Static library not found at $LIB"
        exit 1
    fi
    echo "  Output: $LIB ($(du -h "$LIB" | cut -f1))"

    # Verify symbols
    # Note: nm may exit non-zero on some object files in the archive, so we
    # capture its output in a file to avoid pipefail issues.
    echo "  Verifying symbols..."
    local NM_OUT="$ROOT_DIR/target/.nm-symbols-$TARGET.txt"
    nm -gU "$LIB" > "$NM_OUT" 2>/dev/null || true
    local MISSING=0
    for SYM in "${EXPECTED_SYMBOLS[@]}"; do
        if ! grep -q " T ${SYM}$" "$NM_OUT"; then
            echo "    MISSING: $SYM"
            MISSING=1
        fi
    done
    if [[ "$MISSING" -eq 1 ]]; then
        rm -f "$NM_OUT"
        echo "ERROR: Missing symbols in $LIB"
        exit 1
    fi
    local SYM_COUNT
    SYM_COUNT=$(grep -c " T _lumina_" "$NM_OUT" || true)
    rm -f "$NM_OUT"
    echo "  Symbols OK ($SYM_COUNT lumina_* symbols found)"
    echo ""
}

# =========================================================================
# Build both targets
# =========================================================================

build_target "aarch64-apple-ios" "iphoneos"
build_target "aarch64-apple-ios-sim" "iphonesimulator"

# =========================================================================
# Swift link smoke test
# =========================================================================

echo "Running Swift link smoke tests..."

SWIFT_TEST_DIR="$ROOT_DIR/target/ios-swift-test"
mkdir -p "$SWIFT_TEST_DIR"

cat > "$SWIFT_TEST_DIR/LinkTest.swift" <<'SWIFT'
import Foundation
// lumina_player_state(nil) is safe per FFI contract — returns LUMINA_STATE_ERROR
let state = lumina_player_state(nil)
// lumina_frame_release(nil) is safe per FFI contract — no-op
lumina_frame_release(nil)
SWIFT

LINK_FRAMEWORKS="-framework AVFoundation -framework CoreMedia -framework CoreVideo -framework Metal -framework IOSurface -framework QuartzCore -framework Security -framework CoreFoundation -framework SystemConfiguration"

swift_link_test() {
    local TARGET_TRIPLE="$1"
    local SDK="$2"
    local RUST_TARGET="$3"
    local SDK_PATH
    SDK_PATH="$(xcrun --sdk "$SDK" --show-sdk-path)"
    local LIB_DIR="$ROOT_DIR/target/$RUST_TARGET/$PROFILE"

    echo "  Swift link test: $TARGET_TRIPLE"
    xcrun swiftc \
        -target "$TARGET_TRIPLE" \
        -sdk "$SDK_PATH" \
        -import-objc-header "$HEADER" \
        -L "$LIB_DIR" \
        -llumina_video_ios \
        $LINK_FRAMEWORKS \
        "$SWIFT_TEST_DIR/LinkTest.swift" \
        -o "$SWIFT_TEST_DIR/LinkTest-${SDK}" \
        2>&1
    echo "    PASS"
}

swift_link_test "arm64-apple-ios16.0" "iphoneos" "aarch64-apple-ios"
swift_link_test "arm64-apple-ios16.0-simulator" "iphonesimulator" "aarch64-apple-ios-sim"

echo ""

# =========================================================================
# Summary
# =========================================================================

echo "=== Build Summary ==="
echo ""

for TARGET in aarch64-apple-ios aarch64-apple-ios-sim; do
    local_lib="$ROOT_DIR/target/$TARGET/$PROFILE/lib${CRATE//-/_}.a"
    if [[ -f "$local_lib" ]]; then
        SIZE=$(du -h "$local_lib" | cut -f1)
        NM_TMP="$ROOT_DIR/target/.nm-summary-$TARGET.txt"
        nm -gU "$local_lib" > "$NM_TMP" 2>/dev/null || true
        SYM_COUNT=$(grep -c " T _lumina_" "$NM_TMP" || true)
        rm -f "$NM_TMP"
        echo "  $TARGET: $SIZE ($SYM_COUNT symbols)"
    fi
done

echo ""
echo "  Swift link (device):    ✓"
echo "  Swift link (simulator): ✓"
echo ""
echo "Static libraries are in target/<triple>/$PROFILE/liblumina_video_ios.a"
echo "Done."
