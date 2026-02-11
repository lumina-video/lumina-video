#!/bin/bash
# Generate Android launcher icon mipmaps from the source Ferris icon.
#
# Source: crates/lumina-video-demo/assets/lumina-video-icon-ferris-hybrid.png
# Output: android/app/src/main/res/mipmap-*/ic_launcher*.png
#         android/app/src/main/res/drawable/ic_launcher_foreground.png
#
# Requires: sips (macOS built-in)
#
# Usage:
#   ./scripts/generate-android-icons.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ICON="$ROOT_DIR/crates/lumina-video-demo/assets/lumina-video-icon-ferris-hybrid.png"
RES="$ROOT_DIR/android/app/src/main/res"

if [ ! -f "$ICON" ]; then
    echo "Error: Source icon not found at $ICON"
    exit 1
fi

echo "=== Generating Android launcher icons ==="
echo "Source: $ICON"
echo ""

# Standard mipmap sizes (48dp baseline)
# mdpi=48, hdpi=72, xhdpi=96, xxhdpi=144, xxxhdpi=192
for DENSITY_SIZE in "mdpi:48" "hdpi:72" "xhdpi:96" "xxhdpi:144" "xxxhdpi:192"; do
    DENSITY="${DENSITY_SIZE%%:*}"
    SIZE="${DENSITY_SIZE##*:}"
    mkdir -p "$RES/mipmap-$DENSITY"
    sips -z "$SIZE" "$SIZE" "$ICON" --out "$RES/mipmap-$DENSITY/ic_launcher.png" >/dev/null
    sips -z "$SIZE" "$SIZE" "$ICON" --out "$RES/mipmap-$DENSITY/ic_launcher_round.png" >/dev/null
    echo "  mipmap-$DENSITY: ${SIZE}x${SIZE}"
done

# Adaptive icon foreground (108dp * 4 = 432px for xxxhdpi)
mkdir -p "$RES/drawable"
sips -z 432 432 "$ICON" --out "$RES/drawable/ic_launcher_foreground.png" >/dev/null
echo "  drawable/ic_launcher_foreground: 432x432 (adaptive)"

echo ""
echo "Done. Adaptive icon XML files are in mipmap-anydpi-v26/."
echo "Background color is defined in values/ic_launcher_background.xml."
