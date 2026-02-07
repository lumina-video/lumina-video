# Vendored Runtime Libraries

This directory contains vendored runtime libraries for Linux.

## Structure

```text
vendor/
├── linux-x86_64/           # GStreamer for Linux
│   ├── lib/                # Core libraries
│   └── lib/gstreamer-1.0/  # Plugins
└── README.md
```

## What Gets Vendored

| Platform | Library | Feature | Purpose |
|----------|---------|---------|---------|
| Linux | GStreamer 1.24 | `vendored-runtime` | Video playback, zero-copy |

**macOS note**: macOS uses native AVFoundation for video. For MKV/WebM support via FFmpeg,
run `./scripts/setup-macos-ffmpeg.sh` (one-time setup).

## LGPL Compliance

GStreamer is licensed under LGPL-2.1+.

### Source Availability
- GStreamer: https://gstreamer.freedesktop.org/src/

### Relinking Instructions

Users can replace vendored libraries with system versions:

**Linux (GStreamer):**
```bash
# Install system GStreamer
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Build without vendored-runtime
cargo build  # (omit vendored-runtime feature)
```

## Downloading Vendored Libraries

Libraries are downloaded automatically during build. For manual download:

```bash
# Linux
./scripts/fetch-vendor-libs.sh linux-x86_64
```

## Runtime Deployment Layout

When using the `vendored-runtime` feature, the binary expects libraries in specific
locations relative to the executable. The build sets rpath to search these locations:

```text
your-app/                      # Executable directory
├── your-binary                # Main executable
├── vendor/
│   └── linux-x86_64/
│       └── lib/               # $ORIGIN/vendor/linux-x86_64/lib
│           ├── libgstreamer-1.0.so.0
│           ├── libgstapp-1.0.so.0
│           └── gstreamer-1.0/  # Plugins
└── lib/                       # Alternative: $ORIGIN/../lib
    └── (libraries)            # For installations like /usr/bin + /usr/lib
```

### Deployment Options

1. **Standalone bundle** (recommended for distribution):
   ```text
   my-app/
   ├── my-app                    # Binary
   └── vendor/linux-x86_64/lib/  # Copy from build's vendor/
   ```

2. **System-style installation** (e.g., deb/rpm packages):
   ```text
   /usr/
   ├── bin/my-app               # Binary
   └── lib/                     # Libraries ($ORIGIN/../lib)
   ```

3. **AppImage/Flatpak**: The runtime typically provides GStreamer 1.24+,
   so vendored libraries are usually not needed.

## Supported Platforms

| Platform | Status | Size | Notes |
|----------|--------|------|-------|
| Linux x86_64 (Ubuntu 24.04+) | Supported | ~150MB | glibc 2.39+ required |
| Linux x86_64 (Ubuntu 22.04) | Not supported | - | Use Nix, Flatpak, or PPA |
| Linux aarch64 | Planned | - | |

### Ubuntu 22.04 Compatibility

The GStreamer bundle requires glibc 2.39+ (Ubuntu 24.04).
For Ubuntu 22.04, use one of these alternatives:

- **PPA**: `sudo add-apt-repository ppa:savoury1/multimedia` (GStreamer 1.24)
- **Nix**: `nix develop github:lumina-video/lumina-video`
- **Flatpak**: Bundle your app as Flatpak (runtime includes GStreamer 1.24+)
