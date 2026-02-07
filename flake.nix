{
  description = "lumina-video - Hardware-accelerated video player for egui with zero-copy GPU rendering";

  inputs = {
    # NixOS 24.11 stable - has GStreamer 1.24 for zero-copy DMABuf support
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # GStreamer packages for zero-copy video decoding
        # VA-API support is critical for hardware acceleration
        gstDeps = with pkgs.gst_all_1; [
          gstreamer
          gst-plugins-base
          gst-plugins-good
          gst-plugins-bad
          gst-plugins-ugly
          gst-libav
          gst-vaapi  # VA-API for zero-copy hardware decoding
        ];

        # Build-time only dependencies
        buildDeps = with pkgs; [
          pkg-config
          cmake
          vulkan-headers  # Build-time only (not needed at runtime)
          wayland-protocols  # Build-time only
        ];

        # Runtime dependencies (required for zero-copy + playback)
        runtimeDeps = with pkgs; [
          # Vulkan for zero-copy rendering
          vulkan-loader
          # Graphics - libglvnd provides EGL/GL dispatch (required by wgpu GLES backend)
          libGL
          libglvnd
          mesa  # VA-API support for AMD/Intel
          egl-wayland  # EGL platform for Wayland
          # Windowing
          wayland
          libxkbcommon
          xorg.libX11
          xorg.libXcursor
          xorg.libXrandr
          xorg.libXi
          # VA-API drivers for hardware decoding
          intel-media-driver  # Intel iHD driver
          libva  # VA-API runtime
        ] ++ gstDeps;

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

      in
      {
        packages = {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "lumina-video-demo";
            version = "0.1.0";

            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
              # Git dependencies require explicit hashes for Nix reproducibility.
              # To update these hashes after changing Cargo.lock:
              #   1. Set hash to empty string: "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
              #   2. Run: nix build .#default
              #   3. Nix will error with the correct hash - copy it here
              outputHashes = {
                "android-activity-0.6.0" = "sha256-GhaSPprANT1A9lJYnYmkKeDsHHWAEqCpOAZi/sHi5Wc=";
                "wgpu-24.0.0" = "sha256-p6T3Jw4k8v5VViZLEatFNVeMYjzaBdlU5WTHNM1wv8E=";
              };
            };

            nativeBuildInputs = buildDeps ++ [ pkgs.makeWrapper ];
            buildInputs = runtimeDeps;

            # Build only the demo package
            cargoBuildFlags = [ "--package" "lumina-video-demo" ];

            # Set up GStreamer plugin path for zero-copy
            # Note: libva auto-detects the VA-API driver. No need to set LIBVA_DRIVER_NAME.
            postInstall = ''
              wrapProgram $out/bin/lumina-video-demo \
                --prefix GST_PLUGIN_PATH : "${pkgs.lib.makeSearchPath "lib/gstreamer-1.0" gstDeps}" \
                --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath runtimeDeps}"
            '';

            meta = with pkgs.lib; {
              description = "Demo application for lumina-video video player with zero-copy GPU rendering";
              homepage = "https://github.com/lumina-video/lumina-video";
              license = with licenses; [ mit asl20 ];
              platforms = platforms.linux;
            };
          };
        };

        # Development shell with all dependencies
        devShells.default = pkgs.mkShell {
          buildInputs = buildDeps ++ runtimeDeps ++ [
            rustToolchain
            pkgs.rust-analyzer
            # Diagnostic tools
            pkgs.libva-utils   # vainfo
            pkgs.vulkan-tools  # vulkaninfo
          ];

          # Environment variables for GStreamer zero-copy
          GST_PLUGIN_PATH = pkgs.lib.makeSearchPath "lib/gstreamer-1.0" gstDeps;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeDeps;

          # Note: libva auto-detects the VA-API driver based on your GPU.
          # Override only if auto-detection fails:
          #   export LIBVA_DRIVER_NAME=iHD       # Intel (newer)
          #   export LIBVA_DRIVER_NAME=i965      # Intel (older)
          #   export LIBVA_DRIVER_NAME=radeonsi  # AMD

          shellHook = ''
            echo "══════════════════════════════════════════════════════════════"
            echo "  lumina-video development shell"
            echo "══════════════════════════════════════════════════════════════"
            echo ""
            echo "Build: cargo build --package lumina-video-demo"
            echo "Run:   cargo run --package lumina-video-demo"
            echo ""
            echo "── Hardware Acceleration ──────────────────────────────────────"
            echo ""
            echo "For zero-copy VA-API on NixOS, add to configuration.nix:"
            echo ""
            echo "NixOS 24.11+ (hardware.graphics):"
            echo ""
            echo "  hardware.graphics = {"
            echo "    enable = true;"
            echo "    extraPackages = with pkgs; ["
            echo "      intel-media-driver  # Intel Broadwell+ (iHD)"
            echo "      # vaapiIntel        # Intel pre-Broadwell (i965)"
            echo "    ];"
            echo "  };"
            echo ""
            echo "NixOS 24.05 and earlier (hardware.opengl):"
            echo ""
            echo "  hardware.opengl = {"
            echo "    enable = true;"
            echo "    extraPackages = with pkgs; ["
            echo "      intel-media-driver  # Intel Broadwell+ (iHD)"
            echo "      # vaapiIntel        # Intel pre-Broadwell (i965)"
            echo "    ];"
            echo "  };"
            echo ""
            echo "AMD GPUs: VA-API is provided by Mesa (radeonsi), included in this"
            echo "dev shell. No extraPackages needed. (amdvlk is Vulkan-only, not VA-API.)"
            echo ""
            echo "Check your NixOS release manual before applying these snippets."
            echo ""
            echo "Diagnostics:"
            echo "  vainfo                  # Check VA-API driver"
            echo "  vulkaninfo | grep dma   # Check Vulkan DMA-BUF support"
            echo "  ls /dev/dri/            # Check DRM devices"
            echo ""
            echo "If VA-API shows 'libva error', your system needs hardware.graphics"
            echo "(or hardware.opengl on older releases) in configuration.nix."
            echo "══════════════════════════════════════════════════════════════"
          '';
        };

        # App for `nix run`
        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
        };
      }
    );
}
