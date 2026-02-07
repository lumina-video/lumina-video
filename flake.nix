{
  description = "lumina-video - Hardware-accelerated video player for egui with zero-copy GPU rendering";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
          # Graphics
          libGL
          mesa  # VA-API support for AMD/Intel
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
            echo "lumina-video development shell"
            echo "GStreamer plugins: $GST_PLUGIN_PATH"
            echo ""
            echo "VA-API driver: auto-detected by libva"
            echo "  Override if needed: export LIBVA_DRIVER_NAME=radeonsi (AMD) or iHD (Intel)"
            echo ""
            echo "Build: cargo build --package lumina-video-demo"
            echo "Run:   cargo run --package lumina-video-demo"
          '';
        };

        # App for `nix run`
        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
        };
      }
    );
}
