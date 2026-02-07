//! Build script for lumina-video
//!
//! ## Linux (vendored-runtime feature)
//! Downloads pre-built GStreamer libraries from GitHub releases.
//!
//! ## macOS
//! Requires one-time setup: run `./scripts/setup-macos-ffmpeg.sh`
//! This installs FFmpeg and sets SDKROOT for bindgen.

// Allow unused imports when features aren't enabled
#[allow(unused_imports)]
use std::env;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::{Path, PathBuf};

/// Default URL for Linux GStreamer vendor bundle
#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
const DEFAULT_GSTREAMER_VENDOR_URL: &str =
    "https://github.com/lumina-video/lumina-video/releases/download/vendor-gstreamer-1.24.0-ubuntu24.04-1/gstreamer-vendor-linux-x86_64.tar.gz";

/// Get the vendor bundle URL, allowing override via environment variable
#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
fn get_vendor_url() -> String {
    env::var("EGUI_VID_VENDOR_BUNDLE_URL")
        .unwrap_or_else(|_| DEFAULT_GSTREAMER_VENDOR_URL.to_string())
}

/// Expected minimum size for GStreamer bundle (100MB)
#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
const GSTREAMER_MIN_SIZE: u64 = 100 * 1024 * 1024;

fn main() {
    // macOS: Check if SDKROOT is set (required for FFmpeg/bindgen)
    #[cfg(target_os = "macos")]
    {
        if env::var("SDKROOT").is_err() {
            println!("cargo:warning=SDKROOT not set. Build will fail with 'errno.h not found'.");
            println!("cargo:warning=Run: ./scripts/setup-macos-ffmpeg.sh");
            println!(
                "cargo:warning=Or manually: export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)"
            );
        }
    }

    // Linux: Setup vendored GStreamer
    #[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
    {
        if let Err(e) = setup_vendored_gstreamer() {
            // Print warning but don't fail - let pkg-config try system libs
            println!("cargo:warning=vendored-runtime: {}", e);
            println!(
                "cargo:warning=Falling back to system GStreamer (install libgstreamer1.0-dev)"
            );
        }
    }

    // Re-run if feature flags or environment variables change
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VENDORED_RUNTIME");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=EGUI_VID_VENDOR_BUNDLE_URL");
}

#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
fn setup_vendored_gstreamer() -> Result<(), String> {
    let out_dir = env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {}", e))?;
    let vendor_dir = PathBuf::from(&out_dir).join("gstreamer-vendor");

    // Check if already extracted
    if vendor_dir.join("VERSION").exists() {
        println!(
            "cargo:warning=Using cached vendored GStreamer from {}",
            vendor_dir.display()
        );
    } else {
        download_and_extract(&vendor_dir)?;
    }

    // Patch pkg-config files with actual paths
    patch_pkgconfig_files(&vendor_dir)?;

    // Set environment for pkg-config to find our vendored libs
    let pkgconfig_dir = vendor_dir.join("lib").join("pkgconfig");
    println!(
        "cargo:rustc-env=PKG_CONFIG_PATH={}",
        pkgconfig_dir.display()
    );

    // Also set for the current build process
    env::set_var("PKG_CONFIG_PATH", &pkgconfig_dir);

    // Add library search path for linking
    let lib_dir = vendor_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Set rpath so the binary finds libs at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/vendor/linux-x86_64/lib");

    println!(
        "cargo:warning=vendored-runtime: Using GStreamer from {}",
        vendor_dir.display()
    );

    Ok(())
}

#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
fn download_and_extract(vendor_dir: &Path) -> Result<(), String> {
    use std::process::Command;

    let vendor_url = get_vendor_url();

    println!("cargo:warning=Downloading GStreamer vendor bundle...");

    // Create vendor directory
    fs::create_dir_all(vendor_dir).map_err(|e| format!("Failed to create vendor dir: {}", e))?;

    let tarball_path = vendor_dir.join("gstreamer-vendor.tar.gz");

    // Try curl first, then wget
    let download_result = Command::new("curl")
        .args([
            "-fSL",
            "--progress-bar",
            &vendor_url,
            "-o",
            tarball_path.to_str().unwrap(),
        ])
        .status();

    let success = match download_result {
        Ok(status) if status.success() => true,
        _ => {
            // Try wget as fallback
            Command::new("wget")
                .args([
                    "-q",
                    "--show-progress",
                    &vendor_url,
                    "-O",
                    tarball_path.to_str().unwrap(),
                ])
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }
    };

    if !success {
        return Err(format!(
            "Failed to download vendor bundle from {}. \
            Install curl or wget, or disable vendored-runtime feature.",
            vendor_url
        ));
    }

    // Validate download size
    let metadata =
        fs::metadata(&tarball_path).map_err(|e| format!("Failed to read tarball: {}", e))?;

    if metadata.len() < GSTREAMER_MIN_SIZE {
        return Err(format!(
            "Downloaded file too small ({}MB), expected >100MB. Download may have failed.",
            metadata.len() / 1024 / 1024
        ));
    }

    println!("cargo:warning=Extracting GStreamer vendor bundle...");

    // Extract tarball
    let extract_result = Command::new("tar")
        .args([
            "-xzf",
            tarball_path.to_str().unwrap(),
            "-C",
            vendor_dir.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("Failed to run tar: {}", e))?;

    if !extract_result.success() {
        return Err("Failed to extract vendor bundle".to_string());
    }

    // Remove tarball to save space
    let _ = fs::remove_file(&tarball_path);

    // Verify extraction
    if !vendor_dir.join("VERSION").exists() {
        return Err("Extraction failed - VERSION file not found".to_string());
    }

    println!("cargo:warning=GStreamer vendor bundle ready");

    Ok(())
}

#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
fn patch_pkgconfig_files(vendor_dir: &Path) -> Result<(), String> {
    let pkgconfig_dir = vendor_dir.join("lib").join("pkgconfig");

    if !pkgconfig_dir.exists() {
        return Err("pkg-config directory not found in vendor bundle".to_string());
    }

    let prefix = vendor_dir.to_str().unwrap();

    for entry in
        fs::read_dir(&pkgconfig_dir).map_err(|e| format!("Failed to read pkgconfig dir: {}", e))?
    {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();

        if path.extension().map(|e| e == "pc").unwrap_or(false) {
            let content = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let patched = content.replace("@PREFIX@", prefix);

            fs::write(&path, patched)
                .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
        }
    }

    Ok(())
}
