//! Vendored runtime library initialization for Linux.
//!
//! When the `vendored-runtime` feature is enabled, this module sets up the environment
//! to load GStreamer libraries from the `vendor/` directory bundled with the executable.
//!
//! # How it works
//!
//! 1. Finds the executable's directory
//! 2. Looks for `vendor/linux-x86_64/lib/` relative to that directory
//! 3. Sets `GST_PLUGIN_PATH` and `LD_LIBRARY_PATH` before GStreamer initialization
//!
//! # Usage
//!
//! ```ignore
//! let runtime = VendoredRuntime::new();
//! if !runtime.init() {
//!     tracing::warn!("Vendor directory not found; falling back to system libraries");
//! }
//! ```
//!
//! # LGPL Compliance
//!
//! GStreamer is licensed under LGPL-2.1+. See `vendor/README.md` for:
//! - Source availability
//! - How to relink against system GStreamer

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use tracing::{debug, info, warn};

/// Vendored runtime manager for loading bundled GStreamer libraries.
///
/// This struct manages the initialization of vendored runtime libraries,
/// ensuring thread-safe single-run semantics via an internal `OnceLock`.
///
/// # Example
///
/// ```ignore
/// let runtime = VendoredRuntime::new();
/// if !runtime.init() {
///     // Fall back to system libraries
/// }
/// ```
#[derive(Debug, Default)]
pub struct VendoredRuntime {
    /// Cached initialization result. Uses OnceLock for thread-safe single-run semantics.
    init_result: OnceLock<bool>,
}

impl VendoredRuntime {
    /// Creates a new vendored runtime manager.
    pub fn new() -> Self {
        Self {
            init_result: OnceLock::new(),
        }
    }

    /// Initialize the vendored runtime environment.
    ///
    /// This method:
    /// 1. Locates the vendor directory relative to the executable
    /// 2. Sets `GST_PLUGIN_PATH` for GStreamer plugins (prepended to existing)
    /// 3. Sets `LD_LIBRARY_PATH` for shared libraries (prepended to existing)
    ///
    /// Safe to call multiple times - initialization only happens once per instance.
    /// Subsequent calls return the cached result from the first initialization.
    ///
    /// # Returns
    ///
    /// `true` if vendored libraries were found and environment was configured,
    /// `false` if vendor directory was not found (will fall back to system libraries).
    pub fn init(&self) -> bool {
        *self.init_result.get_or_init(init_inner)
    }

    /// Returns the path to the vendored library directory, if found.
    ///
    /// Only returns a path if the directory exists and is non-empty.
    /// Useful for debugging or advanced configuration.
    pub fn vendor_lib_path(&self) -> Option<PathBuf> {
        vendor_lib_path()
    }
}

/// Internal initialization logic.
fn init_inner() -> bool {
    // Find the executable's directory
    let exe_path = match env::current_exe() {
        Ok(path) => path,
        Err(e) => {
            warn!("vendored-runtime: failed to get executable path: {e}");
            return false;
        }
    };

    let exe_dir = match exe_path.parent() {
        Some(dir) => dir,
        None => {
            warn!("vendored-runtime: executable has no parent directory");
            return false;
        }
    };

    // Look for vendor directory in several locations:
    // 1. Next to executable: ./vendor/linux-x86_64/
    // 2. In parent (for development): ../vendor/linux-x86_64/
    // 3. In workspace root (for cargo run): ../../vendor/linux-x86_64/
    let vendor_paths = [
        exe_dir.join("vendor/linux-x86_64"),
        exe_dir.join("../vendor/linux-x86_64"),
        exe_dir.join("../../vendor/linux-x86_64"),
        // For installed packages, check relative to /usr/bin
        PathBuf::from("/usr/share/lumina-video/vendor/linux-x86_64"),
    ];

    // Find vendor directory with a non-empty lib/ subdirectory
    let vendor_dir = vendor_paths
        .iter()
        .find(|p| is_non_empty_dir(&p.join("lib")));

    let vendor_dir = match vendor_dir {
        Some(dir) => dir.clone(),
        None => {
            debug!(
                "vendored-runtime: no vendor directory found, using system libraries. \
                Searched: {:?}",
                vendor_paths
            );
            return false;
        }
    };

    let lib_dir = vendor_dir.join("lib");
    let plugin_dir = lib_dir.join("gstreamer-1.0");

    info!(
        "vendored-runtime: using vendored GStreamer from {}",
        vendor_dir.display()
    );

    // Set GST_PLUGIN_PATH for GStreamer plugins
    if plugin_dir.exists() {
        let current = env::var("GST_PLUGIN_PATH").unwrap_or_default();
        let new_path = if current.is_empty() {
            plugin_dir.to_string_lossy().to_string()
        } else {
            format!("{}:{}", plugin_dir.display(), current)
        };
        env::set_var("GST_PLUGIN_PATH", &new_path);
        debug!("vendored-runtime: GST_PLUGIN_PATH={new_path}");
    }

    // Set LD_LIBRARY_PATH for shared libraries
    if lib_dir.exists() {
        let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_path = if current.is_empty() {
            lib_dir.to_string_lossy().to_string()
        } else {
            format!("{}:{}", lib_dir.display(), current)
        };
        env::set_var("LD_LIBRARY_PATH", &new_path);
        debug!("vendored-runtime: LD_LIBRARY_PATH={new_path}");
    }

    // Note: We intentionally do NOT set GST_PLUGIN_SYSTEM_PATH="" here.
    // Vendored plugins are prepended to the search path, so they take priority.
    // If the vendor bundle is incomplete, GStreamer can still fall back to
    // system plugins. For strict vendor-only mode, set GST_PLUGIN_SYSTEM_PATH=""
    // in your environment before running.

    true
}

/// Checks if a path is a directory with at least one entry.
fn is_non_empty_dir(path: &PathBuf) -> bool {
    path.is_dir()
        && path
            .read_dir()
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
}

/// Returns the path to the vendored library directory, if found.
///
/// Only returns a path if the directory exists and is non-empty.
/// Useful for debugging or advanced configuration.
pub fn vendor_lib_path() -> Option<PathBuf> {
    let exe_path = env::current_exe().ok()?;
    let exe_dir = exe_path.parent()?;

    let vendor_paths = [
        exe_dir.join("vendor/linux-x86_64/lib"),
        exe_dir.join("../vendor/linux-x86_64/lib"),
        exe_dir.join("../../vendor/linux-x86_64/lib"),
        PathBuf::from("/usr/share/lumina-video/vendor/linux-x86_64/lib"),
    ];

    vendor_paths.into_iter().find(|p| is_non_empty_dir(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_returns_consistent_result() {
        let runtime = VendoredRuntime::new();
        // All calls should return the same result (cached from first call)
        let first = runtime.init();
        let second = runtime.init();
        let third = runtime.init();
        assert_eq!(first, second);
        assert_eq!(second, third);
    }

    #[test]
    fn test_separate_instances_init_independently() {
        let runtime1 = VendoredRuntime::new();
        let runtime2 = VendoredRuntime::new();
        // Both should return the same result (same environment)
        // but they initialize independently
        let result1 = runtime1.init();
        let result2 = runtime2.init();
        assert_eq!(result1, result2);
    }
}
