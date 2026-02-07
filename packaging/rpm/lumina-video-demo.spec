Name:           lumina-video-demo
Version:        0.1.0
Release:        1%{?dist}
Summary:        Hardware-accelerated video player demo for egui

License:        MIT OR Apache-2.0
URL:            https://github.com/lumina-video/lumina-video
Source0:        %{name}-%{version}.tar.gz

# Fedora Rust packaging macros (required for Koji/mock builds)
BuildRequires:  cargo-rpm-macros >= 24
BuildRequires:  rust-packaging
BuildRequires:  pkg-config
BuildRequires:  cmake
BuildRequires:  gstreamer1-devel
BuildRequires:  gstreamer1-plugins-base-devel
BuildRequires:  vulkan-headers
BuildRequires:  vulkan-loader-devel
BuildRequires:  wayland-devel
BuildRequires:  libxkbcommon-devel
BuildRequires:  libX11-devel
BuildRequires:  libXcursor-devel
BuildRequires:  libXrandr-devel
BuildRequires:  libXi-devel

# =============================================================================
# Runtime dependencies for zero-copy playback
# =============================================================================
# Core GStreamer (available in base repos for all distros)
Requires:       gstreamer1-plugins-base
Requires:       gstreamer1-plugins-good
Requires:       gstreamer1-plugins-bad-free
Requires:       vulkan-loader

# VA-API drivers (optional but recommended for hardware acceleration)
# Intel: intel-media-driver or libva-intel-driver
# AMD: mesa-va-drivers
# These are in base repos; users should install appropriate driver for their GPU
Recommends:     mesa-va-drivers

# -----------------------------------------------------------------------------
# Fedora-specific packages (available in base Fedora repos)
# -----------------------------------------------------------------------------
%if 0%{?fedora}
# gstreamer1-plugins-ugly-free: MP3, MPEG-2, etc. (Fedora base repo)
Requires:       gstreamer1-plugins-ugly-free
# gstreamer1-plugin-libav: FFmpeg-based decoders (Fedora base repo)
Requires:       gstreamer1-plugin-libav
# gstreamer1-vaapi: VA-API GStreamer plugin (Fedora base repo)
Requires:       gstreamer1-vaapi
%endif

# -----------------------------------------------------------------------------
# RHEL/CentOS-specific packages
# NOTE: Some packages require EPEL or RPM Fusion repositories
# -----------------------------------------------------------------------------
%if 0%{?rhel}
# gstreamer1-plugins-ugly: Requires RPM Fusion Free repository
# Install RPM Fusion: https://rpmfusion.org/Configuration
# sudo dnf install https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm
Requires:       gstreamer1-plugins-ugly
# gstreamer1-libav: Requires RPM Fusion Free repository
Requires:       gstreamer1-plugin-libav
# gstreamer1-vaapi: Available in EPEL or RPM Fusion
# sudo dnf install epel-release
Requires:       gstreamer1-vaapi
%endif

%description
lumina-video-demo is a demonstration application showcasing the lumina-video
video player library with zero-copy GPU rendering.

Features:
- Hardware-accelerated video decoding (VA-API)
- Zero-copy GPU rendering (DMABuf -> Vulkan)
- HLS/m3u8 streaming support
- Subtitle support (SRT, WebVTT)

%prep
%autosetup
%cargo_prep

%build
%cargo_build --package lumina-video-demo

%install
install -D -m 755 target/release/lumina-video-demo %{buildroot}%{_bindir}/lumina-video-demo

%check
# Skip tests - they require display/GPU access not available in mock
# %cargo_test -- --package lumina-video-demo

%files
%license LICENSE
%doc README.md
%{_bindir}/lumina-video-demo

%changelog
* Mon Feb 02 2026 alltheseas <alltheseas@users.noreply.github.com> - 0.1.0-1
- Initial release
- Hardware-accelerated video playback with zero-copy GPU rendering
- GStreamer + VA-API backend for Linux
