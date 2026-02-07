//! MoQ URL parsing and handling.
//!
//! This module provides parsing for `moq://` and `moqs://` URLs, extracting
//! connection parameters needed to establish a MoQ session.
//!
//! # URL Format
//!
//! ```text
//! moq://host:port/namespace/track
//! moqs://host:port/namespace/track  (TLS required)
//! ```
//!
//! # Examples
//!
//! ```
//! use lumina_video::media::moq::url::MoqUrl;
//!
//! let url = MoqUrl::parse("moqs://relay.example.com:4443/live/stream").unwrap();
//! assert_eq!(url.host(), "relay.example.com");
//! assert_eq!(url.port(), 4443);
//! assert_eq!(url.namespace(), "live");
//! assert!(url.use_tls());
//! ```

use super::error::MoqError;

/// Default port for MoQ connections (HTTPS/QUIC standard port).
pub const DEFAULT_MOQ_PORT: u16 = 443;

/// Parsed MoQ URL with connection and stream parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoqUrl {
    /// Remote host
    host: String,
    /// Remote port
    port: u16,
    /// Whether to use TLS (moqs:// vs moq://)
    use_tls: bool,
    /// Namespace (first path component)
    namespace: String,
    /// Track name (remaining path, may be empty for catalog-only)
    track: Option<String>,
    /// Query string (e.g., "jwt=xxx" for authentication)
    query: Option<String>,
    /// Original URL string
    original: String,
}

impl MoqUrl {
    /// Parses a MoQ URL string.
    ///
    /// Accepts `moq://` (insecure, for testing) or `moqs://` (TLS required).
    ///
    /// # Arguments
    ///
    /// * `url` - The URL string to parse
    ///
    /// # Returns
    ///
    /// A parsed `MoqUrl` or an error if the URL is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use lumina_video::media::moq::url::MoqUrl;
    ///
    /// // Basic URL with namespace and track
    /// let url = MoqUrl::parse("moqs://relay.example.com/live/stream").unwrap();
    /// assert_eq!(url.namespace(), "live");
    /// assert_eq!(url.track(), Some("stream"));
    ///
    /// // URL with custom port
    /// let url = MoqUrl::parse("moq://localhost:4443/test/video").unwrap();
    /// assert_eq!(url.port(), 4443);
    /// assert!(!url.use_tls());
    ///
    /// // Namespace-only URL (for catalog discovery)
    /// let url = MoqUrl::parse("moqs://relay.example.com/live").unwrap();
    /// assert_eq!(url.track(), None);
    /// ```
    pub fn parse(url: &str) -> Result<Self, MoqError> {
        let original = url.to_string();

        // Check scheme
        let (use_tls, rest) = if let Some(rest) = url.strip_prefix("moqs://") {
            (true, rest)
        } else if let Some(rest) = url.strip_prefix("moq://") {
            (false, rest)
        } else {
            return Err(MoqError::InvalidUrl(
                "URL must start with moq:// or moqs://".to_string(),
            ));
        };

        // Split off query string first (everything after '?')
        let (rest, query) = match rest.find('?') {
            Some(idx) => (&rest[..idx], Some(rest[idx + 1..].to_string())),
            None => (rest, None),
        };

        // Split host:port from path
        let (authority, path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], &rest[idx + 1..]),
            None => (rest, ""),
        };

        if authority.is_empty() {
            return Err(MoqError::InvalidUrl("Missing host".to_string()));
        }

        // Parse host and port
        let (host, port) = if authority.starts_with('[') {
            // IPv6 address: [::1]:port - must start with '[' and contain ']'
            let bracket_end = authority.find(']').ok_or_else(|| {
                MoqError::InvalidUrl("IPv6 address missing closing bracket".to_string())
            })?;
            let host = &authority[1..bracket_end];
            // Check for port after the closing bracket using safe access
            let port = match authority.get(bracket_end + 1..) {
                Some("") | None => DEFAULT_MOQ_PORT,
                Some(rest) if rest.starts_with(':') => rest[1..]
                    .parse()
                    .map_err(|_| MoqError::InvalidUrl("Invalid port number".to_string()))?,
                Some(_) => {
                    return Err(MoqError::InvalidUrl(
                        "Invalid IPv6 authority suffix after ']'".to_string(),
                    ));
                }
            };
            (host.to_string(), port)
        } else if let Some(colon_idx) = authority.rfind(':') {
            // hostname:port or IPv4:port
            let host = &authority[..colon_idx];
            let port: u16 = authority[colon_idx + 1..]
                .parse()
                .map_err(|_| MoqError::InvalidUrl("Invalid port number".to_string()))?;
            (host.to_string(), port)
        } else {
            (authority.to_string(), DEFAULT_MOQ_PORT)
        };

        if host.is_empty() {
            return Err(MoqError::InvalidUrl("Empty host".to_string()));
        }

        // Parse path into namespace and optional track
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Err(MoqError::InvalidUrl(
                "Missing namespace in path".to_string(),
            ));
        }

        let (namespace, track) = match path.find('/') {
            Some(idx) => {
                let ns = &path[..idx];
                let track = &path[idx + 1..];
                (
                    ns.to_string(),
                    if track.is_empty() {
                        None
                    } else {
                        Some(track.to_string())
                    },
                )
            }
            None => (path.to_string(), None),
        };

        if namespace.is_empty() {
            return Err(MoqError::InvalidUrl("Empty namespace".to_string()));
        }

        Ok(MoqUrl {
            host,
            port,
            use_tls,
            namespace,
            track,
            query,
            original,
        })
    }

    /// Returns the remote host.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Returns the remote port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Returns whether TLS should be used.
    pub fn use_tls(&self) -> bool {
        self.use_tls
    }

    /// Returns the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the track name, if specified.
    pub fn track(&self) -> Option<&str> {
        self.track.as_deref()
    }

    /// Returns the query string, if present (e.g., "jwt=xxx").
    pub fn query(&self) -> Option<&str> {
        self.query.as_deref()
    }

    /// Returns the original URL string.
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Returns the server address as "host:port" (with IPv6 hosts bracketed).
    pub fn server_addr(&self) -> String {
        if self.host.contains(':') && !self.host.starts_with('[') {
            // IPv6 address needs brackets
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// Returns true if this URL uses the moq:// or moqs:// scheme.
    pub fn is_moq_url(url: &str) -> bool {
        url.starts_with("moq://") || url.starts_with("moqs://")
    }
}

impl std::fmt::Display for MoqUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.original)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_moqs_url() {
        let url = MoqUrl::parse("moqs://relay.example.com/live/stream").unwrap();
        assert_eq!(url.host(), "relay.example.com");
        assert_eq!(url.port(), 443);
        assert!(url.use_tls());
        assert_eq!(url.namespace(), "live");
        assert_eq!(url.track(), Some("stream"));
    }

    #[test]
    fn test_parse_moq_url_no_tls() {
        let url = MoqUrl::parse("moq://localhost:4443/test/video").unwrap();
        assert_eq!(url.host(), "localhost");
        assert_eq!(url.port(), 4443);
        assert!(!url.use_tls());
        assert_eq!(url.namespace(), "test");
        assert_eq!(url.track(), Some("video"));
    }

    #[test]
    fn test_parse_namespace_only() {
        let url = MoqUrl::parse("moqs://relay.example.com/live").unwrap();
        assert_eq!(url.namespace(), "live");
        assert_eq!(url.track(), None);
    }

    #[test]
    fn test_parse_url_with_query() {
        let url = MoqUrl::parse("moqs://cdn.moq.dev/demo/bbb?jwt=abc123").unwrap();
        assert_eq!(url.host(), "cdn.moq.dev");
        assert_eq!(url.port(), 443);
        assert_eq!(url.namespace(), "demo");
        assert_eq!(url.track(), Some("bbb"));
        assert_eq!(url.query(), Some("jwt=abc123"));
    }

    #[test]
    fn test_parse_nested_track_path() {
        let url = MoqUrl::parse("moqs://relay.example.com/live/channel/hd").unwrap();
        assert_eq!(url.namespace(), "live");
        assert_eq!(url.track(), Some("channel/hd"));
    }

    #[test]
    fn test_parse_ipv6() {
        let url = MoqUrl::parse("moqs://[::1]:4443/test/stream").unwrap();
        assert_eq!(url.host(), "::1");
        assert_eq!(url.port(), 4443);
    }

    #[test]
    fn test_parse_ipv6_default_port() {
        let url = MoqUrl::parse("moqs://[::1]/test/stream").unwrap();
        assert_eq!(url.host(), "::1");
        assert_eq!(url.port(), 443);
    }

    #[test]
    fn test_parse_ipv6_malformed_suffix() {
        assert!(MoqUrl::parse("moqs://[::1]oops/ns/track").is_err());
        assert!(MoqUrl::parse("moqs://[::1]garbage:4443/ns/track").is_err());
    }

    #[test]
    fn test_invalid_scheme() {
        assert!(MoqUrl::parse("https://example.com/live/stream").is_err());
        assert!(MoqUrl::parse("rtmp://example.com/live/stream").is_err());
    }

    #[test]
    fn test_missing_host() {
        assert!(MoqUrl::parse("moqs:///live/stream").is_err());
    }

    #[test]
    fn test_missing_namespace() {
        assert!(MoqUrl::parse("moqs://relay.example.com").is_err());
        assert!(MoqUrl::parse("moqs://relay.example.com/").is_err());
    }

    #[test]
    fn test_invalid_port() {
        assert!(MoqUrl::parse("moqs://relay.example.com:abc/live/stream").is_err());
        assert!(MoqUrl::parse("moqs://relay.example.com:99999/live/stream").is_err());
    }

    #[test]
    fn test_is_moq_url() {
        assert!(MoqUrl::is_moq_url("moq://localhost/test"));
        assert!(MoqUrl::is_moq_url("moqs://relay.example.com/live/stream"));
        assert!(!MoqUrl::is_moq_url("https://example.com/video.mp4"));
        assert!(!MoqUrl::is_moq_url("rtmp://stream.example.com/live"));
    }

    #[test]
    fn test_server_addr() {
        let url = MoqUrl::parse("moqs://relay.example.com:4443/live/stream").unwrap();
        assert_eq!(url.server_addr(), "relay.example.com:4443");
    }

    #[test]
    fn test_server_addr_ipv6() {
        let url = MoqUrl::parse("moqs://[::1]:4443/test/stream").unwrap();
        assert_eq!(url.server_addr(), "[::1]:4443");
    }
}
