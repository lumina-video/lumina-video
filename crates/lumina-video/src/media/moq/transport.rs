//! MoQ transport layer for QUIC connection management.
//!
//! This module handles establishing and managing MoQ sessions over QUIC,
//! using the moq-native crate for connection handling.

use super::error::MoqError;
use super::url::MoqUrl;

use std::sync::Arc;
use tokio::sync::Mutex;

/// MoQ transport state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportState {
    /// Not connected
    Disconnected,
    /// Connecting to server
    Connecting,
    /// Connected and session established
    Connected,
    /// Connection failed
    Failed,
}

/// Configuration for MoQ transport.
#[derive(Debug, Clone)]
pub struct MoqTransportConfig {
    /// Whether to disable TLS certificate verification (for testing only)
    pub disable_tls_verify: bool,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Whether to enable WebSocket fallback
    pub websocket_fallback: bool,
}

impl Default for MoqTransportConfig {
    fn default() -> Self {
        Self {
            disable_tls_verify: false,
            connect_timeout_ms: 10000,
            websocket_fallback: true,
        }
    }
}

/// The protocol used for the MoQ connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqProtocol {
    /// QUIC / WebTransport (preferred, low latency)
    Quic,
    /// WebSocket fallback (TCP-based, higher latency)
    WebSocket,
    /// Not yet connected
    Unknown,
}

impl std::fmt::Display for MoqProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoqProtocol::Quic => write!(f, "QUIC"),
            MoqProtocol::WebSocket => write!(f, "WebSocket"),
            MoqProtocol::Unknown => write!(f, "unknown"),
        }
    }
}

/// MoQ transport managing QUIC connection and session.
pub struct MoqTransport {
    /// Parsed MoQ URL
    url: MoqUrl,
    /// Transport configuration
    config: MoqTransportConfig,
    /// Current transport state
    state: Arc<Mutex<TransportState>>,
    /// Active session (set after successful connection)
    session: Option<moq_lite::Session>,
    /// Which protocol was used for the connection
    protocol: MoqProtocol,
}

impl MoqTransport {
    /// Creates a new MoQ transport for the given URL.
    pub fn new(url: MoqUrl, config: MoqTransportConfig) -> Self {
        Self {
            url,
            config,
            state: Arc::new(Mutex::new(TransportState::Disconnected)),
            session: None,
            protocol: MoqProtocol::Unknown,
        }
    }

    /// Returns the current transport state.
    pub async fn state(&self) -> TransportState {
        *self.state.lock().await
    }

    /// Returns the parsed URL.
    pub fn url(&self) -> &MoqUrl {
        &self.url
    }

    /// Connects to the MoQ server and establishes a session.
    ///
    /// This performs the QUIC handshake and MoQ session establishment.
    /// Returns the session for subscribing to tracks.
    pub async fn connect(&mut self) -> Result<&moq_lite::Session, MoqError> {
        // Helper to set state to Failed - inlined to avoid lifetime issues with async closures
        async fn set_failed(state: &Arc<Mutex<TransportState>>) {
            let mut s = state.lock().await;
            *s = TransportState::Failed;
        }

        // Update state to connecting
        {
            let mut state = self.state.lock().await;
            *state = TransportState::Connecting;
        }

        tracing::info!(
            "MoQ transport: connecting to {}:{} (tls={}, ws_fallback={}, timeout={}ms)",
            self.url.host(),
            self.url.port(),
            self.url.use_tls(),
            self.config.websocket_fallback,
            self.config.connect_timeout_ms,
        );

        // Build the connection URL
        // moq-native expects https:// or http:// URLs
        let scheme = if self.url.use_tls() { "https" } else { "http" };
        let base_url = format!(
            "{}://{}/{}",
            scheme,
            self.url.server_addr(),
            self.url.namespace()
        );
        let connect_url = match self.url.query() {
            Some(query) => format!("{}?{}", base_url, query),
            None => base_url,
        };

        // Parse the URL for moq-native
        let parsed_url: url::Url = match connect_url.parse() {
            Ok(u) => u,
            Err(e) => {
                tracing::error!("MoQ transport: invalid URL: {}", e);
                set_failed(&self.state).await;
                return Err(MoqError::InvalidUrl(format!("Invalid URL: {e}")));
            }
        };

        let connect_start = std::time::Instant::now();

        let redacted_connect_url = connect_url
            .split_once('?')
            .map(|(base, _)| format!("{}?<redacted>", base))
            .unwrap_or_else(|| connect_url.clone());
        tracing::info!("MoQ transport: connecting to {}", redacted_connect_url);

        // Two-phase connect: try QUIC first, fall back to WebSocket.
        // This lets us accurately track which protocol was used, since
        // moq-native's built-in race doesn't expose the winner.
        let timeout_duration = std::time::Duration::from_millis(self.config.connect_timeout_ms);
        let quic_probe_timeout =
            std::time::Duration::from_millis(self.config.connect_timeout_ms.min(1500));

        let (session, protocol) = if self.config.websocket_fallback {
            // Phase 1: QUIC-only with short timeout
            let quic_result = {
                let mut cfg = moq_native::ClientConfig::default();
                if self.config.disable_tls_verify {
                    cfg.tls.disable_verify = Some(true);
                }
                cfg.websocket.enabled = false;
                match cfg.init() {
                    Ok(client) => {
                        match tokio::time::timeout(
                            quic_probe_timeout,
                            client.connect(parsed_url.clone()),
                        )
                        .await
                        {
                            Ok(Ok(s)) => Ok(s),
                            Ok(Err(e)) => {
                                tracing::debug!("MoQ transport: QUIC connect error: {}", e);
                                Err(())
                            }
                            Err(_) => {
                                tracing::debug!(
                                    "MoQ transport: QUIC timed out ({:?})",
                                    quic_probe_timeout
                                );
                                Err(())
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("MoQ transport: QUIC client init failed: {}", e);
                        Err(())
                    }
                }
            };

            match quic_result {
                Ok(s) => {
                    tracing::info!(
                        "MoQ transport: connected via QUIC in {:.1}ms",
                        connect_start.elapsed().as_secs_f64() * 1000.0
                    );
                    (s, MoqProtocol::Quic)
                }
                Err(()) => {
                    // Phase 2: WebSocket fallback
                    tracing::info!("MoQ transport: QUIC unavailable, trying WebSocket...");
                    let mut cfg = moq_native::ClientConfig::default();
                    if self.config.disable_tls_verify {
                        cfg.tls.disable_verify = Some(true);
                    }
                    cfg.websocket.enabled = true;
                    cfg.websocket.delay = Some(std::time::Duration::ZERO);
                    let client = match cfg.init() {
                        Ok(c) => c,
                        Err(e) => {
                            set_failed(&self.state).await;
                            return Err(MoqError::ConnectionFailed(format!(
                                "WebSocket client init failed: {e}"
                            )));
                        }
                    };
                    match tokio::time::timeout(timeout_duration, client.connect(parsed_url)).await {
                        Ok(Ok(s)) => {
                            tracing::info!(
                                "MoQ transport: connected via WebSocket in {:.1}ms",
                                connect_start.elapsed().as_secs_f64() * 1000.0
                            );
                            (s, MoqProtocol::WebSocket)
                        }
                        Ok(Err(e)) => {
                            set_failed(&self.state).await;
                            return Err(MoqError::ConnectionFailed(format!(
                                "WebSocket connection failed: {e}"
                            )));
                        }
                        Err(_) => {
                            set_failed(&self.state).await;
                            return Err(MoqError::Timeout("Connection timed out".to_string()));
                        }
                    }
                }
            }
        } else {
            // WebSocket disabled, QUIC only
            let mut cfg = moq_native::ClientConfig::default();
            if self.config.disable_tls_verify {
                cfg.tls.disable_verify = Some(true);
            }
            cfg.websocket.enabled = false;
            let client = match cfg.init() {
                Ok(c) => c,
                Err(e) => {
                    set_failed(&self.state).await;
                    return Err(MoqError::ConnectionFailed(format!(
                        "Failed to init client: {e}"
                    )));
                }
            };
            match tokio::time::timeout(timeout_duration, client.connect(parsed_url)).await {
                Ok(Ok(s)) => {
                    tracing::info!(
                        "MoQ transport: connected via QUIC in {:.1}ms",
                        connect_start.elapsed().as_secs_f64() * 1000.0
                    );
                    (s, MoqProtocol::Quic)
                }
                Ok(Err(e)) => {
                    set_failed(&self.state).await;
                    return Err(MoqError::ConnectionFailed(format!(
                        "Connection failed: {e}"
                    )));
                }
                Err(_) => {
                    set_failed(&self.state).await;
                    return Err(MoqError::Timeout("Connection timed out".to_string()));
                }
            }
        };

        // Store session and protocol
        self.session = Some(session);
        self.protocol = protocol;

        // Update state to connected
        {
            let mut state = self.state.lock().await;
            *state = TransportState::Connected;
        }

        match self.session.as_ref() {
            Some(s) => Ok(s),
            None => {
                set_failed(&self.state).await;
                Err(MoqError::ConnectionFailed(
                    "Session not established".to_string(),
                ))
            }
        }
    }

    /// Returns a reference to the active session, if connected.
    pub fn session(&self) -> Option<&moq_lite::Session> {
        self.session.as_ref()
    }

    /// Returns the protocol used for the connection.
    pub fn protocol(&self) -> MoqProtocol {
        self.protocol
    }

    /// Disconnects from the server.
    pub async fn disconnect(&mut self) {
        tracing::info!(
            "MoQ transport: disconnecting from {}:{}",
            self.url.host(),
            self.url.port()
        );
        self.session = None;

        let mut state = self.state.lock().await;
        *state = TransportState::Disconnected;
    }

    /// Returns whether the transport is currently connected.
    pub async fn is_connected(&self) -> bool {
        *self.state.lock().await == TransportState::Connected
    }
}

impl std::fmt::Debug for MoqTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoqTransport")
            .field("url", &self.url)
            .field("config", &self.config)
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_config_default() {
        let config = MoqTransportConfig::default();
        assert!(!config.disable_tls_verify);
        assert_eq!(config.connect_timeout_ms, 10000);
        assert!(config.websocket_fallback);
    }

    #[test]
    fn test_transport_new() {
        let url = MoqUrl::parse("moqs://relay.example.com/live/stream").unwrap();
        let transport = MoqTransport::new(url, MoqTransportConfig::default());
        assert_eq!(transport.url().host(), "relay.example.com");
    }

    #[tokio::test]
    async fn test_transport_initial_state() {
        let url = MoqUrl::parse("moqs://relay.example.com/live/stream").unwrap();
        let transport = MoqTransport::new(url, MoqTransportConfig::default());
        assert_eq!(transport.state().await, TransportState::Disconnected);
        assert!(!transport.is_connected().await);
    }
}
