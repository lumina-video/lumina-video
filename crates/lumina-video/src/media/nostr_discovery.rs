//! Nostr-based MoQ stream discovery using NIP-53 Live Activities.
//!
//! This module discovers live MoQ streams by subscribing to Nostr relays
//! and filtering for kind:30311 (Live Streaming Event) with MoQ URLs.
//!
//! # NIP-53 Live Activities
//!
//! Live streams are announced via addressable events (kind:30311) containing:
//! - `d` tag: unique stream identifier
//! - `title` tag: stream name
//! - `streaming` tag: stream URL (we filter for moq:// or moqs://)
//! - `status` tag: planned, live, or ended
//! - `image` tag: thumbnail URL
//! - `p` tags: participants (host, speakers)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use nostr_sdk::prelude::*;
use parking_lot::RwLock;

/// Default Nostr relays for stream discovery (reliable subset from zap.stream).
/// Using fewer relays to avoid timeout issues.
pub const DEFAULT_RELAYS: &[&str] = &["wss://nos.lol", "wss://relay.damus.io"];

/// A discovered MoQ live stream from Nostr.
#[derive(Debug, Clone)]
pub struct MoqStream {
    /// Unique identifier (from `d` tag)
    pub id: String,
    /// Stream title (from `title` tag)
    pub title: Option<String>,
    /// MoQ streaming URL (from `streaming` tag)
    pub url: String,
    /// Stream status: planned, live, ended
    pub status: StreamStatus,
    /// Thumbnail image URL (from `image` tag)
    pub image: Option<String>,
    /// Host pubkey (first `p` tag with Host role)
    pub host: Option<String>,
    /// Event timestamp
    pub updated_at: Timestamp,
    /// Original Nostr event ID
    pub event_id: EventId,
}

/// Stream status from NIP-53.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamStatus {
    Planned,
    #[default]
    Live,
    Ended,
}

impl From<&str> for StreamStatus {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "planned" => StreamStatus::Planned,
            "live" => StreamStatus::Live,
            "ended" => StreamStatus::Ended,
            _ => StreamStatus::Live,
        }
    }
}

/// Event sent from the discovery background task.
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// A new stream was discovered or updated
    StreamUpdated(MoqStream),
    /// A stream went offline (status changed to ended)
    StreamEnded(String),
    /// Connection status changed
    Connected(bool),
    /// Error occurred
    Error(String),
}

/// Nostr MoQ stream discovery service.
pub struct NostrDiscovery {
    /// Discovered streams indexed by ID
    streams: Arc<RwLock<HashMap<String, MoqStream>>>,
    /// Shutdown flag for background thread
    shutdown: Arc<AtomicBool>,
    /// Whether discovery is running
    running: Arc<AtomicBool>,
    /// Handle to the discovery thread
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl NostrDiscovery {
    /// Creates a new discovery service.
    pub fn new() -> Self {
        Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: None,
        }
    }

    /// Starts the discovery background task.
    ///
    /// Returns a receiver for discovery events.
    pub fn start(&mut self) -> std::sync::mpsc::Receiver<DiscoveryEvent> {
        // Stop any existing task
        self.stop();

        let (event_tx, event_rx) = std::sync::mpsc::channel();

        let streams = self.streams.clone();
        let shutdown = self.shutdown.clone();
        let running = self.running.clone();

        // Reset shutdown flag
        shutdown.store(false, Ordering::SeqCst);
        running.store(true, Ordering::SeqCst);

        // Spawn a plain thread that creates its own Tokio runtime
        let handle = std::thread::Builder::new()
            .name("nostr-discovery".into())
            .spawn(move || {
                // Create a Tokio runtime inside this thread
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = event_tx.send(DiscoveryEvent::Error(format!(
                            "Failed to create runtime: {}",
                            e
                        )));
                        running.store(false, Ordering::SeqCst);
                        return;
                    }
                };

                // Run the async discovery on this runtime
                rt.block_on(async move {
                    if let Err(e) = run_discovery(streams, event_tx.clone(), shutdown).await {
                        let _ = event_tx.send(DiscoveryEvent::Error(e.to_string()));
                    }
                });

                running.store(false, Ordering::SeqCst);
            })
            .ok();

        self.thread_handle = handle;
        event_rx
    }

    /// Stops the discovery background task.
    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Returns true if discovery is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Gets all currently known live streams.
    pub fn get_streams(&self) -> Vec<MoqStream> {
        self.streams
            .read()
            .values()
            .filter(|s| s.status == StreamStatus::Live)
            .cloned()
            .collect()
    }

    /// Gets a stream by ID.
    pub fn get_stream(&self, id: &str) -> Option<MoqStream> {
        self.streams.read().get(id).cloned()
    }
}

impl Default for NostrDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NostrDiscovery {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Background task that connects to Nostr relays and discovers MoQ streams.
async fn run_discovery(
    streams: Arc<RwLock<HashMap<String, MoqStream>>>,
    event_tx: std::sync::mpsc::Sender<DiscoveryEvent>,
    shutdown: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing::info!("Nostr discovery: starting");

    // Install rustls crypto provider (required for rustls 0.23+)
    // This must be done before any TLS connections are made
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Create Nostr client
    let client = Client::default();

    // Add relays
    for relay in DEFAULT_RELAYS {
        if let Err(e) = client.add_relay(*relay).await {
            tracing::warn!("Nostr discovery: failed to add relay {}: {}", relay, e);
        }
    }

    // Connect to relays
    client.connect().await;
    let _ = event_tx.send(DiscoveryEvent::Connected(true));
    tracing::info!("Nostr discovery: connected to relays");

    // Create filter for NIP-53 Live Activities (kind:30311)
    // Like zap.stream, we don't use since/limit - just subscribe to all live stream events
    let filter = Filter::new().kind(Kind::Custom(30311));

    // Subscribe directly (like zap.stream's leaveOpen: true)
    // This receives both historical events and real-time updates
    tracing::info!("Nostr discovery: subscribing to live streams...");
    let sub_output = client.subscribe(vec![filter], None).await?;
    let sub_id = sub_output.val;
    let _ = event_tx.send(DiscoveryEvent::Connected(true));
    tracing::info!("Nostr discovery: subscribed, waiting for events...");

    // Get notifications receiver (must be kept alive for the loop)
    let mut notifications = client.notifications();
    let mut event_count = 0usize;

    // Handle events until shutdown
    loop {
        // Check shutdown flag
        if shutdown.load(Ordering::SeqCst) {
            tracing::info!("Nostr discovery: shutting down");
            break;
        }

        // Use timeout to periodically check shutdown flag
        match tokio::time::timeout(Duration::from_millis(500), notifications.recv()).await {
            Ok(Ok(RelayPoolNotification::Event { event, .. })) => {
                event_count += 1;
                if event_count <= 5 || event_count.is_multiple_of(50) {
                    tracing::debug!(
                        "Nostr: received event {} (kind {})",
                        event_count,
                        event.kind.as_u16()
                    );
                }
                if let Some(stream) = parse_live_event(&event) {
                    tracing::info!(
                        "Nostr discovery: found MoQ stream '{}' at {}",
                        stream.title.as_deref().unwrap_or(&stream.id),
                        stream.url
                    );
                    if stream.status == StreamStatus::Ended {
                        streams.write().remove(&stream.id);
                        let _ = event_tx.send(DiscoveryEvent::StreamEnded(stream.id));
                    } else {
                        streams.write().insert(stream.id.clone(), stream.clone());
                        let _ = event_tx.send(DiscoveryEvent::StreamUpdated(stream));
                    }
                }
            }
            Ok(Ok(RelayPoolNotification::Shutdown)) => {
                tracing::warn!("Nostr discovery: relay pool shutdown");
                let _ = event_tx.send(DiscoveryEvent::Connected(false));
                break;
            }
            Ok(Err(e)) => {
                tracing::error!("Nostr discovery: notification error: {}", e);
            }
            Ok(Ok(_)) => {}
            Err(_) => {
                // Timeout - just loop again to check shutdown flag
            }
        }
    }

    // Cleanup
    client.unsubscribe(sub_id).await;
    client.disconnect().await?;
    let _ = event_tx.send(DiscoveryEvent::Connected(false));

    Ok(())
}

/// Parses a NIP-53 Live Activity event into a MoqStream.
///
/// Returns None if the event doesn't contain a MoQ URL.
fn parse_live_event(event: &Event) -> Option<MoqStream> {
    // Must be kind 30311
    if event.kind != Kind::Custom(30311) {
        return None;
    }

    let tags = &event.tags;

    // Extract d tag (identifier)
    let id = tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("d"))
        .and_then(|t| t.as_slice().get(1))
        .map(|s| s.to_string())?;

    // Extract MoQ streaming URL - there may be multiple streaming tags (HLS + MoQ)
    // We need to find the one that starts with moq:// or moqs://
    //
    // Per NIP-53 and zap.stream's implementation:
    // - The `streaming` tag contains the base MoQ relay URL (e.g., "moq://api-core.zap.stream:1443/")
    // - The `d` tag contains the broadcast identifier (e.g., "537a365c-f1ec-44ac-af10-22d14a7319fb")
    // - zap.stream passes these separately to the Hang library:
    //   - URL → connection (Hang.Moq.Connection.Reload)
    //   - d-tag → broadcast path (Hang.Moq.Path.from(`/${id}`))
    //
    // For our MoqUrl parser, we need a single URL. When the streaming URL ends with '/',
    // we append the d-tag so the MoqUrl parser can extract it as the namespace/broadcast path.
    // The moq_decoder then detects UUID-like namespaces and connects to the base URL while
    // using the namespace as the broadcast path.
    let mut url = tags
        .iter()
        .filter(|t| t.as_slice().first().map(|s| s.as_str()) == Some("streaming"))
        .filter_map(|t| t.as_slice().get(1).map(|s| s.to_string()))
        .find(|url| url.starts_with("moq://") || url.starts_with("moqs://"))?;

    // Append d-tag as path if URL has no path (ends with '/')
    // This allows the MoqUrl parser to extract the broadcast ID
    if url.ends_with('/') {
        url.push_str(&id);
    }

    // Extract optional fields
    let title = tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("title"))
        .and_then(|t| t.as_slice().get(1))
        .map(|s| s.to_string());

    let status = tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("status"))
        .and_then(|t| t.as_slice().get(1))
        .map(|s| StreamStatus::from(s.as_str()))
        .unwrap_or(StreamStatus::Live);

    let image = tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("image"))
        .and_then(|t| t.as_slice().get(1))
        .map(|s| s.to_string());

    // Find host (first p tag with "Host" role)
    let host = tags
        .iter()
        .filter(|t| t.as_slice().first().map(|s| s.as_str()) == Some("p"))
        .find(|t| t.as_slice().get(3).map(|s| s.as_str()) == Some("Host"))
        .and_then(|t| t.as_slice().get(1))
        .map(|s| s.to_string());

    Some(MoqStream {
        id,
        title,
        url,
        status,
        image,
        host,
        updated_at: event.created_at,
        event_id: event.id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_status_from_str() {
        assert_eq!(StreamStatus::from("live"), StreamStatus::Live);
        assert_eq!(StreamStatus::from("Live"), StreamStatus::Live);
        assert_eq!(StreamStatus::from("planned"), StreamStatus::Planned);
        assert_eq!(StreamStatus::from("ended"), StreamStatus::Ended);
        assert_eq!(StreamStatus::from("unknown"), StreamStatus::Live);
    }
}
