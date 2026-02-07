use std::{error::Error, fmt};

use http_body_util::{BodyExt, Empty};
use hyper::{
    body::Bytes,
    header::{self},
    Request, Uri,
};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use url::Url;

/// Default max body size (20MB) for general HTTP requests
const MAX_BODY_BYTES: usize = 20 * 1024 * 1024;

/// Max body size for video downloads (1GB)
const MAX_VIDEO_BODY_BYTES: usize = 1024 * 1024 * 1024;

pub async fn http_req(url: &str) -> Result<HyperHttpResponse, HyperHttpError> {
    http_req_with_limit(url, MAX_BODY_BYTES).await
}

/// HTTP request with larger body size limit for video downloads
pub async fn http_req_video(url: &str) -> Result<HyperHttpResponse, HyperHttpError> {
    http_req_with_limit(url, MAX_VIDEO_BODY_BYTES).await
}

/// Stream a video download to a file, calling progress callback for each chunk.
/// The on_start callback receives the content_length (if known from headers) before streaming.
/// Returns (content_length, content_type) if available.
pub async fn http_stream_to_file<S, F>(
    url: &str,
    mut file: std::fs::File,
    mut on_start: S,
    mut on_chunk: F,
) -> Result<(Option<u64>, Option<String>), HyperHttpError>
where
    S: FnMut(Option<u64>),
    F: FnMut(usize),
{
    use std::io::Write;

    let mut current_uri: Uri = url.parse().map_err(|_| HyperHttpError::Uri)?;

    #[cfg(target_os = "android")]
    let https = {
        HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .build()
    };

    #[cfg(not(target_os = "android"))]
    let https = {
        let builder = HttpsConnectorBuilder::new()
            .with_native_roots()
            .map_err(|err| {
                tracing::error!("Failed to load native root certificates: {err}");
                HyperHttpError::TlsConfig
            })?;
        builder.https_or_http().enable_http1().build()
    };

    let client: Client<_, Empty<Bytes>> = Client::builder(TokioExecutor::new()).build(https);

    const MAX_REDIRECTS: usize = 5;
    let mut redirects = 0;

    let res = loop {
        let authority = current_uri.authority().ok_or(HyperHttpError::Host)?.clone();

        let req = Request::builder()
            .uri(current_uri.clone())
            .header(hyper::header::HOST, authority.as_str())
            .body(Empty::<Bytes>::new())
            .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        let res = client
            .request(req)
            .await
            .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        if res.status().is_redirection() {
            if redirects >= MAX_REDIRECTS {
                return Err(HyperHttpError::TooManyRedirects);
            }

            let location_header = res
                .headers()
                .get(header::LOCATION)
                .ok_or(HyperHttpError::MissingRedirectLocation)?
                .clone();

            let location = location_header
                .to_str()
                .map_err(|_| HyperHttpError::InvalidRedirectLocation)?
                .to_string();

            res.into_body()
                .collect()
                .await
                .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

            current_uri = resolve_redirect(&current_uri, &location)?;
            redirects += 1;
            continue;
        } else if !res.status().is_success() {
            // Return error for non-2xx status codes
            return Err(HyperHttpError::HttpStatus(res.status().as_u16()));
        } else {
            break res;
        }
    };

    let content_type: Option<String> = res
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|t| t.to_str().ok())
        .map(|s| s.to_string());

    let content_length: Option<u64> = res
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|s| s.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    // Notify caller of content length BEFORE streaming starts
    on_start(content_length);

    // Stream body directly to file
    let mut body = res.into_body();

    while let Some(frame_result) = body.frame().await {
        let frame = frame_result.map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        if let Ok(chunk) = frame.into_data() {
            let chunk: Bytes = chunk;
            file.write_all(&chunk)
                .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;
            on_chunk(chunk.len());
        }
    }

    // Sync once at the end to ensure all data is flushed to disk
    file.sync_all()
        .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

    Ok((content_length, content_type))
}

async fn http_req_with_limit(
    url: &str,
    max_body_bytes: usize,
) -> Result<HyperHttpResponse, HyperHttpError> {
    let mut current_uri: Uri = url.parse().map_err(|_| HyperHttpError::Uri)?;

    // On Android, use bundled webpki-roots since native root certs aren't accessible
    // from Rust native code. On other platforms, use native system roots.
    #[cfg(target_os = "android")]
    let https = {
        let builder = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .build();
        builder
    };

    #[cfg(not(target_os = "android"))]
    let https = {
        let builder = HttpsConnectorBuilder::new()
            .with_native_roots()
            .map_err(|err| {
                tracing::error!("Failed to load native root certificates: {err}");
                HyperHttpError::TlsConfig
            })?;

        builder.https_or_http().enable_http1().build()
    };

    let client: Client<_, Empty<Bytes>> = Client::builder(TokioExecutor::new()).build(https);

    const MAX_REDIRECTS: usize = 5;
    let mut redirects = 0;

    let res = loop {
        let authority = current_uri.authority().ok_or(HyperHttpError::Host)?.clone();

        // Fetch the url...
        let req = Request::builder()
            .uri(current_uri.clone())
            .header(hyper::header::HOST, authority.as_str())
            .body(Empty::<Bytes>::new())
            .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        let res = client
            .request(req)
            .await
            .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        if res.status().is_redirection() {
            if redirects >= MAX_REDIRECTS {
                return Err(HyperHttpError::TooManyRedirects);
            }

            let location_header = res
                .headers()
                .get(header::LOCATION)
                .ok_or(HyperHttpError::MissingRedirectLocation)?
                .clone();

            let location = location_header
                .to_str()
                .map_err(|_| HyperHttpError::InvalidRedirectLocation)?
                .to_string();

            res.into_body()
                .collect()
                .await
                .map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

            current_uri = resolve_redirect(&current_uri, &location)?;
            redirects += 1;
            continue;
        } else {
            break res;
        }
    };

    let content_type: Option<String> = res
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|t| t.to_str().ok())
        .map(|s| s.to_string());

    let content_length: Option<usize> = res
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|s| s.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok());

    if let Some(len) = content_length {
        if len > max_body_bytes {
            return Err(HyperHttpError::BodyTooLarge);
        }
    }

    // Stream body with incremental size checking to avoid memory exhaustion
    let mut body = res.into_body();
    let mut bytes = Vec::with_capacity(content_length.unwrap_or(0).min(max_body_bytes));

    while let Some(frame_result) = body.frame().await {
        let frame = frame_result.map_err(|e| HyperHttpError::Hyper(Box::new(e)))?;

        if let Ok(chunk) = frame.into_data() {
            let chunk: Bytes = chunk;
            // Check size BEFORE allocating
            if bytes.len() + chunk.len() > max_body_bytes {
                return Err(HyperHttpError::BodyTooLarge);
            }
            bytes.extend_from_slice(&chunk);
        }
    }

    Ok(HyperHttpResponse {
        content_type,
        bytes,
    })
}

#[derive(Debug)]
pub enum HyperHttpError {
    Hyper(Box<dyn std::error::Error + Send + Sync>),
    Host,
    Uri,
    BodyTooLarge,
    TooManyRedirects,
    MissingRedirectLocation,
    InvalidRedirectLocation,
    TlsConfig,
    HttpStatus(u16),
}

#[derive(Debug)]
pub struct HyperHttpResponse {
    pub content_type: Option<String>,
    pub bytes: Vec<u8>,
}

impl Error for HyperHttpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hyper(e) => Some(&**e),
            _ => None,
        }
    }
}

impl fmt::Display for HyperHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hyper(e) => write!(f, "Hyper error: {e}"),
            Self::Host => write!(f, "Missing host in URL"),
            Self::Uri => write!(f, "Invalid URI"),
            Self::BodyTooLarge => write!(f, "Body too large"),
            Self::TooManyRedirects => write!(f, "Too many redirect responses"),
            Self::MissingRedirectLocation => write!(f, "Redirect response missing Location header"),
            Self::InvalidRedirectLocation => write!(f, "Invalid redirect Location header"),
            Self::TlsConfig => write!(f, "TLS configuration error (missing root certificates)"),
            Self::HttpStatus(code) => write!(f, "HTTP error status: {code}"),
        }
    }
}

fn resolve_redirect(current: &Uri, location: &str) -> Result<Uri, HyperHttpError> {
    if let Ok(uri) = location.parse::<Uri>() {
        if uri.scheme().is_some() {
            return Ok(uri);
        }
    }

    let base = Url::parse(&current.to_string()).map_err(|_| HyperHttpError::Uri)?;
    let joined = base
        .join(location)
        .map_err(|_| HyperHttpError::InvalidRedirectLocation)?;

    joined
        .as_str()
        .parse::<Uri>()
        .map_err(|_| HyperHttpError::InvalidRedirectLocation)
}
