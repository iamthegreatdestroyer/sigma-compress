//! Streaming decompression and download client for sigma-compress.
//!
//! Provides async streaming decompression using tokio and reqwest,
//! enabling chunked download-and-decompress workflows.

use crate::config::CompressionConfig;
use crate::error::CompressError;
use crate::{CompressedOutput, Compressor};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};

/// Configuration for streaming operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Size of each read chunk in bytes (default: 64KB).
    pub chunk_size: usize,
    /// Request timeout in seconds (default: 30).
    pub timeout_secs: u64,
    /// Number of retry attempts on transient failure (default: 3).
    pub max_retries: u32,
    /// Base compression config.
    pub compression: CompressionConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 65_536,
            timeout_secs: 30,
            max_retries: 3,
            compression: CompressionConfig::default(),
        }
    }
}

/// A streaming decompressor that wraps a `Compressor` for async operations.
pub struct StreamingDecompressor {
    compressor: Compressor,
    config: StreamingConfig,
}

impl StreamingDecompressor {
    /// Create a new streaming decompressor with default streaming config.
    pub fn new(compressor: Compressor) -> Self {
        Self {
            compressor,
            config: StreamingConfig::default(),
        }
    }

    /// Create a new streaming decompressor with custom streaming config.
    pub fn with_config(config: StreamingConfig) -> Self {
        let compressor = Compressor::new(config.compression.clone());
        Self { compressor, config }
    }

    /// Download compressed data from a URL and decompress it.
    ///
    /// Uses reqwest to fetch the payload, deserializes a `CompressedOutput`
    /// envelope, then decompresses via the inner `Compressor`.
    pub async fn stream_download(&self, url: &str) -> Result<Vec<u8>, CompressError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(|e| CompressError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("HTTP client error: {e}"),
            )))?;

        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tracing::warn!(attempt, url, "Retrying download");
            }

            match self.try_download(&client, url).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::error!(attempt, url, error = %e, "Download attempt failed");
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(CompressError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "All download attempts exhausted",
        ))))
    }

    /// Single download attempt.
    async fn try_download(
        &self,
        client: &reqwest::Client,
        url: &str,
    ) -> Result<Vec<u8>, CompressError> {
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| CompressError::IoError(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                format!("Request failed: {e}"),
            )))?;

        if !response.status().is_success() {
            return Err(CompressError::RyzansteinError(format!(
                "HTTP {} from {}",
                response.status(),
                url
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| CompressError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Failed to read response body: {e}"),
            )))?;

        let compressed: CompressedOutput = bincode::deserialize(&bytes)
            .map_err(|e| CompressError::SerializationError(format!(
                "Failed to deserialize CompressedOutput: {e}"
            )))?;

        self.compressor.decompress(&compressed)
    }

    /// Decompress data from an async reader that yields a serialized
    /// `CompressedOutput` envelope.
    ///
    /// Reads the entire stream into memory, deserializes, then decompresses.
    pub async fn decompress_stream<R: AsyncRead + Unpin>(
        &self,
        reader: &mut R,
    ) -> Result<Vec<u8>, CompressError> {
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .await
            .map_err(CompressError::IoError)?;

        if buf.is_empty() {
            return Err(CompressError::EmptyInput);
        }

        let compressed: CompressedOutput = bincode::deserialize(&buf)
            .map_err(|e| CompressError::SerializationError(format!(
                "Stream deserialization failed: {e}"
            )))?;

        self.compressor.decompress(&compressed)
    }

    /// Decompress data from an async reader in chunks, yielding each
    /// decompressed block via callback.
    ///
    /// Each chunk is a length-prefixed (4-byte LE) serialized `CompressedOutput`.
    pub async fn decompress_chunked<R, F>(
        &self,
        reader: &mut R,
        mut on_chunk: F,
    ) -> Result<usize, CompressError>
    where
        R: AsyncRead + Unpin,
        F: FnMut(Vec<u8>) -> Result<(), CompressError>,
    {
        let mut total_bytes = 0usize;
        let mut len_buf = [0u8; 4];

        loop {
            match reader.read_exact(&mut len_buf).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(CompressError::IoError(e)),
            }

            let chunk_len = u32::from_le_bytes(len_buf) as usize;
            if chunk_len == 0 {
                break;
            }

            let mut chunk_buf = vec![0u8; chunk_len];
            reader
                .read_exact(&mut chunk_buf)
                .await
                .map_err(CompressError::IoError)?;

            let compressed: CompressedOutput = bincode::deserialize(&chunk_buf)
                .map_err(|e| CompressError::SerializationError(format!(
                    "Chunk deserialization failed: {e}"
                )))?;

            let decompressed = self.compressor.decompress(&compressed)?;
            total_bytes += decompressed.len();
            on_chunk(decompressed)?;
        }

        Ok(total_bytes)
    }

    /// Access the underlying compressor.
    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    /// Access the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Convenience function: download and decompress from a URL with default config.
pub async fn stream_download(url: &str) -> Result<Vec<u8>, CompressError> {
    let decompressor = StreamingDecompressor::new(Compressor::default());
    decompressor.stream_download(url).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 65_536);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_streaming_decompressor_creation() {
        let sd = StreamingDecompressor::new(Compressor::default());
        assert_eq!(sd.config().chunk_size, 65_536);
    }

    #[test]
    fn test_with_config() {
        let mut cfg = StreamingConfig::default();
        cfg.chunk_size = 1024;
        cfg.timeout_secs = 10;
        let sd = StreamingDecompressor::with_config(cfg);
        assert_eq!(sd.config().chunk_size, 1024);
        assert_eq!(sd.config().timeout_secs, 10);
    }

    #[tokio::test]
    async fn test_decompress_stream_empty() {
        let sd = StreamingDecompressor::new(Compressor::default());
        let mut cursor = tokio::io::BufReader::new(&b""[..]);
        let result = sd.decompress_stream(&mut cursor).await;
        assert!(result.is_err());
    }
}
