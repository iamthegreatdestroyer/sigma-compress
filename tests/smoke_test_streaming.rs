//! Smoke tests for the streaming decompression module.
//!
//! These tests verify the streaming API surface: config defaults,
//! decompressor construction, stream decompression error paths,
//! and chunked decompression with length-prefixed framing.

use sigma_compress::streaming::{StreamingConfig, StreamingDecompressor};
use sigma_compress::*;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::AsyncRead;

// ---------------------------------------------------------------------------
// Sync tests
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_config_defaults() {
    let config = StreamingConfig::default();
    assert_eq!(config.chunk_size, 65536);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_decompressor_new_uses_defaults() {
    let compressor = Compressor::default();
    let decompressor = StreamingDecompressor::new(compressor);
    let config = decompressor.config();
    assert_eq!(config.chunk_size, 65536);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_decompressor_with_config_custom_values() {
    let config = StreamingConfig {
        chunk_size: 1024,
        timeout_secs: 10,
        max_retries: 5,
        compression: sigma_compress::config::CompressionConfig::default(),
    };
    let decompressor = StreamingDecompressor::with_config(config);
    let cfg = decompressor.config();
    assert_eq!(cfg.chunk_size, 1024);
    assert_eq!(cfg.timeout_secs, 10);
    assert_eq!(cfg.max_retries, 5);
}

#[test]
fn test_decompressor_compressor_accessor() {
    let compressor = Compressor::default();
    let decompressor = StreamingDecompressor::new(compressor);
    // Accessor should return a reference without panicking.
    let _comp = decompressor.compressor();
}

// ---------------------------------------------------------------------------
// Async tests
// ---------------------------------------------------------------------------

/// A helper AsyncRead that yields a fixed byte slice then EOF.
struct MockReader {
    data: Vec<u8>,
    pos: usize,
}

impl MockReader {
    fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }
}

impl AsyncRead for MockReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let remaining = &self.data[self.pos..];
        if remaining.is_empty() {
            return Poll::Ready(Ok(()));
        }
        let to_copy = remaining.len().min(buf.remaining());
        buf.put_slice(&remaining[..to_copy]);
        self.pos += to_copy;
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn test_decompress_stream_empty_returns_error() {
    let decompressor = StreamingDecompressor::new(Compressor::default());
    let mut reader = MockReader::empty();
    let result = decompressor.decompress_stream(&mut reader).await;
    assert!(result.is_err(), "Empty input should produce an error");
}

#[tokio::test]
async fn test_decompress_stream_invalid_data_returns_error() {
    let decompressor = StreamingDecompressor::new(Compressor::default());
    // Feed random bytes that are not valid bincode-serialized CompressedOutput.
    let mut reader = MockReader::new(vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]);
    let result = decompressor.decompress_stream(&mut reader).await;
    assert!(
        result.is_err(),
        "Invalid data should produce a deserialization error"
    );
}

#[tokio::test]
async fn test_decompress_chunked_single_chunk() {
    // 1. Compress real data
    let compressor = Compressor::default();
    let original = b"Hello streaming world! This is a chunked decompression test.";
    let compressed = compressor
        .compress(original, CompressionMethod::DeflateSemantic)
        .expect("compress should succeed");

    // 2. Serialize to bincode
    let serialized = bincode::serialize(&compressed).expect("serialize should succeed");

    // 3. Build length-prefixed frame: [4-byte LE length][serialized data][4 zero bytes sentinel]
    let len = serialized.len() as u32;
    let mut wire = Vec::new();
    wire.extend_from_slice(&len.to_le_bytes());
    wire.extend_from_slice(&serialized);
    wire.extend_from_slice(&0u32.to_le_bytes()); // sentinel

    // 4. Feed to decompress_chunked
    let decompressor = StreamingDecompressor::new(Compressor::default());
    let mut reader = MockReader::new(wire);
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let total_bytes = decompressor
        .decompress_chunked(&mut reader, |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .await
        .expect("decompress_chunked should succeed");

    // 5. Verify
    assert_eq!(chunks.len(), 1, "Should have exactly one chunk");
    assert_eq!(chunks[0].as_slice(), original);
    assert_eq!(total_bytes, original.len());
}

#[tokio::test]
async fn test_decompress_chunked_multiple_chunks() {
    let compressor = Compressor::default();

    // Compress two distinct payloads
    let data_a = b"First chunk payload for multi-chunk test";
    let data_b = b"Second chunk payload -- verifying multi-frame";
    let compressed_a = compressor
        .compress(data_a, CompressionMethod::DeflateSemantic)
        .expect("compress a");
    let compressed_b = compressor
        .compress(data_b, CompressionMethod::DeflateSemantic)
        .expect("compress b");

    let ser_a = bincode::serialize(&compressed_a).expect("serialize a");
    let ser_b = bincode::serialize(&compressed_b).expect("serialize b");

    // Build wire: [len_a][ser_a][len_b][ser_b][0000 sentinel]
    let mut wire = Vec::new();
    wire.extend_from_slice(&(ser_a.len() as u32).to_le_bytes());
    wire.extend_from_slice(&ser_a);
    wire.extend_from_slice(&(ser_b.len() as u32).to_le_bytes());
    wire.extend_from_slice(&ser_b);
    wire.extend_from_slice(&0u32.to_le_bytes());

    let decompressor = StreamingDecompressor::new(Compressor::default());
    let mut reader = MockReader::new(wire);
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let total_bytes = decompressor
        .decompress_chunked(&mut reader, |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .await
        .expect("decompress_chunked multi should succeed");

    assert_eq!(chunks.len(), 2, "Should have exactly two chunks");
    assert_eq!(chunks[0].as_slice(), data_a);
    assert_eq!(chunks[1].as_slice(), data_b);
    assert_eq!(total_bytes, data_a.len() + data_b.len());
}

#[tokio::test]
async fn test_decompress_chunked_empty_sentinel_only() {
    // Wire contains only the zero-length sentinel → zero chunks, zero bytes.
    let wire = 0u32.to_le_bytes().to_vec();
    let decompressor = StreamingDecompressor::new(Compressor::default());
    let mut reader = MockReader::new(wire);
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let total_bytes = decompressor
        .decompress_chunked(&mut reader, |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .await
        .expect("sentinel-only should succeed");

    assert!(
        chunks.is_empty(),
        "No chunks expected for sentinel-only stream"
    );
    assert_eq!(total_bytes, 0);
}
