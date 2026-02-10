//! sigma-compress: Semantic-aware compression engine for the Ryzanstein ecosystem.
//!
//! Combines multiple compression strategies:
//! - Huffman coding for symbol-level compression
//! - LZ4 for fast block compression
//! - Entropy coding for near-optimal bitstream output
//! - Semantic deduplication via Ryzanstein embeddings
//!
//! Chooses the optimal strategy based on content analysis.

pub mod config;
pub mod error;
pub mod huffman;
pub mod lz4_wrapper;
pub mod entropy;
pub mod semantic;
pub mod ryzanstein_integration;

use crate::config::CompressionConfig;
use crate::error::CompressError;

/// Compression method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompressionMethod {
    Huffman,
    Lz4Semantic,
    EntropyCoding,
    SemanticDedupe,
    Auto,
}

/// Compressed output container
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressedOutput {
    pub method: CompressionMethod,
    pub original_size: usize,
    pub compressed_size: usize,
    pub data: Vec<u8>,
    pub ratio: f64,
    pub metadata: CompressionMetadata,
}

/// Metadata about the compression process
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionMetadata {
    pub entropy_bits: f64,
    pub semantic_dedup_count: usize,
    pub block_count: usize,
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_compressed: usize,
    pub total_decompressed: usize,
    pub avg_ratio: f64,
    pub best_method_counts: std::collections::HashMap<String, usize>,
}

/// The main compressor engine
pub struct Compressor {
    config: CompressionConfig,
}

impl Compressor {
    /// Create a new compressor with the given configuration
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Create a compressor with default configuration
    pub fn default() -> Self {
        Self::new(CompressionConfig::default())
    }

    /// Compress data using the specified method
    pub fn compress(&self, data: &[u8], method: CompressionMethod) -> Result<CompressedOutput, CompressError> {
        if data.is_empty() {
            return Err(CompressError::EmptyInput);
        }

        let method = if method == CompressionMethod::Auto {
            self.select_method(data)
        } else {
            method
        };

        let compressed = match method {
            CompressionMethod::Huffman => huffman::compress(data)?,
            CompressionMethod::Lz4Semantic => lz4_wrapper::compress(data, self.config.lz4_block_size)?,
            CompressionMethod::EntropyCoding => entropy::compress(data)?,
            CompressionMethod::SemanticDedupe => semantic::compress(data, self.config.dedup_threshold)?,
            CompressionMethod::Auto => unreachable!(),
        };

        let ratio = if data.is_empty() {
            1.0
        } else {
            compressed.len() as f64 / data.len() as f64
        };

        Ok(CompressedOutput {
            method,
            original_size: data.len(),
            compressed_size: compressed.len(),
            data: compressed,
            ratio,
            metadata: CompressionMetadata {
                entropy_bits: self.compute_entropy(data),
                semantic_dedup_count: 0,
                block_count: (data.len() / self.config.lz4_block_size).max(1),
            },
        })
    }

    /// Decompress data
    pub fn decompress(&self, output: &CompressedOutput) -> Result<Vec<u8>, CompressError> {
        match output.method {
            CompressionMethod::Huffman => huffman::decompress(&output.data, output.original_size),
            CompressionMethod::Lz4Semantic => lz4_wrapper::decompress(&output.data, output.original_size),
            CompressionMethod::EntropyCoding => entropy::decompress(&output.data, output.original_size),
            CompressionMethod::SemanticDedupe => semantic::decompress(&output.data, output.original_size),
            CompressionMethod::Auto => Err(CompressError::InvalidMethod),
        }
    }

    /// Automatically select the best compression method based on data analysis
    fn select_method(&self, data: &[u8]) -> CompressionMethod {
        let entropy = self.compute_entropy(data);
        if entropy < 3.0 {
            CompressionMethod::Huffman
        } else if data.len() > 4096 {
            CompressionMethod::Lz4Semantic
        } else {
            CompressionMethod::EntropyCoding
        }
    }

    /// Compute Shannon entropy of data in bits per byte
    fn compute_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut freq = [0u64; 256];
        for &b in data {
            freq[b as usize] += 1;
        }
        let len = data.len() as f64;
        let mut entropy = 0.0;
        for &f in &freq {
            if f > 0 {
                let p = f as f64 / len;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_huffman() {
        let compressor = Compressor::default();
        let data = b"hello world hello world hello world";
        let result = compressor.compress(data, CompressionMethod::Huffman).unwrap();
        assert!(result.compressed_size > 0);
        assert_eq!(result.original_size, data.len());
        assert_eq!(result.method, CompressionMethod::Huffman);
    }

    #[test]
    fn test_compress_lz4() {
        let compressor = Compressor::default();
        let data = b"repeated repeated repeated repeated";
        let result = compressor.compress(data, CompressionMethod::Lz4Semantic).unwrap();
        assert!(result.compressed_size > 0);
    }

    #[test]
    fn test_compress_empty() {
        let compressor = Compressor::default();
        let result = compressor.compress(b"", CompressionMethod::Huffman);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_huffman() {
        let compressor = Compressor::default();
        let data = b"the quick brown fox jumps over the lazy dog";
        let compressed = compressor.compress(data, CompressionMethod::Huffman).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_auto_selection() {
        let compressor = Compressor::default();
        let low_entropy = vec![0u8; 1000];
        let result = compressor.compress(&low_entropy, CompressionMethod::Auto).unwrap();
        assert_eq!(result.method, CompressionMethod::Huffman);
    }

    #[test]
    fn test_entropy_computation() {
        let compressor = Compressor::default();
        let uniform = vec![42u8; 100];
        let entropy = compressor.compute_entropy(&uniform);
        assert!(entropy < 0.01, "uniform data should have ~0 entropy");
    }

    #[test]
    fn test_compression_ratio() {
        let compressor = Compressor::default();
        let data = "aaaaaaaaaa".repeat(100);
        let result = compressor.compress(data.as_bytes(), CompressionMethod::Huffman).unwrap();
        assert!(result.ratio < 1.0, "repetitive data should compress well");
    }
}
