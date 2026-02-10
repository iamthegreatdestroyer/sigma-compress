//! Integration tests for sigma-compress

use sigma_compress::*;

#[test]
fn test_full_lifecycle() {
    let compressor = Compressor::default();
    let data = b"the quick brown fox jumps over the lazy dog".repeat(50);
    let compressed = compressor.compress(&data, CompressionMethod::Auto).unwrap();
    assert!(compressed.compressed_size > 0);
    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_all_methods_roundtrip() {
    let compressor = Compressor::default();
    let data = b"test data for all compression methods roundtrip";

    for method in [
        CompressionMethod::Huffman,
        CompressionMethod::Lz4Semantic,
        CompressionMethod::EntropyCoding,
        CompressionMethod::SemanticDedupe,
    ] {
        let compressed = compressor.compress(data, method).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data, "roundtrip failed for {:?}", method);
    }
}

#[test]
fn test_large_data() {
    let compressor = Compressor::default();
    let data = vec![0xABu8; 100_000];
    let compressed = compressor.compress(&data, CompressionMethod::Auto).unwrap();
    assert!(
        compressed.ratio < 0.5,
        "large uniform data should compress well"
    );
    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_binary_data() {
    let compressor = Compressor::default();
    let data: Vec<u8> = (0..=255).cycle().take(2000).collect();
    let compressed = compressor
        .compress(&data, CompressionMethod::Lz4Semantic)
        .unwrap();
    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_metadata_populated() {
    let compressor = Compressor::default();
    let data = b"metadata test data here";
    let compressed = compressor
        .compress(data, CompressionMethod::Huffman)
        .unwrap();
    assert!(compressed.metadata.entropy_bits > 0.0);
    assert!(compressed.metadata.block_count >= 1);
}

#[test]
fn test_compression_config() {
    use sigma_compress::config::CompressionConfig;
    let config = CompressionConfig {
        lz4_block_size: 1024,
        dedup_threshold: 0.9,
        ..CompressionConfig::default()
    };
    let compressor = Compressor::new(config);
    let data = b"config test data with custom block size";
    let result = compressor
        .compress(data, CompressionMethod::Lz4Semantic)
        .unwrap();
    assert!(result.compressed_size > 0);
}

#[test]
fn test_empty_input_error() {
    let compressor = Compressor::default();
    let result = compressor.compress(b"", CompressionMethod::Huffman);
    assert!(result.is_err());
}
