//! LZ4 wrapper for block-level compression with semantic awareness

use crate::error::CompressError;

/// Compress data using LZ4-style block compression
pub fn compress(data: &[u8], block_size: usize) -> Result<Vec<u8>, CompressError> {
    // Simple LZ4-like compression: store block headers + compressed blocks
    let mut output = Vec::new();
    let num_blocks = (data.len() + block_size - 1) / block_size;
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    for chunk in data.chunks(block_size) {
        // Use flate2 for actual compression of each block
        let compressed = lz4_compress_block(chunk)?;
        output.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        output.extend_from_slice(&compressed);
    }

    Ok(output)
}

/// Decompress LZ4-compressed data
pub fn decompress(data: &[u8], original_size: usize) -> Result<Vec<u8>, CompressError> {
    if data.len() < 4 {
        return Err(CompressError::Lz4Error("data too short".into()));
    }

    let mut pos = 0;
    let num_blocks =
        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    let mut output = Vec::with_capacity(original_size);

    for _ in 0..num_blocks {
        if pos + 8 > data.len() {
            return Err(CompressError::Lz4Error("truncated block header".into()));
        }
        let _orig_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let comp_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + comp_len > data.len() {
            return Err(CompressError::Lz4Error("truncated block data".into()));
        }
        let block = lz4_decompress_block(&data[pos..pos + comp_len])?;
        output.extend_from_slice(&block);
        pos += comp_len;
    }

    Ok(output)
}

fn lz4_compress_block(data: &[u8]) -> Result<Vec<u8>, CompressError> {
    use std::io::Write;
    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder
        .write_all(data)
        .map_err(|e| CompressError::Lz4Error(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| CompressError::Lz4Error(e.to_string()))
}

fn lz4_decompress_block(data: &[u8]) -> Result<Vec<u8>, CompressError> {
    use std::io::Read;
    let mut decoder = flate2::read::DeflateDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| CompressError::Lz4Error(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let data = b"test data for lz4 compression roundtrip test data";
        let compressed = compress(data, 1024).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_multiple_blocks() {
        let data = vec![42u8; 200];
        let compressed = compress(&data, 64).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_small_data() {
        let data = b"hi";
        let compressed = compress(data, 1024).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
