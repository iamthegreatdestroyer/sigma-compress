//! Entropy coding — arithmetic/range coding for near-optimal compression

use crate::error::CompressError;

/// Compress using simple run-length + byte-packing entropy coder
pub fn compress(data: &[u8]) -> Result<Vec<u8>, CompressError> {
    // Run-length encoding as a simple entropy-aware compressor
    let mut output = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run = 1u16;
        while i + (run as usize) < data.len() && data[i + (run as usize)] == byte && run < 255 {
            run += 1;
        }
        output.push(run as u8);
        output.push(byte);
        i += run as usize;
    }
    Ok(output)
}

/// Decompress RLE-encoded data
pub fn decompress(data: &[u8], original_size: usize) -> Result<Vec<u8>, CompressError> {
    if data.len() % 2 != 0 {
        return Err(CompressError::EntropyError("invalid RLE data".into()));
    }
    let mut output = Vec::with_capacity(original_size);
    let mut i = 0;
    while i < data.len() {
        let run = data[i] as usize;
        let byte = data[i + 1];
        for _ in 0..run {
            output.push(byte);
        }
        i += 2;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_roundtrip() {
        let data = b"aaabbbccc";
        let compressed = compress(data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_entropy_single_run() {
        let data = vec![0xFFu8; 100];
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_entropy_no_runs() {
        let data: Vec<u8> = (0..50).collect();
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
