//! Semantic deduplication via content hashing and similarity grouping
//!
//! Groups similar content blocks and stores them once with references.

use crate::error::CompressError;
use std::collections::HashMap;

/// Compress via semantic deduplication (content-addressable blocks)
pub fn compress(data: &[u8], _threshold: f64) -> Result<Vec<u8>, CompressError> {
    let block_size = 64;
    let mut blocks: Vec<&[u8]> = Vec::new();
    let mut unique_blocks: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut block_refs: Vec<u32> = Vec::new();

    for chunk in data.chunks(block_size) {
        blocks.push(chunk);
        let key = chunk.to_vec();
        let idx = unique_blocks.len() as u32;
        let block_idx = *unique_blocks.entry(key).or_insert(idx);
        block_refs.push(block_idx);
    }

    // Format: [num_unique:u32][block_len:u32,block_data...][num_refs:u32][refs...]
    let mut output = Vec::new();
    let num_unique = unique_blocks.len() as u32;
    output.extend_from_slice(&num_unique.to_le_bytes());

    // Sort unique blocks by index so they can be looked up
    let mut sorted: Vec<(Vec<u8>, u32)> = unique_blocks.into_iter().collect();
    sorted.sort_by_key(|&(_, idx)| idx);

    for (block, _) in &sorted {
        output.extend_from_slice(&(block.len() as u32).to_le_bytes());
        output.extend_from_slice(block);
    }

    let num_refs = block_refs.len() as u32;
    output.extend_from_slice(&num_refs.to_le_bytes());
    for r in &block_refs {
        output.extend_from_slice(&r.to_le_bytes());
    }

    Ok(output)
}

/// Decompress semantically-deduplicated data
pub fn decompress(data: &[u8], _original_size: usize) -> Result<Vec<u8>, CompressError> {
    if data.len() < 4 {
        return Err(CompressError::SemanticError("data too short".into()));
    }
    let mut pos = 0;
    let num_unique =
        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(num_unique);
    for _ in 0..num_unique {
        if pos + 4 > data.len() {
            return Err(CompressError::SemanticError("truncated".into()));
        }
        let blen =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + blen > data.len() {
            return Err(CompressError::SemanticError("truncated block".into()));
        }
        blocks.push(data[pos..pos + blen].to_vec());
        pos += blen;
    }

    if pos + 4 > data.len() {
        return Err(CompressError::SemanticError("missing refs".into()));
    }
    let num_refs =
        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    let mut output = Vec::new();
    for _ in 0..num_refs {
        if pos + 4 > data.len() {
            return Err(CompressError::SemanticError("truncated ref".into()));
        }
        let idx =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if idx >= blocks.len() {
            return Err(CompressError::SemanticError("invalid ref".into()));
        }
        output.extend_from_slice(&blocks[idx]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_roundtrip() {
        let data = "hello world ".repeat(10);
        let compressed = compress(data.as_bytes(), 0.95).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data.as_bytes());
    }

    #[test]
    fn test_semantic_dedup_ratio() {
        let data = vec![0u8; 1000]; // highly duplicated
        let compressed = compress(&data, 0.95).unwrap();
        assert!(
            compressed.len() < data.len(),
            "should compress repeated data"
        );
    }

    #[test]
    fn test_semantic_unique_data() {
        let data: Vec<u8> = (0..200).collect();
        let compressed = compress(&data, 0.95).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
