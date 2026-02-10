//! Huffman compression and decompression
//!
//! Implements classic Huffman coding for symbol-level compression.

use crate::error::CompressError;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
struct HuffNode {
    freq: u64,
    symbol: Option<u8>,
    left: Option<Box<HuffNode>>,
    right: Option<Box<HuffNode>>,
}

impl Eq for HuffNode {}
impl PartialEq for HuffNode {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq
    }
}
impl PartialOrd for HuffNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HuffNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.freq.cmp(&self.freq) // min-heap
    }
}

fn build_tree(data: &[u8]) -> Option<HuffNode> {
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }

    let mut heap = BinaryHeap::new();
    for (i, &f) in freq.iter().enumerate() {
        if f > 0 {
            heap.push(HuffNode {
                freq: f,
                symbol: Some(i as u8),
                left: None,
                right: None,
            });
        }
    }

    if heap.is_empty() {
        return None;
    }
    if heap.len() == 1 {
        let node = heap.pop().unwrap();
        return Some(HuffNode {
            freq: node.freq,
            symbol: None,
            left: Some(Box::new(node)),
            right: Some(Box::new(HuffNode {
                freq: 0,
                symbol: None,
                left: None,
                right: None,
            })),
        });
    }

    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();
        heap.push(HuffNode {
            freq: left.freq + right.freq,
            symbol: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        });
    }

    heap.pop()
}

fn build_codes(node: &HuffNode, prefix: Vec<bool>, codes: &mut HashMap<u8, Vec<bool>>) {
    if let Some(sym) = node.symbol {
        let code = if prefix.is_empty() { vec![false] } else { prefix };
        codes.insert(sym, code);
        return;
    }
    if let Some(ref left) = node.left {
        let mut p = prefix.clone();
        p.push(false);
        build_codes(left, p, codes);
    }
    if let Some(ref right) = node.right {
        let mut p = prefix.clone();
        p.push(true);
        build_codes(right, p, codes);
    }
}

/// Compress data using Huffman coding
pub fn compress(data: &[u8]) -> Result<Vec<u8>, CompressError> {
    let tree = build_tree(data).ok_or_else(|| CompressError::HuffmanError("empty tree".into()))?;
    let mut codes = HashMap::new();
    build_codes(&tree, vec![], &mut codes);

    // Encode: [num_symbols:u16][symbol:u8,code_len:u8,code_bits...][data_bits...]
    let mut output = Vec::new();
    let num_symbols = codes.len() as u16;
    output.extend_from_slice(&num_symbols.to_le_bytes());

    // Write code table
    for (&sym, code) in &codes {
        output.push(sym);
        output.push(code.len() as u8);
        let mut byte = 0u8;
        let mut bit_pos = 0;
        for &bit in code {
            if bit {
                byte |= 1 << bit_pos;
            }
            bit_pos += 1;
            if bit_pos == 8 {
                output.push(byte);
                byte = 0;
                bit_pos = 0;
            }
        }
        if bit_pos > 0 {
            output.push(byte);
        }
    }

    // Write data length
    let data_len = data.len() as u32;
    output.extend_from_slice(&data_len.to_le_bytes());

    // Encode data
    let mut bits = Vec::new();
    for &b in data {
        if let Some(code) = codes.get(&b) {
            bits.extend_from_slice(code);
        }
    }

    // Pack bits into bytes
    let mut byte = 0u8;
    let mut bit_pos = 0;
    for &bit in &bits {
        if bit {
            byte |= 1 << bit_pos;
        }
        bit_pos += 1;
        if bit_pos == 8 {
            output.push(byte);
            byte = 0;
            bit_pos = 0;
        }
    }
    if bit_pos > 0 {
        output.push(byte);
    }

    Ok(output)
}

/// Decompress Huffman-encoded data
pub fn decompress(data: &[u8], original_size: usize) -> Result<Vec<u8>, CompressError> {
    if data.len() < 2 {
        return Err(CompressError::HuffmanError("data too short".into()));
    }

    let mut pos = 0;
    let num_symbols = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
    pos += 2;

    // Read code table
    let mut code_to_symbol: HashMap<Vec<bool>, u8> = HashMap::new();
    for _ in 0..num_symbols {
        if pos >= data.len() {
            return Err(CompressError::HuffmanError("truncated table".into()));
        }
        let sym = data[pos];
        pos += 1;
        let code_len = data[pos] as usize;
        pos += 1;

        let num_bytes = (code_len + 7) / 8;
        let mut code = Vec::with_capacity(code_len);
        for byte_idx in 0..num_bytes {
            if pos >= data.len() {
                return Err(CompressError::HuffmanError("truncated code".into()));
            }
            let byte = data[pos];
            pos += 1;
            for bit_idx in 0..8 {
                if byte_idx * 8 + bit_idx >= code_len {
                    break;
                }
                code.push((byte >> bit_idx) & 1 == 1);
            }
        }
        code_to_symbol.insert(code, sym);
    }

    // Read original data length
    if pos + 4 > data.len() {
        return Err(CompressError::HuffmanError("missing data length".into()));
    }
    let _stored_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    // Decode bits
    let mut output = Vec::with_capacity(original_size);
    let mut current_code = Vec::new();

    'outer: for &byte in &data[pos..] {
        for bit_idx in 0..8 {
            current_code.push((byte >> bit_idx) & 1 == 1);
            if let Some(&sym) = code_to_symbol.get(&current_code) {
                output.push(sym);
                current_code.clear();
                if output.len() >= original_size {
                    break 'outer;
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_roundtrip() {
        let data = b"hello world hello world hello";
        let compressed = compress(data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_huffman_single_char() {
        let data = b"aaaaaa";
        let compressed = compress(data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_huffman_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_huffman_compression_ratio() {
        let data = "aaabbbccc".repeat(100);
        let compressed = compress(data.as_bytes()).unwrap();
        assert!(compressed.len() < data.len());
    }
}
