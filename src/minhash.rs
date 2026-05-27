//! MinHash LSH for O(1) near-duplicate detection at scale.
//!
//! Uses k independent hash functions over 8-byte shingles to produce
//! a MinHash signature, then bands those signatures into LSH buckets
//! for sublinear candidate lookup.

use std::collections::{HashMap, HashSet};

const DEFAULT_NUM_HASHES: usize = 128;
const DEFAULT_BANDS: usize = 16;
const DEFAULT_ROWS: usize = 8;

/// FNV-1a inspired mixing with a per-function seed.
#[inline]
fn hash_with_seed(value: u64, seed: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    let bytes = value.to_le_bytes();
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Produces MinHash signatures from byte data via 8-byte shingles.
pub struct MinHasher {
    num_hashes: usize,
}

impl MinHasher {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "num_hashes must be > 0");
        Self { num_hashes: k }
    }

    /// Build a MinHash signature over 8-byte overlapping shingles.
    /// Returns a Vec<u64> of length `num_hashes`.
    pub fn signature(&self, data: &[u8]) -> Vec<u64> {
        let mut sig = vec![u64::MAX; self.num_hashes];

        if data.is_empty() {
            return sig;
        }

        let shingle_len = 8.min(data.len());
        let window_count = if data.len() >= shingle_len {
            data.len() - shingle_len + 1
        } else {
            1
        };

        for start in 0..window_count {
            let slice = &data[start..start + shingle_len];
            // Pack up to 8 bytes into a u64.
            let mut word = 0u64;
            for (i, &b) in slice.iter().enumerate() {
                word |= (b as u64) << (i * 8);
            }

            for (i, slot) in sig.iter_mut().enumerate() {
                let h = hash_with_seed(word, i as u64);
                if h < *slot {
                    *slot = h;
                }
            }
        }

        sig
    }

    /// Jaccard similarity estimate: fraction of equal positions.
    pub fn similarity(a: &[u64], b: &[u64]) -> f64 {
        if a.is_empty() || a.len() != b.len() {
            return 0.0;
        }
        let matches = a.iter().zip(b).filter(|(x, y)| x == y).count();
        matches as f64 / a.len() as f64
    }
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new(DEFAULT_NUM_HASHES)
    }
}

/// Band + row LSH index for sublinear near-duplicate candidate retrieval.
///
/// A signature of length `bands * rows` is split into `bands` sub-vectors
/// of `rows` elements each. Each sub-vector is hashed into a per-band bucket.
/// Two signatures sharing any bucket are returned as candidates.
pub struct LSHBuckets {
    bands: usize,
    rows: usize,
    /// bucket_map[band][bucket_hash] = list of ids
    bucket_map: Vec<HashMap<u64, Vec<u64>>>,
}

impl LSHBuckets {
    pub fn new(bands: usize, rows: usize) -> Self {
        assert!(bands > 0 && rows > 0);
        Self {
            bands,
            rows,
            bucket_map: vec![HashMap::new(); bands],
        }
    }

    fn band_hash(&self, band_slice: &[u64]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &v in band_slice {
            let bytes = v.to_le_bytes();
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// Insert a document with the given `id` and `signature`.
    /// `signature` must have at least `bands * rows` elements;
    /// extra elements are ignored.
    pub fn insert(&mut self, id: u64, signature: &[u64]) {
        for b in 0..self.bands {
            let start = b * self.rows;
            let end = (start + self.rows).min(signature.len());
            if start >= signature.len() {
                break;
            }
            let h = self.band_hash(&signature[start..end]);
            self.bucket_map[b].entry(h).or_default().push(id);
        }
    }

    /// Return all candidate ids that share at least one band bucket with `signature`.
    pub fn candidates(&self, signature: &[u64]) -> HashSet<u64> {
        let mut result = HashSet::new();
        for b in 0..self.bands {
            let start = b * self.rows;
            let end = (start + self.rows).min(signature.len());
            if start >= signature.len() {
                break;
            }
            let h = self.band_hash(&signature[start..end]);
            if let Some(ids) = self.bucket_map[b].get(&h) {
                result.extend(ids.iter().copied());
            }
        }
        result
    }
}

impl Default for LSHBuckets {
    fn default() -> Self {
        Self::new(DEFAULT_BANDS, DEFAULT_ROWS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_documents() {
        let hasher = MinHasher::default();
        let data = b"the quick brown fox jumps over the lazy dog";
        let sig_a = hasher.signature(data);
        let sig_b = hasher.signature(data);
        let sim = MinHasher::similarity(&sig_a, &sig_b);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical docs must have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_similar_texts() {
        let hasher = MinHasher::default();
        let a = b"fn compress(data: &[u8]) -> Vec<u8> { let mut out = Vec::new(); out }";
        let b = b"fn compress(data: &[u8]) -> Vec<u8> { let mut buf = Vec::new(); buf }";
        let sim = MinHasher::similarity(&hasher.signature(a), &hasher.signature(b));
        assert!(
            sim > 0.5,
            "similar texts should have similarity > 0.5, got {sim}"
        );
    }

    #[test]
    fn test_unrelated_texts() {
        let hasher = MinHasher::default();
        let a = b"hello world";
        let b = b"fn main() { let x: Vec<u8> = (0..255).collect(); println!(\"{:?}\", x); }";
        let sim = MinHasher::similarity(&hasher.signature(a), &hasher.signature(b));
        assert!(
            sim < 0.4,
            "unrelated texts should have low similarity, got {sim}"
        );
    }

    #[test]
    fn test_lsh_candidates_identical() {
        let hasher = MinHasher::new(128);
        let mut lsh = LSHBuckets::new(16, 8);
        let data = b"some document content for testing";
        let sig = hasher.signature(data);
        lsh.insert(42, &sig);
        let cands = lsh.candidates(&sig);
        assert!(
            cands.contains(&42),
            "identical signature must be its own candidate"
        );
    }

    #[test]
    fn test_lsh_no_false_negatives_for_identical() {
        let hasher = MinHasher::default();
        let mut lsh = LSHBuckets::default();
        let data = b"repeated block content";
        let sig = hasher.signature(data);
        lsh.insert(1, &sig);
        // A second copy with same signature must find candidate 1.
        let query_sig = hasher.signature(data);
        let cands = lsh.candidates(&query_sig);
        assert!(cands.contains(&1));
    }
}
