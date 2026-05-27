//! Semantic deduplication via content hashing, MinHash LSH, and optional Ryzanstein embeddings.
//!
//! Provides both a low-level stateful deduplicator and the stateless compress/decompress
//! functions consumed by the top-level Compressor.

use crate::error::CompressError;
use crate::minhash::{LSHBuckets, MinHasher};
use std::collections::HashMap;

// ── Merge types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MergeStrategy {
    /// Store the block once; subsequent identical/near-identical blocks reference it.
    StoreOnce,
    /// Store base block + XOR delta for near-duplicates.
    Delta,
    /// Store embedding vector as a retrieval proxy (future use).
    Embedding,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MergeAction {
    /// Block was novel and stored.
    Stored,
    /// Block is a duplicate of the canonical block with this hash.
    DuplicateOf(u64),
    /// Block is a near-duplicate; stored as XOR delta of the canonical.
    DeltaOf(u64),
}

#[derive(Debug, Clone)]
pub struct MergeResult {
    pub action: MergeAction,
    pub canonical_hash: Option<u64>,
}

// ── SemanticDeduplicator ─────────────────────────────────────────────────────

/// Stateful deduplicator backed by MinHash LSH + optional Ryzanstein embeddings.
pub struct SemanticDeduplicator {
    hasher: MinHasher,
    lsh: LSHBuckets,
    /// canonical_hash → stored block bytes
    store: HashMap<u64, Vec<u8>>,
    /// canonical_hash → MinHash signature
    signatures: HashMap<u64, Vec<u64>>,
    /// Jaccard threshold above which we check for duplicates (default 0.85)
    threshold: f64,
    /// Optional Ryzanstein base URL; None means MinHash-only path.
    ryzanstein_url: Option<String>,
}

impl SemanticDeduplicator {
    pub fn new(threshold: f64, ryzanstein_url: Option<String>) -> Self {
        Self {
            hasher: MinHasher::default(),
            lsh: LSHBuckets::default(),
            store: HashMap::new(),
            signatures: HashMap::new(),
            threshold,
            ryzanstein_url,
        }
    }

    /// Decide whether to store a new block or reference an existing one.
    ///
    /// Algorithm:
    /// 1. Compute FNV hash of block — exact-duplicate fast path.
    /// 2. Compute MinHash signature; query LSH for candidates.
    /// 3. For each candidate: if Jaccard ≥ threshold, optionally upgrade with
    ///    cosine similarity via Ryzanstein (skipped if URL unset).
    /// 4. If no duplicate found: insert into store + LSH index.
    pub fn merge_or_store(&mut self, block: &[u8]) -> MergeResult {
        let block_hash = fnv_hash(block);

        // Fast path: exact duplicate.
        if self.store.contains_key(&block_hash) {
            return MergeResult {
                action: MergeAction::DuplicateOf(block_hash),
                canonical_hash: Some(block_hash),
            };
        }

        let sig = self.hasher.signature(block);
        let candidates = self.lsh.candidates(&sig);

        for cand_id in &candidates {
            if let Some(cand_sig) = self.signatures.get(cand_id) {
                let jaccard = MinHasher::similarity(&sig, cand_sig);
                if jaccard >= self.threshold {
                    // Ryzanstein upgrade path (graceful skip if unavailable).
                    let is_dup = if self.ryzanstein_url.is_some() {
                        self.cosine_check(block, *cand_id)
                    } else {
                        // MinHash-only: trust Jaccard ≥ threshold
                        true
                    };

                    if is_dup {
                        return MergeResult {
                            action: MergeAction::DuplicateOf(*cand_id),
                            canonical_hash: Some(*cand_id),
                        };
                    }
                }
            }
        }

        // Novel block: store it.
        self.store.insert(block_hash, block.to_vec());
        self.lsh.insert(block_hash, &sig);
        self.signatures.insert(block_hash, sig);

        MergeResult {
            action: MergeAction::Stored,
            canonical_hash: Some(block_hash),
        }
    }

    /// Retrieve a stored canonical block by hash.
    pub fn get(&self, hash: u64) -> Option<&[u8]> {
        self.store.get(&hash).map(|v| v.as_slice())
    }

    /// Cosine-similarity check via Ryzanstein fallback embeddings.
    /// Returns true if cosine ≥ 0.9. Never panics; returns false on any error.
    fn cosine_check(&self, block: &[u8], cand_id: u64) -> bool {
        let Some(cand_block) = self.store.get(&cand_id) else {
            return false;
        };
        // Use the same fallback embedder as ryzanstein_integration.rs
        // (avoids async; real HTTP calls happen in the async layer above).
        let emb_a = fallback_embed(block);
        let emb_b = fallback_embed(cand_block);
        cosine_similarity(&emb_a, &emb_b) >= 0.9
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fnv_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fallback_embed(data: &[u8]) -> Vec<f32> {
    let mut emb = vec![0.0f32; 128];
    for (i, &b) in data.iter().enumerate() {
        emb[i % 128] += (b as f32) / 255.0;
    }
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut emb {
            *v /= norm;
        }
    }
    emb
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if mag_a * mag_b < 1e-10 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

// ── Stateless compress / decompress (used by Compressor) ────────────────────

/// Compress via semantic deduplication (content-addressable blocks).
/// Format: [num_unique:u32][block_len:u32, block_data...][num_refs:u32][refs:u32...]
pub fn compress(data: &[u8], _threshold: f64) -> Result<Vec<u8>, CompressError> {
    let block_size = 64;
    let mut unique_blocks: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut block_refs: Vec<u32> = Vec::new();

    for chunk in data.chunks(block_size) {
        let key = chunk.to_vec();
        let idx = unique_blocks.len() as u32;
        let block_idx = *unique_blocks.entry(key).or_insert(idx);
        block_refs.push(block_idx);
    }

    let mut output = Vec::new();
    let num_unique = unique_blocks.len() as u32;
    output.extend_from_slice(&num_unique.to_le_bytes());

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

/// Decompress semantically-deduplicated data.
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

// ── Tests ────────────────────────────────────────────────────────────────────

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
        let data = vec![0u8; 1000];
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

    #[test]
    fn test_merge_or_store_exact_duplicate() {
        let mut dedup = SemanticDeduplicator::new(0.85, None);
        let block = b"fn main() { println!(\"hello\"); }";
        let r1 = dedup.merge_or_store(block);
        assert_eq!(r1.action, MergeAction::Stored);
        let r2 = dedup.merge_or_store(block);
        assert!(matches!(r2.action, MergeAction::DuplicateOf(_)));
    }

    #[test]
    fn test_merge_or_store_novel_blocks() {
        let mut dedup = SemanticDeduplicator::new(0.85, None);
        let a = b"completely different content alpha";
        let b = b"totally unrelated material beta";
        let r_a = dedup.merge_or_store(a);
        let r_b = dedup.merge_or_store(b);
        assert_eq!(r_a.action, MergeAction::Stored);
        assert_eq!(r_b.action, MergeAction::Stored);
    }

    #[test]
    fn test_merge_fallback_without_ryzanstein() {
        // Must not panic when ryzanstein_url is None
        let mut dedup = SemanticDeduplicator::new(0.85, None);
        let block = b"some block of data";
        let r = dedup.merge_or_store(block);
        assert_eq!(r.action, MergeAction::Stored);
        let r2 = dedup.merge_or_store(block);
        assert!(matches!(r2.action, MergeAction::DuplicateOf(_)));
    }

    #[test]
    fn test_get_stored_block() {
        let mut dedup = SemanticDeduplicator::new(0.85, None);
        let block = b"retrievable content here";
        let result = dedup.merge_or_store(block);
        let hash = result.canonical_hash.unwrap();
        assert_eq!(dedup.get(hash), Some(block.as_slice()));
    }
}
