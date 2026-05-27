//! External integration tests for MinHash LSH.

use sigma_compress::minhash::{LSHBuckets, MinHasher};
use std::time::Instant;

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
    let hasher = MinHasher::new(128);
    // Two nearly identical Rust functions — single identifier changed.
    let a = b"fn compress_block(data: &[u8], level: u32) -> Vec<u8> { \
               let mut output = Vec::with_capacity(data.len()); \
               for chunk in data.chunks(64) { output.extend_from_slice(chunk); } output }";
    let b = b"fn compress_block(data: &[u8], level: u32) -> Vec<u8> { \
               let mut result = Vec::with_capacity(data.len()); \
               for chunk in data.chunks(64) { result.extend_from_slice(chunk); } result }";
    let sim = MinHasher::similarity(&hasher.signature(a), &hasher.signature(b));
    assert!(
        sim > 0.5,
        "nearly identical code should have similarity > 0.5, got {sim}"
    );
}

#[test]
fn test_unrelated_texts() {
    let hasher = MinHasher::default();
    let a = b"hello world";
    let b = b"fn main() { \
               let x: Vec<u8> = (0u8..=255).collect(); \
               println!(\"{}\", x.len()); \
               let y = x.iter().map(|v| v.wrapping_add(1)).collect::<Vec<_>>(); \
               drop(y); }";
    let sim = MinHasher::similarity(&hasher.signature(a), &hasher.signature(b));
    assert!(
        sim < 0.4,
        "unrelated texts should have low similarity, got {sim}"
    );
}

#[test]
fn test_minhash_performance() {
    // Sign 10,000 documents and insert into LSH — must complete in < 1s (release)
    // or < 30s (debug, no optimisation).
    // Doc size is 128 bytes: enough shingles to be representative, fast in debug mode.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let hasher = MinHasher::default();
    let mut lsh = LSHBuckets::default();

    let start = Instant::now();

    for i in 0u64..10_000 {
        // Deterministic pseudo-random 128-byte block.
        let mut doc = Vec::with_capacity(128);
        let mut state = i.wrapping_mul(0x9e3779b97f4a7c15);
        for _ in 0..16 {
            let mut h = DefaultHasher::new();
            state.hash(&mut h);
            state = h.finish();
            doc.extend_from_slice(&state.to_le_bytes());
        }
        let sig = hasher.signature(&doc);
        lsh.insert(i, &sig);
    }

    let elapsed = start.elapsed();
    // Release builds must hit < 1s; debug builds get a generous 30s budget
    // because unoptimised Rust is ~30× slower.
    let budget = if cfg!(debug_assertions) { 30.0 } else { 1.0 };
    assert!(
        elapsed.as_secs_f64() < budget,
        "10k document index+sign took {elapsed:?}, must be < {budget}s"
    );

    // Sanity: query a known doc and confirm it's its own candidate.
    let known_doc: Vec<u8> = (0u8..=255).cycle().take(128).collect();
    let known_sig = hasher.signature(&known_doc);
    lsh.insert(99_999, &known_sig);
    let cands = lsh.candidates(&known_sig);
    assert!(cands.contains(&99_999));
}
