use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sigma_compress::minhash::{LSHBuckets, MinHasher};
use sigma_compress::semantic::SemanticDeduplicator;
use sigma_compress::{CompressionMethod, Compressor};

// ── Realistic test data ───────────────────────────────────────────────────────

/// 1 MB of realistic Rust source snippets tiled to fill the buffer.
fn make_rust_text_1mb() -> Vec<u8> {
    let snippet = r#"
use std::collections::HashMap;

/// Compresses data using the selected method.
pub fn compress(data: &[u8], method: CompressionMethod) -> Result<Vec<u8>, CompressError> {
    match method {
        CompressionMethod::Huffman => huffman::compress(data),
        CompressionMethod::Lz4Semantic => lz4_wrapper::compress(data, 65536),
        CompressionMethod::EntropyCoding => entropy::compress(data),
        CompressionMethod::SemanticDedupe => semantic::compress(data, 0.95),
        CompressionMethod::Auto => {
            let entropy = compute_entropy(data);
            if entropy < 3.0 {
                huffman::compress(data)
            } else {
                lz4_wrapper::compress(data, 65536)
            }
        }
    }
}

fn compute_entropy(data: &[u8]) -> f64 {
    let mut freq = [0u64; 256];
    for &b in data { freq[b as usize] += 1; }
    let len = data.len() as f64;
    let mut entropy = 0.0_f64;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

pub struct BlockIndex {
    map: HashMap<u64, Vec<u8>>,
}

impl BlockIndex {
    pub fn new() -> Self { Self { map: HashMap::new() } }
    pub fn insert(&mut self, key: u64, val: Vec<u8>) { self.map.insert(key, val); }
    pub fn get(&self, key: u64) -> Option<&Vec<u8>> { self.map.get(&key) }
}
"#;
    let target = 1024 * 1024;
    let mut buf = Vec::with_capacity(target);
    while buf.len() < target {
        buf.extend_from_slice(snippet.as_bytes());
    }
    buf.truncate(target);
    buf
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_huffman_1mb(c: &mut Criterion) {
    let data = make_rust_text_1mb();
    let compressor = Compressor::default();
    c.bench_function("huffman_1mb", |b| {
        b.iter(|| {
            compressor
                .compress(black_box(&data), CompressionMethod::Huffman)
                .unwrap()
        })
    });
}

fn bench_lz4_1mb(c: &mut Criterion) {
    let data = make_rust_text_1mb();
    let compressor = Compressor::default();
    c.bench_function("lz4_1mb", |b| {
        b.iter(|| {
            compressor
                .compress(black_box(&data), CompressionMethod::Lz4Semantic)
                .unwrap()
        })
    });
}

fn bench_entropy_1mb(c: &mut Criterion) {
    let data = make_rust_text_1mb();
    let compressor = Compressor::default();
    c.bench_function("entropy_1mb", |b| {
        b.iter(|| {
            compressor
                .compress(black_box(&data), CompressionMethod::EntropyCoding)
                .unwrap()
        })
    });
}

fn bench_semantic_dedup_1k(c: &mut Criterion) {
    // 800 unique blocks + 200 near-duplicates (same content with minor tweak).
    let base: Vec<u8> = make_rust_text_1mb()[..512].to_vec();
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(1000);
    for i in 0u8..200 {
        let mut b = base.clone();
        // Unique: vary last byte.
        if let Some(last) = b.last_mut() {
            *last = i;
        }
        blocks.push(b);
    }
    for i in 0u8..200 {
        // Near-duplicate: change only one interior byte.
        let mut b = base.clone();
        b[10] = i;
        blocks.push(b);
    }
    // Fill remainder with fully unique random-ish blocks.
    for i in 0u64..600 {
        let mut b: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        b[0] = (i & 0xff) as u8;
        b[1] = ((i >> 8) & 0xff) as u8;
        blocks.push(b);
    }

    c.bench_function("semantic_dedup_1k", |b| {
        b.iter(|| {
            let mut dedup = SemanticDeduplicator::new(0.85, None);
            for block in black_box(&blocks) {
                dedup.merge_or_store(block);
            }
        })
    });
}

fn bench_minhash_10k(c: &mut Criterion) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Pre-generate 10k 128-byte docs.  128 bytes → 121 shingles × 128 hash functions
    // per document, keeping total work tractable while still exercising the full LSH path.
    let docs: Vec<Vec<u8>> = (0u64..10_000)
        .map(|i| {
            let mut doc = Vec::with_capacity(128);
            let mut state = i.wrapping_mul(0x9e3779b97f4a7c15);
            for _ in 0..16 {
                let mut h = DefaultHasher::new();
                state.hash(&mut h);
                state = h.finish();
                doc.extend_from_slice(&state.to_le_bytes());
            }
            doc
        })
        .collect();

    let query_docs: Vec<Vec<u8>> = docs[..100].to_vec();

    c.bench_function("minhash_10k_sign_index_query", |b| {
        b.iter(|| {
            let hasher = MinHasher::default();
            let mut lsh = LSHBuckets::default();
            for (i, doc) in black_box(&docs).iter().enumerate() {
                let sig = hasher.signature(doc);
                lsh.insert(i as u64, &sig);
            }
            for doc in &query_docs {
                let sig = hasher.signature(doc);
                let _ = lsh.candidates(&sig);
            }
        })
    });
}

fn bench_auto_select_1mb(c: &mut Criterion) {
    let data = make_rust_text_1mb();
    let compressor = Compressor::default();
    c.bench_function("auto_select_1mb", |b| {
        b.iter(|| {
            compressor
                .compress(black_box(&data), CompressionMethod::Auto)
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_huffman_1mb,
    bench_lz4_1mb,
    bench_entropy_1mb,
    bench_semantic_dedup_1k,
    bench_minhash_10k,
    bench_auto_select_1mb,
);
criterion_main!(benches);
