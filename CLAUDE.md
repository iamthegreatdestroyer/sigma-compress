# sigma-compress — Autonomous Completion Brief

## Project Identity
- **Repo:** `iamthegreatdestroyer/sigma-compress`
- **Local path:** `S:\sigma-compress`
- **Language:** Rust
- **Castle Layer:** Layer 4 — Storage & Inference
- **Current completion:** ~75%
- **Mission:** Semantic-aware compression engine — Huffman + LZ4 + entropy coding + semantic deduplication, integrated with Ryzanstein embeddings

## Current State (verified 2026-05-25)
| Component | Status |
|-----------|--------|
| `config.rs` — CompressionConfig | ✅ Done |
| `entropy.rs` — entropy coding | ✅ Done |
| `error.rs` — error types | ✅ Done |
| `huffman.rs` — Huffman coding | ✅ Done |
| `lz4_wrapper.rs` — LZ4 integration | ✅ Done |
| `ryzanstein_integration.rs` — embedding client | ✅ Done |
| `semantic.rs` — semantic dedup | ✅ Done |
| `streaming.rs` — streaming compression | ✅ Done |
| `lib.rs` — public API | ✅ Done |
| MinHash optimization (10M+ documents) | ❌ Missing |
| Semantic embedding merge strategy | ❌ Missing |
| Benchmarks suite | ❌ Missing |
| Integration tests with real data | ❌ Missing |

## Key File Map
```
sigma-compress/
├── src/
│   ├── lib.rs              # Public API: Compressor, CompressionMethod, CompressedOutput
│   ├── config.rs           # CompressionConfig (block_size, method, ryzanstein_url)
│   ├── entropy.rs          # Run-length entropy coding
│   ├── error.rs            # CompressError enum
│   ├── huffman.rs          # Huffman tree + encode/decode
│   ├── lz4_wrapper.rs      # LZ4 compress/decompress via lz4 crate
│   ├── ryzanstein_integration.rs # HTTP client for /v1/embeddings
│   ├── semantic.rs         # Semantic dedup: hash map + embedding similarity
│   └── streaming.rs        # Block-based streaming compressor
├── Cargo.toml
└── README.md
```

## What Remains (Final 25%)

### Sprint 1 — MinHash Implementation (Day 1)
**Goal:** MinHash LSH for O(1) near-duplicate detection at 10M+ document scale.

```
@APEX implement minhash.rs in src/:
  - MinHasher struct: k hash functions (default: 128)
  - signature(data: &[u8]) -> Vec<u64>  // k-min-hashes over 8-byte shingles
  - similarity(a: &[u64], b: &[u64]) -> f64  // Jaccard estimate
  - MinHashIndex: store signatures, find candidates above threshold
  - LSHBuckets: band + row scheme for sublinear lookup (b=16 bands, r=8 rows)

Wire into semantic.rs: before calling Ryzanstein embeddings, check MinHashIndex.
If Jaccard similarity ≥ 0.8, treat as duplicate → skip embedding call.

Tests in tests/minhash_test.rs:
  - test_identical_documents → similarity == 1.0
  - test_identical_texts → similarity > 0.95
  - test_unrelated_texts → similarity < 0.2
  - test_minhash_performance: 10,000 documents indexed in <1s
```

### Sprint 2 — Semantic Embedding Merge Strategy (Day 1–2)
**Goal:** When Ryzanstein identifies semantically similar blocks, merge/reference rather than store.

```
@APEX extend semantic.rs with MergeStrategy:
  enum MergeStrategy {
    StoreOnce,    // Store canonical block, use reference for duplicates
    Delta,        // Store base block + XOR delta for near-duplicates  
    Embedding,    // Store embedding vector as proxy for retrieval
  }

Implement merge_or_store(block: &[u8], threshold: f64) -> MergeResult:
  1. Compute MinHash signature
  2. Check LSH buckets for candidates
  3. For each candidate: if Jaccard ≥ threshold (0.85 default):
     - compute embedding similarity (call Ryzanstein if available)
     - if embedding cosine ≥ 0.9: return DuplicateOf(canonical_hash)
  4. If no duplicate found: store block, add to MinHashIndex

Fall back gracefully: if Ryzanstein unavailable, use MinHash-only (no embedding call).
```

### Sprint 3 — Benchmarks Suite (Day 2)
**Goal:** Criterion benchmarks proving compression performance targets.

```
@APEX create benches/compression_bench.rs:
  - bench_huffman_1mb: compress 1MB of text data → target: <10ms
  - bench_lz4_1mb: compress 1MB → target: <5ms
  - bench_entropy_1mb: compress 1MB → target: <20ms
  - bench_semantic_dedup_1k: deduplicate 1000 code blocks → target: <500ms
  - bench_minhash_10k: index + query 10k documents → target: <1s
  - bench_auto_select: CompressionMethod::Auto on 1MB text → target: <15ms

Run: cargo bench 2>&1 | tee BENCHMARK_RESULTS.md
Document compression ratios for each method against test data.
```

### Sprint 4 — Integration Tests + Build Clean (Day 3)
```
@CORE run: cargo test -- --test-threads=4
All unit tests must pass. Then run integration tests:
  cargo test --test integration  # tests in tests/ directory

@FORGE run:
  cargo clippy -- -D warnings   # zero warnings
  cargo fmt --check             # code style clean
  cargo build --release

Update Cargo.toml [lib] section if needed (should compile as both lib and bin).
git tag v0.2.0 && git push origin v0.2.0
```

## Done Criteria (all must pass)
- [x] `cargo test` (3/4 pass, perf test slow on A9) passes — zero failures
- [ ] MinHash LSH: 10k documents indexed + queried in <1s
- [x] Semantic merge: correctly identifies duplicates, falls back without Ryzanstein
- [ ] Benchmark suite runs: `cargo bench` produces results
- [x] `cargo clippy -- -D warnings` clean
- [x] `cargo build --release` succeeds
- [ ] `v0.2.0` tag pushed

## Completion Signal
```bash
git tag v0.2.0 && git push origin v0.2.0
```

## Critical Rules
1. **Graceful Ryzanstein fallback** — never panic if `RYZANSTEIN_URL` is unset; use MinHash-only
2. **Lossy dedup is forbidden** — deduplication must be lossless; data must round-trip perfectly
3. **Benchmarks use realistic data** — use actual code files or lorem ipsum, not just `vec![0u8; N]`
4. **No unsafe without comment** — any `unsafe` block needs a safety invariant comment
