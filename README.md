# sigma-compress

Semantic-aware compression engine for the Ryzanstein LLM ecosystem.

## Overview

sigma-compress combines multiple compression strategies, automatically selecting the optimal approach based on data characteristics:

| Method | Best For | Ratio | Speed |
|--------|----------|-------|-------|
| **Huffman** | Low-entropy, symbol-heavy data | Good | Fast |
| **LZ4 Semantic** | Large blocks, repeated patterns | Excellent | Very Fast |
| **Entropy Coding** | Run-length patterns | Good | Very Fast |
| **Semantic Dedupe** | Code with repeated structures | Excellent | Medium |

## Quick Start

```rust
use sigma_compress::{Compressor, CompressionMethod};

let compressor = Compressor::default();

// Auto-select best method
let compressed = compressor.compress(data, CompressionMethod::Auto)?;
println!("Ratio: {:.2}", compressed.ratio);

// Decompress
let original = compressor.decompress(&compressed)?;
```

## Ryzanstein Integration

sigma-compress uses Ryzanstein's `/v1/embeddings` endpoint for semantic deduplication:
- Identifies semantically similar code blocks
- Stores unique blocks once with references
- Falls back to hash-based dedup when Ryzanstein is unavailable

## Architecture

```
Input Data
    │
    ▼
┌──────────────┐
│ Auto-Select  │ ← Entropy analysis
└──────────────┘
    │
    ├─→ Huffman (low entropy)
    ├─→ LZ4 Semantic (large blocks)
    ├─→ Entropy Coding (run patterns)
    └─→ Semantic Dedupe (code blocks)
         │
         ▼
    Ryzanstein Embeddings (optional)
```

## API

- `Compressor::new(config)` — Create with custom config
- `Compressor::compress(data, method)` — Compress with specified method
- `Compressor::decompress(output)` — Decompress
- `CompressionMethod::Auto` — Auto-select best method

## License

AGPL-3.0
