//! Configuration for sigma-compress

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub ryzanstein_url: String,
    pub lz4_block_size: usize,
    pub dedup_threshold: f64,
    pub max_input_size: usize,
    pub enable_semantic: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            ryzanstein_url: "http://localhost:8000".to_string(),
            lz4_block_size: 65536,
            dedup_threshold: 0.95,
            max_input_size: 100 * 1024 * 1024, // 100 MB
            enable_semantic: true,
        }
    }
}
