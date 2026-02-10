//! Error types for sigma-compress

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompressError {
    #[error("empty input")]
    EmptyInput,

    #[error("invalid compression method for this operation")]
    InvalidMethod,

    #[error("huffman encoding error: {0}")]
    HuffmanError(String),

    #[error("lz4 error: {0}")]
    Lz4Error(String),

    #[error("entropy coding error: {0}")]
    EntropyError(String),

    #[error("semantic dedup error: {0}")]
    SemanticError(String),

    #[error("decompression size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("ryzanstein integration error: {0}")]
    RyzansteinError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    SerializationError(String),
}
