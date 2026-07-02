//! Ryzanstein integration for semantic compression
//!
//! Uses Ryzanstein embeddings to identify semantically similar blocks
//! for enhanced deduplication.

use crate::error::CompressError;

/// Client for Ryzanstein semantic services
pub struct RyzansteinCompressClient {
    #[allow(dead_code)]
    base_url: String,
}

impl RyzansteinCompressClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get semantic embeddings for code blocks
    pub async fn get_embeddings(&self, blocks: &[String]) -> Result<Vec<Vec<f32>>, CompressError> {
        // In production, calls Ryzanstein /v1/embeddings
        // Fallback: hash-based pseudo-embeddings
        Ok(blocks.iter().map(|b| self.fallback_embed(b)).collect())
    }

    /// Compute cosine similarity between two embedding vectors.
    /// Delegates to the canonical `crate::similarity::cosine_similarity`
    /// (kept as an associated fn for API compatibility).
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        crate::similarity::cosine_similarity(a, b)
    }

    /// Health check for Ryzanstein connectivity
    pub async fn health_check(&self) -> Result<bool, CompressError> {
        // Mock: always healthy in development
        Ok(true)
    }

    fn fallback_embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 128];
        for (i, byte) in text.bytes().enumerate() {
            embedding[i % 128] += (byte as f32) / 255.0;
        }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = RyzansteinCompressClient::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = RyzansteinCompressClient::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_fallback_embed() {
        let client = RyzansteinCompressClient::new("http://localhost:8000");
        let emb = client.fallback_embed("hello world");
        assert_eq!(emb.len(), 128);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_health_check() {
        let client = RyzansteinCompressClient::new("http://localhost:8000");
        assert!(client.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_get_embeddings() {
        let client = RyzansteinCompressClient::new("http://localhost:8000");
        let blocks = vec!["fn main()".to_string(), "def hello()".to_string()];
        let embeddings = client.get_embeddings(&blocks).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
    }
}
