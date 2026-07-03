//! Ryzanstein integration for semantic compression
//!
//! Uses Ryzanstein embeddings to identify semantically similar blocks
//! for enhanced deduplication.

use serde::Deserialize;
use std::time::Duration;

use crate::error::CompressError;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

/// Client for Ryzanstein semantic services
pub struct RyzansteinCompressClient {
    base_url: String,
    http: reqwest::Client,
}

impl RyzansteinCompressClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    /// Get semantic embeddings for code blocks via Ryzanstein's
    /// `/v1/embeddings`. Callers are expected to fall back to
    /// `fallback_embed` on `Err` -- this method reports failure honestly
    /// rather than swallowing it, so the caller can apply the
    /// "never panic, always produce a usable result" policy itself.
    pub async fn get_embeddings(&self, blocks: &[String]) -> Result<Vec<Vec<f32>>, CompressError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let payload = serde_json::json!({ "input": blocks });

        let resp = self
            .http
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| CompressError::RyzansteinError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(CompressError::RyzansteinError(format!(
                "embeddings request failed: HTTP {}",
                resp.status()
            )));
        }

        let parsed: EmbeddingsResponse = resp
            .json()
            .await
            .map_err(|e| CompressError::RyzansteinError(format!("bad embeddings response: {e}")))?;

        if parsed.data.len() != blocks.len() {
            return Err(CompressError::RyzansteinError(format!(
                "expected {} embeddings, got {}",
                blocks.len(),
                parsed.data.len()
            )));
        }

        Ok(parsed.data.into_iter().map(|item| item.embedding).collect())
    }

    /// Compute cosine similarity between two embedding vectors.
    /// Delegates to the canonical `crate::similarity::cosine_similarity`
    /// (kept as an associated fn for API compatibility).
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        crate::similarity::cosine_similarity(a, b)
    }

    /// Health check for Ryzanstein connectivity. Never returns `Err` for a
    /// normal "service is down" condition -- that's expected, handleable
    /// state (see `merge_or_store`'s graceful-fallback contract), not an
    /// exceptional error.
    pub async fn health_check(&self) -> Result<bool, CompressError> {
        let url = format!("{}/health", self.base_url);
        match self.http.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reserved, essentially-never-listened-on port: gives a fast
    /// "connection refused" instead of waiting out a timeout, so these
    /// tests stay hermetic (no live Ryzanstein needed) without being slow.
    const UNREACHABLE_URL: &str = "http://127.0.0.1:1";

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

    #[tokio::test]
    async fn test_health_check_reports_false_when_unreachable() {
        let client = RyzansteinCompressClient::new(UNREACHABLE_URL);
        assert_eq!(client.health_check().await.unwrap(), false);
    }

    #[tokio::test]
    async fn test_get_embeddings_errs_when_unreachable() {
        // The client itself must report failure honestly (Err), not
        // silently substitute the fallback -- that policy belongs to the
        // caller (SemanticDeduplicator::cosine_check).
        let client = RyzansteinCompressClient::new(UNREACHABLE_URL);
        let blocks = vec!["fn main()".to_string(), "def hello()".to_string()];
        let result = client.get_embeddings(&blocks).await;
        assert!(result.is_err());
    }
}
