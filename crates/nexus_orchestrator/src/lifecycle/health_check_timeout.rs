//! Health Check Timeout Manager
//! Provides strict timeout windows for subsystem initialization health checks

use std::time::{Duration, Instant};
use tokio::time::{timeout, Timeout};
use tracing::{warn, error};

/// Configuration for health check timeouts
#[derive(Debug, Clone)]
pub struct HealthCheckTimeout {
    /// Default timeout for standard health checks
    pub default_timeout: Duration,
    /// Extended timeout for critical operations (e.g., order book healing)
    pub extended_timeout: Duration,
    /// Maximum retries before failing
    pub max_retries: u32,
}

impl Default for HealthCheckTimeout {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(10),
            extended_timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

impl HealthCheckTimeout {
    pub fn new(default_ms: u64, extended_ms: u64, retries: u32) -> Self {
        Self {
            default_timeout: Duration::from_millis(default_ms),
            extended_timeout: Duration::from_millis(extended_ms),
            max_retries: retries,
        }
    }

    /// Execute a health check with timeout and retry logic
    pub async fn execute_with_retry<F, T, E>(
        &self,
        check_name: &str,
        mut check_fn: F,
    ) -> Result<T, HealthCheckError>
    where
        F: FnMut() -> futures_util::future::BoxFuture<'static, Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut last_error: Option<String> = None;
        let start = Instant::now();

        for attempt in 0..self.max_retries {
            match self.execute_single(check_name, &mut check_fn).await {
                Ok(result) => {
                    if attempt > 0 {
                        warn!(
                            "Health check '{}' succeeded after {} attempts (total time: {:?})",
                            check_name,
                            attempt + 1,
                            start.elapsed()
                        );
                    }
                    return Ok(result);
                }
                Err(err) => {
                    last_error = Some(format!("{}", err));
                    warn!(
                        "Health check '{}' failed (attempt {}/{}): {}",
                        check_name,
                        attempt + 1,
                        self.max_retries,
                        last_error.as_ref().unwrap()
                    );

                    if attempt < self.max_retries - 1 {
                        // Exponential backoff before retry
                        let backoff = Duration::from_millis(100 * (2u64.pow(attempt)));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        Err(HealthCheckError::MaxRetriesExceeded {
            check_name: check_name.to_string(),
            attempts: self.max_retries,
            last_error: last_error.unwrap_or_default(),
            total_time: start.elapsed(),
        })
    }

    async fn execute_single<F, T, E>(
        &self,
        check_name: &str,
        check_fn: &mut F,
    ) -> Result<T, HealthCheckError>
    where
        F: FnMut() -> futures_util::future::BoxFuture<'static, Result<T, E>>,
        E: std::fmt::Display,
    {
        let result = timeout(self.default_timeout, check_fn()).await;

        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => Err(HealthCheckError::CheckFailed {
                check_name: check_name.to_string(),
                reason: e.to_string(),
            }),
            Err(_) => Err(HealthCheckError::Timeout {
                check_name: check_name.to_string(),
                timeout: self.default_timeout,
            }),
        }
    }

    /// Execute a single health check with extended timeout (for order book healing, etc.)
    pub async fn execute_extended<F, T, E>(
        &self,
        check_name: &str,
        check_fn: F,
    ) -> Result<T, HealthCheckError>
    where
        F: futures_util::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let result = timeout(self.extended_timeout, check_fn).await;

        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => Err(HealthCheckError::CheckFailed {
                check_name: check_name.to_string(),
                reason: e.to_string(),
            }),
            Err(_) => Err(HealthCheckError::Timeout {
                check_name: check_name.to_string(),
                timeout: self.extended_timeout,
            }),
        }
    }
}

/// Health check error types
#[derive(Debug, thiserror::Error)]
pub enum HealthCheckError {
    #[error("Health check '{check_name}' failed: {reason}")]
    CheckFailed {
        check_name: String,
        reason: String,
    },

    #[error("Health check '{check_name}' timed out after {timeout:?}")]
    Timeout {
        check_name: String,
        timeout: Duration,
    },

    #[error("Health check '{check_name}' exceeded max retries ({attempts}): {last_error} (total time: {total_time:?})")]
    MaxRetriesExceeded {
        check_name: String,
        attempts: u32,
        last_error: String,
        total_time: Duration,
    },
}

/// Helper trait for health-checkable components
pub trait HealthCheckable {
    fn health_check(&self) -> futures_util::future::BoxFuture<'static, Result<(), String>>;
    fn component_name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_success() {
        let hc = HealthCheckTimeout::default();
        
        let result = hc.execute_with_retry(
            "test_check",
            || {
                Box::pin(async {
                    Ok::<(), String>("success".to_string())
                })
            },
        ).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_timeout() {
        let hc = HealthCheckTimeout::new(50, 100, 1);
        
        let result = hc.execute_with_retry(
            "slow_check",
            || {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    Ok::<(), String>("done".to_string())
                })
            },
        ).await;
        
        assert!(matches!(result, Err(HealthCheckError::Timeout { .. })));
    }
}
