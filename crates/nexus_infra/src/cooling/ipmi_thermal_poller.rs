//! IPMI/Redfish Thermal Poller for Silicon Die Temperature Monitoring
//! 
//! Implements zero-allocation polling of hardware temperature sensors via
//! IPMI (Intelligent Platform Management Interface) and Redfish REST API.

use core::fmt;
use core::time::Duration;
use crate::cooling::microfluidic_pid::{PidError, ThermalSensor};

/// Maximum number of thermal zones to track
const MAX_THERMAL_ZONES: usize = 32;
/// Default polling interval in milliseconds
const DEFAULT_POLL_INTERVAL_MS: u64 = 100;
/// Maximum consecutive failures before declaring sensor dead
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Errors specific to IPMI/Redfish communication
#[derive(Debug, Clone, PartialEq)]
pub enum IpmiError {
    ConnectionFailed,
    AuthenticationFailed,
    Timeout,
    InvalidResponse,
    SensorNotFound,
    RateLimitExceeded,
    HardwareFailure,
}

impl fmt::Display for IpmiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpmiError::ConnectionFailed => write!(f, "Failed to connect to BMC"),
            IpmiError::AuthenticationFailed => write!(f, "IPMI/Redfish authentication failed"),
            IpmiError::Timeout => write!(f, "Request timeout"),
            IpmiError::InvalidResponse => write!(f, "Invalid response from BMC"),
            IpmiError::SensorNotFound => write!(f, "Thermal sensor not found"),
            IpmiError::RateLimitExceeded => write!(f, "BMC rate limit exceeded"),
            IpmiError::HardwareFailure => write!(f, "BMC hardware failure"),
        }
    }
}

/// Thermal zone data structure
#[derive(Debug, Clone, Copy)]
pub struct ThermalZone {
    /// Zone identifier
    pub id: u8,
    /// Zone name
    pub name: [u8; 32],
    /// Current temperature (°C)
    pub temperature: f64,
    /// High threshold (°C)
    pub high_threshold: f64,
    /// Critical threshold (°C)
    pub critical_threshold: f64,
    /// Sensor health flag
    pub healthy: bool,
    /// Consecutive read failures
    pub consecutive_failures: u32,
    /// Last successful read timestamp (ms)
    pub last_read_ms: u64,
}

impl Default for ThermalZone {
    fn default() -> Self {
        Self {
            id: 0,
            name: [0u8; 32],
            temperature: 0.0,
            high_threshold: 85.0,
            critical_threshold: 105.0,
            healthy: true,
            consecutive_failures: 0,
            last_read_ms: 0,
        }
    }
}

/// IPMI/Redfish thermal poller state
pub struct IpmiThermalPoller {
    /// Array of thermal zones (zero-allocation)
    zones: [ThermalZone; MAX_THERMAL_ZONES],
    /// Number of active zones
    active_zone_count: usize,
    /// Polling interval (ms)
    poll_interval_ms: u64,
    /// Last poll timestamp (ms)
    last_poll_ms: u64,
    /// Connection established flag
    connected: bool,
    /// Base URL for Redfish API
    redfish_base_url: [u8; 128],
    /// IPMI address
    ipmi_address: [u8; 16],
}

impl IpmiThermalPoller {
    /// Create a new thermal poller
    pub fn new() -> Self {
        Self {
            zones: [ThermalZone::default(); MAX_THERMAL_ZONES],
            active_zone_count: 0,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            last_poll_ms: 0,
            connected: false,
            redfish_base_url: [0u8; 128],
            ipmi_address: [0u8; 16],
        }
    }

    /// Initialize connection to BMC via IPMI
    pub fn init_ipmi(&mut self, address: &[u8]) -> Result<(), IpmiError> {
        if address.len() > 16 {
            return Err(IpmiError::InvalidResponse);
        }
        
        // Copy address
        let mut addr = [0u8; 16];
        addr[..address.len()].copy_from_slice(address);
        self.ipmi_address = addr;
        
        // Simulate connection (real implementation would use IPMI library)
        self.connected = true;
        
        // Discover thermal zones
        self.discover_zones()?;
        
        Ok(())
    }

    /// Initialize connection to BMC via Redfish
    pub fn init_redfish(&mut self, base_url: &str) -> Result<(), IpmiError> {
        if base_url.len() > 127 {
            return Err(IpmiError::InvalidResponse);
        }
        
        // Copy URL
        let mut url = [0u8; 128];
        let bytes = base_url.as_bytes();
        url[..bytes.len()].copy_from_slice(bytes);
        self.redfish_base_url = url;
        
        // Simulate connection (real implementation would use HTTP client)
        self.connected = true;
        
        // Discover thermal zones
        self.discover_zones()?;
        
        Ok(())
    }

    /// Discover available thermal zones
    fn discover_zones(&mut self) -> Result<(), IpmiError> {
        if !self.connected {
            return Err(IpmiError::ConnectionFailed);
        }

        // In real implementation, this would query the BMC
        // For now, simulate discovering CPU and GPU zones
        
        // Zone 0: CPU Package
        self.zones[0] = ThermalZone {
            id: 0,
            name: *b"CPU_PACKAGE_TEMP                    ",
            temperature: 45.0,
            high_threshold: 85.0,
            critical_threshold: 105.0,
            healthy: true,
            ..Default::default()
        };
        
        // Zone 1: GPU 0
        self.zones[1] = ThermalZone {
            id: 1,
            name: *b"GPU_0_TEMP                          ",
            temperature: 50.0,
            high_threshold: 90.0,
            critical_threshold: 110.0,
            healthy: true,
            ..Default::default()
        };
        
        // Zone 2: GPU 1
        self.zones[2] = ThermalZone {
            id: 2,
            name: *b"GPU_1_TEMP                          ",
            temperature: 48.0,
            high_threshold: 90.0,
            critical_threshold: 110.0,
            healthy: true,
            ..Default::default()
        };
        
        self.active_zone_count = 3;
        
        Ok(())
    }

    /// Poll all thermal zones
    pub fn poll(&mut self, timestamp_ms: u64) -> Result<(), IpmiError> {
        if !self.connected {
            return Err(IpmiError::ConnectionFailed);
        }

        // Check polling interval
        if timestamp_ms - self.last_poll_ms < self.poll_interval_ms {
            return Ok(()); // Not time to poll yet
        }

        self.last_poll_ms = timestamp_ms;

        // Poll each active zone
        for i in 0..self.active_zone_count {
            match self.read_zone(i) {
                Ok(temp) => {
                    self.zones[i].temperature = temp;
                    self.zones[i].healthy = true;
                    self.zones[i].consecutive_failures = 0;
                    self.zones[i].last_read_ms = timestamp_ms;
                }
                Err(_) => {
                    self.zones[i].consecutive_failures += 1;
                    if self.zones[i].consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        self.zones[i].healthy = false;
                    }
                }
            }
        }

        Ok(())
    }

    /// Read a single thermal zone (simulated)
    fn read_zone(&self, index: usize) -> Result<f64, IpmiError> {
        if index >= self.active_zone_count {
            return Err(IpmiError::SensorNotFound);
        }

        // In real implementation, this would read from hardware
        // Simulate small temperature variations
        let base_temp = self.zones[index].temperature;
        
        // Add small noise (±0.5°C)
        let noise = ((index as f64 * 0.12345).sin() * 0.5);
        
        Ok(base_temp + noise)
    }

    /// Get temperature for a specific zone by ID
    pub fn get_temperature(&self, zone_id: u8) -> Result<f64, IpmiError> {
        for i in 0..self.active_zone_count {
            if self.zones[i].id == zone_id {
                if !self.zones[i].healthy {
                    return Err(IpmiError::HardwareFailure);
                }
                return Ok(self.zones[i].temperature);
            }
        }
        Err(IpmiError::SensorNotFound)
    }

    /// Get the hottest zone temperature
    pub fn get_max_temperature(&self) -> Result<f64, IpmiError> {
        if self.active_zone_count == 0 {
            return Err(IpmiError::SensorNotFound);
        }

        let mut max_temp = f64::MIN;
        for i in 0..self.active_zone_count {
            if self.zones[i].healthy && self.zones[i].temperature > max_temp {
                max_temp = self.zones[i].temperature;
            }
        }

        if max_temp == f64::MIN {
            Err(IpmiError::HardwareFailure)
        } else {
            Ok(max_temp)
        }
    }

    /// Get average temperature across all healthy zones
    pub fn get_average_temperature(&self) -> Result<f64, IpmiError> {
        if self.active_zone_count == 0 {
            return Err(IpmiError::SensorNotFound);
        }

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.active_zone_count {
            if self.zones[i].healthy {
                sum += self.zones[i].temperature;
                count += 1;
            }
        }

        if count == 0 {
            Err(IpmiError::HardwareFailure)
        } else {
            Ok(sum / count as f64)
        }
    }

    /// Set polling interval
    pub fn set_poll_interval(&mut self, interval_ms: u64) {
        self.poll_interval_ms = interval_ms;
    }

    /// Check if any zone is approaching critical temperature
    pub fn is_approaching_critical(&self, margin: f64) -> bool {
        for i in 0..self.active_zone_count {
            if self.zones[i].healthy {
                let threshold = self.zones[i].critical_threshold - margin;
                if self.zones[i].temperature >= threshold {
                    return true;
                }
            }
        }
        false
    }

    /// Get zone that is closest to critical
    pub fn get_hottest_zone(&self) -> Option<&ThermalZone> {
        let mut hottest_idx: Option<usize> = None;
        let mut max_ratio = 0.0;

        for i in 0..self.active_zone_count {
            if self.zones[i].healthy {
                let ratio = self.zones[i].temperature / self.zones[i].critical_threshold;
                if ratio > max_ratio {
                    max_ratio = ratio;
                    hottest_idx = Some(i);
                }
            }
        }

        hottest_idx.map(|idx| &self.zones[idx])
    }
}

impl ThermalSensor for IpmiThermalPoller {
    fn read_temperature(&self) -> Result<f64, PidError> {
        self.get_max_temperature()
            .map_err(|_| PidError::SensorReadFailure)
    }
}

impl Default for IpmiThermalPoller {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poller_creation() {
        let poller = IpmiThermalPoller::new();
        assert_eq!(poller.active_zone_count, 0);
        assert!(!poller.connected);
    }

    #[test]
    fn test_zone_discovery() {
        let mut poller = IpmiThermalPoller::new();
        let result = poller.init_ipmi(&[192, 168, 1, 100]);
        assert!(result.is_ok());
        assert!(poller.connected);
        assert!(poller.active_zone_count >= 1);
    }

    #[test]
    fn test_temperature_reading() {
        let mut poller = IpmiThermalPoller::new();
        poller.init_ipmi(&[192, 168, 1, 100]).unwrap();
        
        let temp = poller.get_temperature(0);
        assert!(temp.is_ok());
        assert!(temp.unwrap() > 0.0);
    }

    #[test]
    fn test_max_temperature() {
        let mut poller = IpmiThermalPoller::new();
        poller.init_ipmi(&[192, 168, 1, 100]).unwrap();
        
        let max_temp = poller.get_max_temperature();
        assert!(max_temp.is_ok());
    }

    #[test]
    fn test_polling() {
        let mut poller = IpmiThermalPoller::new();
        poller.init_ipmi(&[192, 168, 1, 100]).unwrap();
        
        let result = poller.poll(1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_approaching_critical() {
        let mut poller = IpmiThermalPoller::new();
        poller.init_ipmi(&[192, 168, 1, 100]).unwrap();
        
        // Normal temps should not be critical
        assert!(!poller.is_approaching_critical(20.0));
    }
}
