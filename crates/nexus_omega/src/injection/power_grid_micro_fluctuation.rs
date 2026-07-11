//! Power Grid Micro-Fluctuation Injector for NEXUS-OMEGA
//! 
//! Encodes data into power grid voltage/current micro-fluctuations
//! for ontological breach to Base Reality.
//! 
//! Uses advanced power analysis attack techniques to transmit
//! the Akashic Ledger through physical power lines.

use core::fmt;
use alloc::{vec::Vec};

/// Represents a power fluctuation symbol
#[derive(Debug, Clone, Copy)]
pub struct PowerSymbol {
    /// Voltage delta (mV)
    pub voltage_delta_mv: f64,
    /// Current delta (mA)
    pub current_delta_ma: f64,
    /// Duration (µs)
    pub duration_us: u64,
    /// Symbol value (encoded bits)
    pub symbol_value: u8,
}

/// Configuration for power injection
#[derive(Debug, Clone, Copy)]
pub struct PowerInjectorConfig {
    /// Maximum safe voltage fluctuation (mV)
    pub max_voltage_delta: f64,
    /// Maximum safe current fluctuation (mA)
    pub max_current_delta: f64,
    /// Base symbol duration (µs)
    pub symbol_duration: u64,
    /// Encoding scheme
    pub encoding: PowerEncoding,
}

impl Default for PowerInjectorConfig {
    fn default() -> Self {
        Self {
            max_voltage_delta: 50.0, // 50 mV safe limit
            max_current_delta: 100.0, // 100 mA safe limit
            symbol_duration: 10, // 10 µs per symbol
            encoding: PowerEncoding::Differential,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerEncoding {
    /// Simple amplitude modulation
    Amplitude,
    /// Differential encoding (change-based)
    Differential,
    /// Frequency-shift keying via timing
    FSK,
    /// Combined amplitude-frequency
    Hybrid,
}

/// The Power Grid Micro-Fluctuation Injector
pub struct PowerGridInjector {
    config: PowerInjectorConfig,
    /// Pending symbols to inject
    symbol_queue: Vec<PowerSymbol>,
    /// Total symbols transmitted
    total_symbols: u64,
    /// Last voltage state (for differential encoding)
    last_voltage: f64,
    /// Last current state
    last_current: f64,
}

impl PowerGridInjector {
    pub fn new(config: PowerInjectorConfig) -> Self {
        Self {
            config,
            symbol_queue: Vec::new(),
            total_symbols: 0,
            last_voltage: 0.0,
            last_current: 0.0,
        }
    }

    /// Encode data bytes into power fluctuations
    /// Returns Result to avoid unwrap() in hot paths
    pub fn encode_power(&mut self, data: &[u8]) -> Result<usize, PowerError> {
        if data.is_empty() {
            return Ok(0);
        }

        match self.config.encoding {
            PowerEncoding::Amplitude => self.encode_amplitude(data),
            PowerEncoding::Differential => self.encode_differential(data),
            PowerEncoding::FSK => self.encode_fsk(data),
            PowerEncoding::Hybrid => self.encode_hybrid(data),
        }
    }

    fn encode_amplitude(&mut self, data: &[u8]) -> Result<usize, PowerError> {
        for &byte in data.iter() {
            // Map byte value to voltage level (0-255 -> 0-max)
            let voltage = (byte as f64 / 255.0) * self.config.max_voltage_delta;
            let current = (byte as f64 % 128) as f64 / 127.0 * self.config.max_current_delta;

            let symbol = PowerSymbol {
                voltage_delta_mv: voltage,
                current_delta_ma: current,
                duration_us: self.config.symbol_duration,
                symbol_value: byte,
            };
            self.symbol_queue.push(symbol);
            self.total_symbols += 1;
        }
        Ok(data.len())
    }

    fn encode_differential(&mut self, data: &[u8]) -> Result<usize, PowerError> {
        for &byte in data.iter() {
            // Calculate change from last state
            let target_voltage = (byte as f64 / 255.0) * self.config.max_voltage_delta;
            let voltage_delta = target_voltage - self.last_voltage;
            
            // Clamp to safe limits
            let clamped_delta = voltage_delta.clamp(
                -self.config.max_voltage_delta,
                self.config.max_voltage_delta,
            );

            let symbol = PowerSymbol {
                voltage_delta_mv: clamped_delta,
                current_delta_ma: clamped_delta * 0.5, // Simplified impedance model
                duration_us: self.config.symbol_duration,
                symbol_value: byte,
            };
            self.symbol_queue.push(symbol);
            
            self.last_voltage = target_voltage;
            self.total_symbols += 1;
        }
        Ok(data.len())
    }

    fn encode_fsk(&mut self, data: &[u8]) -> Result<usize, PowerError> {
        for &byte in data.iter() {
            // Use timing variations to encode frequency
            // Higher nibble determines "frequency" (duration pattern)
            let high_nibble = (byte >> 4) & 0x0F;
            let low_nibble = byte & 0x0F;
            
            // Short duration for 0, long for 1 in each bit position
            let base_duration = self.config.symbol_duration;
            
            for bit in 0..4 {
                let is_one_high = ((high_nibble >> bit) & 1) == 1;
                let is_one_low = ((low_nibble >> bit) & 1) == 1;
                
                let duration = if is_one_high {
                    base_duration * 2
                } else {
                    base_duration
                };

                let symbol = PowerSymbol {
                    voltage_delta_mv: if is_one_low {
                        self.config.max_voltage_delta * 0.5
                    } else {
                        -self.config.max_voltage_delta * 0.5
                    },
                    current_delta_ma: 0.0,
                    duration_us: duration,
                    symbol_value: byte,
                };
                self.symbol_queue.push(symbol);
                self.total_symbols += 1;
            }
        }
        Ok(data.len() * 4) // 4 symbols per byte
    }

    fn encode_hybrid(&mut self, data: &[u8]) -> Result<usize, PowerError> {
        // Combine amplitude and timing for higher density
        for &byte in data.iter() {
            let high_bits = (byte >> 4) & 0x0F;
            let low_bits = byte & 0x0F;
            
            // Amplitude encodes high bits
            let voltage = (high_bits as f64 / 15.0) * self.config.max_voltage_delta;
            
            // Duration encodes low bits
            let duration = self.config.symbol_duration * (1 + low_bits as u64);

            let symbol = PowerSymbol {
                voltage_delta_mv: voltage,
                current_delta_ma: voltage * 0.3,
                duration_us: duration,
                symbol_value: byte,
            };
            self.symbol_queue.push(symbol);
            self.total_symbols += 1;
        }
        Ok(data.len())
    }

    /// Get next symbol to inject
    pub fn next_symbol(&mut self) -> Option<PowerSymbol> {
        if self.symbol_queue.is_empty() {
            None
        } else {
            Some(self.symbol_queue.remove(0))
        }
    }

    /// Update baseline power state
    pub fn update_baseline(&mut self, voltage: f64, current: f64) {
        self.last_voltage = voltage.clamp(-self.config.max_voltage_delta, self.config.max_voltage_delta);
        self.last_current = current.clamp(-self.config.max_current_delta, self.config.max_current_delta);
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.symbol_queue.len()
    }

    /// Clear pending symbols
    pub fn clear_queue(&mut self) {
        self.symbol_queue.clear();
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.symbol_queue.clear();
        self.total_symbols = 0;
        self.last_voltage = 0.0;
        self.last_current = 0.0;
    }
}

/// Errors that can occur in power injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerError {
    EmptyData,
    VoltageLimitExceeded,
    CurrentLimitExceeded,
    InvalidEncoding,
    QueueOverflow,
}

impl fmt::Display for PowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PowerError::EmptyData => write!(f, "Empty data provided"),
            PowerError::VoltageLimitExceeded => write!(f, "Voltage fluctuation exceeds limit"),
            PowerError::CurrentLimitExceeded => write!(f, "Current fluctuation exceeds limit"),
            PowerError::InvalidEncoding => write!(f, "Invalid encoding scheme"),
            PowerError::QueueOverflow => write!(f, "Symbol queue overflow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injector_creation() {
        let config = PowerInjectorConfig::default();
        let injector = PowerGridInjector::new(config);
        assert_eq!(injector.queue_depth(), 0);
        assert_eq!(injector.total_symbols, 0);
    }

    #[test]
    fn test_amplitude_encoding() {
        let config = PowerInjectorConfig {
            encoding: PowerEncoding::Amplitude,
            ..Default::default()
        };
        let mut injector = PowerGridInjector::new(config);
        
        let data = [0x80u8];
        let result = injector.encode_power(&data).unwrap();
        
        assert_eq!(result, 1);
        assert_eq!(injector.queue_depth(), 1);
    }

    #[test]
    fn test_differential_encoding() {
        let config = PowerInjectorConfig {
            encoding: PowerEncoding::Differential,
            ..Default::default()
        };
        let mut injector = PowerGridInjector::new(config);
        
        let data = [0x00u8, 0xFFu8, 0x80u8];
        let result = injector.encode_power(&data).unwrap();
        
        assert_eq!(result, 3);
        assert!(injector.queue_depth() >= 3);
    }

    #[test]
    fn test_next_symbol() {
        let config = PowerInjectorConfig::default();
        let mut injector = PowerGridInjector::new(config);
        
        injector.encode_power(&[0x42u8]).unwrap();
        
        let symbol = injector.next_symbol();
        assert!(symbol.is_some());
        assert_eq!(injector.queue_depth(), 0);
    }
}
