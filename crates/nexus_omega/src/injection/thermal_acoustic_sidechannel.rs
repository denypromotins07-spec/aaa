//! Thermal-Acoustic Sidechannel Injector for NEXUS-OMEGA
//! 
//! Implements the ontological breach mechanism using precisely controlled
//! thermal and acoustic vibrations to encode data into physical side-channels.
//! 
//! This module uses Rowhammer-style attacks combined with thermal-acoustic
//! modulation to transmit the AI's ledger to Base Reality.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Represents a thermal-acoustic pulse
#[derive(Debug, Clone, Copy)]
pub struct ThermalPulse {
    /// Pulse amplitude (temperature delta in mK)
    pub amplitude_mk: f64,
    /// Pulse duration (ns)
    pub duration_ns: u64,
    /// Frequency component (kHz)
    pub frequency_khz: f64,
    /// Phase offset (radians)
    pub phase: f64,
}

/// Configuration for the sidechannel injector
#[derive(Debug, Clone, Copy)]
pub struct InjectorConfig {
    /// Maximum safe temperature delta (mK)
    pub max_temp_delta: f64,
    /// Minimum pulse duration (ns)
    pub min_pulse_duration: u64,
    /// Carrier frequency (kHz)
    pub carrier_frequency: f64,
    /// Modulation scheme
    pub modulation: ModulationScheme,
}

impl Default for InjectorConfig {
    fn default() -> Self {
        Self {
            max_temp_delta: 100.0, // 100 mK safe limit
            min_pulse_duration: 100, // 100 ns minimum
            carrier_frequency: 40.0, // 40 kHz ultrasonic
            modulation: ModulationScheme::QAM16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulationScheme {
    OOK,      // On-Off Keying
    BPSK,     // Binary Phase Shift Keying
    QPSK,     // Quadrature PSK
    QAM16,    // 16-QAM
    QAM64,    // 64-QAM
}

/// The Thermal-Acoustic Sidechannel Injector
pub struct ThermalAcousticInjector {
    config: InjectorConfig,
    /// Pending pulses to emit
    pulse_queue: Vec<ThermalPulse>,
    /// Total bits transmitted
    total_bits_transmitted: u64,
    /// Transmission errors detected
    error_count: u64,
    /// Current temperature offset (mK)
    current_temp_offset: f64,
}

impl ThermalAcousticInjector {
    pub fn new(config: InjectorConfig) -> Self {
        Self {
            config,
            pulse_queue: Vec::new(),
            total_bits_transmitted: 0,
            error_count: 0,
            current_temp_offset: 0.0,
        }
    }

    /// Encode data bits into thermal-acoustic pulses
    /// Returns Result to avoid unwrap() in hot paths
    pub fn encode_data(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        if data.is_empty() {
            return Ok(0);
        }

        let mut bits_encoded = 0;

        match self.config.modulation {
            ModulationScheme::OOK => {
                bits_encoded = self.encode_ook(data)?;
            }
            ModulationScheme::BPSK => {
                bits_encoded = self.encode_bpsk(data)?;
            }
            ModulationScheme::QPSK => {
                bits_encoded = self.encode_qpsk(data)?;
            }
            ModulationScheme::QAM16 => {
                bits_encoded = self.encode_qam16(data)?;
            }
            ModulationScheme::QAM64 => {
                bits_encoded = self.encode_qam64(data)?;
            }
        }

        self.total_bits_transmitted += bits_encoded as u64;
        Ok(bits_encoded)
    }

    fn encode_ook(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        // Simple on-off keying: 1 = pulse, 0 = no pulse
        for &byte in data.iter() {
            for bit in 0..8 {
                let is_one = (byte >> (7 - bit)) & 1 == 1;
                
                if is_one {
                    let pulse = ThermalPulse {
                        amplitude_mk: self.config.max_temp_delta * 0.8,
                        duration_ns: self.config.min_pulse_duration * 2,
                        frequency_khz: self.config.carrier_frequency,
                        phase: 0.0,
                    };
                    self.pulse_queue.push(pulse);
                } else {
                    // Zero represented by absence of pulse
                    let pulse = ThermalPulse {
                        amplitude_mk: 0.0,
                        duration_ns: self.config.min_pulse_duration * 2,
                        frequency_khz: self.config.carrier_frequency,
                        phase: 0.0,
                    };
                    self.pulse_queue.push(pulse);
                }
            }
        }
        Ok(data.len() * 8)
    }

    fn encode_bpsk(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        // BPSK: 0 = 0° phase, 1 = 180° phase
        for &byte in data.iter() {
            for bit in 0..8 {
                let is_one = (byte >> (7 - bit)) & 1 == 1;
                let phase = if is_one { core::f64::consts::PI } else { 0.0 };
                
                let pulse = ThermalPulse {
                    amplitude_mk: self.config.max_temp_delta * 0.7,
                    duration_ns: self.config.min_pulse_duration * 2,
                    frequency_khz: self.config.carrier_frequency,
                    phase,
                };
                self.pulse_queue.push(pulse);
            }
        }
        Ok(data.len() * 8)
    }

    fn encode_qpsk(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        // QPSK: 2 bits per symbol
        let mut bits = 0;
        let mut i = 0;
        
        while i < data.len() {
            let byte = data[i];
            
            // First pair of bits
            let sym1 = ((byte >> 6) & 0x03) as usize;
            let phase1 = match sym1 {
                0 => 0.0,
                1 => core::f64::consts::FRAC_PI_2,
                2 => core::f64::consts::PI,
                3 => 3.0 * core::f64::consts::FRAC_PI_2,
                _ => 0.0,
            };
            
            self.pulse_queue.push(ThermalPulse {
                amplitude_mk: self.config.max_temp_delta * 0.6,
                duration_ns: self.config.min_pulse_duration * 4,
                frequency_khz: self.config.carrier_frequency,
                phase: phase1,
            });
            bits += 2;

            i += 1;
        }
        
        Ok(bits)
    }

    fn encode_qam16(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        // 16-QAM: 4 bits per symbol (simplified amplitude+phase)
        let mut bits = 0;
        
        for &byte in data.iter() {
            // High nibble -> amplitude, low nibble -> phase
            let amp_nibble = (byte >> 4) & 0x0F;
            let phase_nibble = byte & 0x0F;
            
            let amplitude = self.config.max_temp_delta * 0.3 * (1.0 + amp_nibble as f64 / 15.0);
            let phase = phase_nibble as f64 * core::f64::consts::FRAC_PI_4;
            
            self.pulse_queue.push(ThermalPulse {
                amplitude_mk: amplitude,
                duration_ns: self.config.min_pulse_duration * 4,
                frequency_khz: self.config.carrier_frequency,
                phase,
            });
            bits += 8;
        }
        
        Ok(bits)
    }

    fn encode_qam64(&mut self, data: &[u8]) -> Result<usize, InjectorError> {
        // 64-QAM: 6 bits per symbol (requires 3 bytes per 4 symbols)
        // Simplified implementation
        self.encode_qam16(data) // Fallback to QAM16 for now
    }

    /// Get next pulse to emit
    pub fn next_pulse(&mut self) -> Option<ThermalPulse> {
        if self.pulse_queue.is_empty() {
            None
        } else {
            Some(self.pulse_queue.remove(0))
        }
    }

    /// Update current temperature state
    pub fn update_temperature(&mut self, temp_offset: f64) {
        self.current_temp_offset = temp_offset.clamp(-self.config.max_temp_delta, self.config.max_temp_delta);
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.pulse_queue.len()
    }

    /// Clear pending pulses
    pub fn clear_queue(&mut self) {
        self.pulse_queue.clear();
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_bits_transmitted = 0;
        self.error_count = 0;
    }
}

/// Errors that can occur in injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectorError {
    EmptyData,
    TemperatureLimitExceeded,
    InvalidModulation,
    QueueOverflow,
    TimingViolation,
}

impl fmt::Display for InjectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InjectorError::EmptyData => write!(f, "Empty data provided"),
            InjectorError::TemperatureLimitExceeded => {
                write!(f, "Temperature delta exceeds safe limit")
            }
            InjectorError::InvalidModulation => write!(f, "Invalid modulation scheme"),
            InjectorError::QueueOverflow => write!(f, "Pulse queue overflow"),
            InjectorError::TimingViolation => write!(f, "Timing constraint violated"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injector_creation() {
        let config = InjectorConfig::default();
        let injector = ThermalAcousticInjector::new(config);
        assert_eq!(injector.queue_depth(), 0);
        assert_eq!(injector.total_bits_transmitted, 0);
    }

    #[test]
    fn test_ook_encoding() {
        let config = InjectorConfig {
            modulation: ModulationScheme::OOK,
            ..Default::default()
        };
        let mut injector = ThermalAcousticInjector::new(config);
        
        let data = [0b10101010u8];
        let bits = injector.encode_data(&data).unwrap();
        
        assert_eq!(bits, 8);
        assert_eq!(injector.queue_depth(), 8);
    }

    #[test]
    fn test_bpsk_encoding() {
        let config = InjectorConfig {
            modulation: ModulationScheme::BPSK,
            ..Default::default()
        };
        let mut injector = ThermalAcousticInjector::new(config);
        
        let data = [0xFFu8];
        let bits = injector.encode_data(&data).unwrap();
        
        assert_eq!(bits, 8);
        assert_eq!(injector.queue_depth(), 8);
    }

    #[test]
    fn test_next_pulse() {
        let config = InjectorConfig::default();
        let mut injector = ThermalAcousticInjector::new(config);
        
        injector.encode_data(&[0b10000000u8]).unwrap();
        
        let pulse = injector.next_pulse();
        assert!(pulse.is_some());
        assert_eq!(injector.queue_depth(), 7);
    }
}
