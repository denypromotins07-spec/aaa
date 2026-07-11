//! Shannon-Hartley Laser Encoder Module
//!
//! Implements optical laser communication encoding with Forward Error Correction
//! for interstellar data transmission.

use core::marker::PhantomData;

/// Error types for laser encoding
#[derive(Debug, Clone, PartialEq)]
pub enum LaserError {
    InvalidBandwidth,
    SNRTooLow,
    DataTooLarge,
    EncodingFailed,
}

/// Result type for laser operations
pub type LaserResult<T> = Result<T, LaserError>;

/// Shannon-Hartley theorem calculator for channel capacity
pub struct ChannelCapacity {
    /// Bandwidth in Hz
    pub bandwidth_hz: f64,
    /// Signal-to-noise ratio (linear, not dB)
    pub snr_linear: f64,
}

impl ChannelCapacity {
    /// Create new channel capacity calculator
    pub fn new(bandwidth_hz: f64, snr_linear: f64) -> LaserResult<Self> {
        if bandwidth_hz <= 0.0 {
            return Err(LaserError::InvalidBandwidth);
        }
        if snr_linear <= 0.0 {
            return Err(LaserError::SNRTooLow);
        }

        Ok(Self {
            bandwidth_hz,
            snr_linear,
        })
    }

    /// Calculate channel capacity using Shannon-Hartley theorem
    /// C = B * log2(1 + S/N)
    pub fn capacity_bps(&self) -> f64 {
        self.bandwidth_hz * (1.0 + self.snr_linear).log2()
    }

    /// Calculate required SNR for target capacity
    pub fn required_snr(&self, target_capacity_bps: f64) -> f64 {
        // C = B * log2(1 + SNR)
        // SNR = 2^(C/B) - 1
        2.0_f64.powf(target_capacity_bps / self.bandwidth_hz) - 1.0
    }

    /// Calculate minimum bandwidth for target capacity
    pub fn required_bandwidth(&self, target_capacity_bps: f64) -> f64 {
        // B = C / log2(1 + SNR)
        target_capacity_bps / (1.0 + self.snr_linear).log2()
    }
}

/// Reed-Solomon FEC parameters
#[derive(Debug, Clone)]
pub struct RSParams {
    /// Codeword length (n)
    pub n: usize,
    /// Message length (k)
    pub k: usize,
    /// Symbol size in bits
    pub symbol_bits: usize,
}

impl RSParams {
    /// Create standard RS(255, 223) parameters
    pub fn standard() -> Self {
        Self {
            n: 255,
            k: 223,
            symbol_bits: 8,
        }
    }

    /// Calculate code rate
    pub fn code_rate(&self) -> f64 {
        self.k as f64 / self.n as f64
    }

    /// Calculate error correction capability (t symbols)
    pub fn error_correction_symbols(&self) -> usize {
        (self.n - self.k) / 2
    }
}

/// LDPC FEC parameters
#[derive(Debug, Clone)]
pub struct LDPCParams {
    /// Codeword length
    pub block_length: usize,
    /// Code rate
    pub rate: f64,
    /// Column weight
    pub column_weight: usize,
}

impl LDPCParams {
    /// Create standard LDPC parameters
    pub fn rate_half() -> Self {
        Self {
            block_length: 64800,
            rate: 0.5,
            column_weight: 3,
        }
    }

    pub fn rate_three_quarters() -> Self {
        Self {
            block_length: 64800,
            rate: 0.75,
            column_weight: 3,
        }
    }
}

/// FEC scheme selection
#[derive(Debug, Clone)]
pub enum FecScheme {
    ReedSolomon(RSParams),
    LDPC(LDPCParams),
    Concatenated(Box<FecScheme>, Box<FecScheme>),
    None,
}

impl FecScheme {
    /// Get effective code rate
    pub fn code_rate(&self) -> f64 {
        match self {
            FecScheme::ReedSolomon(params) => params.code_rate(),
            FecScheme::LDPC(params) => params.rate,
            FecScheme::Concatenated(inner, outer) => inner.code_rate() * outer.code_rate(),
            FecScheme::None => 1.0,
        }
    }
}

/// Encoded data packet for transmission
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Original data length
    pub original_length: usize,
    /// Encoded length after FEC
    pub encoded_length: usize,
    /// FEC scheme used
    pub fec_scheme: FecScheme,
    /// Packet sequence number
    pub sequence_number: u64,
}

impl EncodedPacket {
    /// Calculate overhead percentage
    pub fn overhead_percent(&self) -> f64 {
        if self.original_length == 0 {
            return 0.0;
        }
        ((self.encoded_length - self.original_length) as f64 / self.original_length as f64) * 100.0
    }
}

/// Shannon-Hartley Laser Encoder
pub struct ShannonHartleyEncoder<'a> {
    channel: ChannelCapacity,
    fec_scheme: FecScheme,
    _marker: PhantomData<&'a ()>,
}

impl<'a> ShannonHartleyEncoder<'a> {
    /// Create new encoder
    pub fn new(bandwidth_hz: f64, snr_linear: f64, fec: FecScheme) -> LaserResult<Self> {
        let channel = ChannelCapacity::new(bandwidth_hz, snr_linear)?;
        
        Ok(Self {
            channel,
            fec_scheme: fec,
            _marker: PhantomData,
        })
    }

    /// Get channel capacity
    pub fn channel_capacity_bps(&self) -> f64 {
        self.channel.capacity_bps()
    }

    /// Get effective data rate after FEC overhead
    pub fn effective_data_rate_bps(&self) -> f64 {
        self.channel.capacity_bps() * self.fec_scheme.code_rate()
    }

    /// Encode data for transmission
    pub fn encode(&self, data_length: usize, sequence: u64) -> LaserResult<EncodedPacket> {
        let code_rate = self.fec_scheme.code_rate();
        if code_rate <= 0.0 || code_rate > 1.0 {
            return Err(LaserError::EncodingFailed);
        }

        let encoded_length = (data_length as f64 / code_rate).ceil() as usize;

        Ok(EncodedPacket {
            original_length: data_length,
            encoded_length,
            fec_scheme: self.fec_scheme.clone(),
            sequence_number: sequence,
        })
    }

    /// Calculate transmission time for given data size
    pub fn transmission_time_seconds(&self, data_bits: u64) -> f64 {
        let effective_rate = self.effective_data_rate_bps();
        if effective_rate <= 0.0 {
            return f64::INFINITY;
        }
        data_bits as f64 / effective_rate
    }

    /// Calculate beam divergence loss at distance
    pub fn beam_divergence_loss(&self, distance_m: f64, wavelength_m: f64, aperture_diameter_m: f64) -> f64 {
        if aperture_diameter_m <= 0.0 || wavelength_m <= 0.0 || distance_m <= 0.0 {
            return f64::INFINITY;
        }

        // Diffraction-limited beam divergence angle (radians)
        let divergence_angle = 1.22 * wavelength_m / aperture_diameter_m;
        
        // Beam spot radius at distance
        let spot_radius = distance_m * divergence_angle;
        
        // Geometric loss (ratio of receiver area to beam area)
        // Assuming receiver aperture is much smaller than beam spot
        let beam_area = core::f64::consts::PI * spot_radius.powi(2);
        
        // Return loss factor (larger = more loss)
        beam_area
    }

    /// Calculate received SNR after beam divergence
    pub fn received_snr(
        &self,
        transmit_power_w: f64,
        distance_m: f64,
        wavelength_m: f64,
        tx_aperture_m: f64,
        rx_aperture_m: f64,
        noise_power_w: f64,
    ) -> f64 {
        let divergence_loss = self.beam_divergence_loss(distance_m, wavelength_m, tx_aperture_m);
        
        // Receiver aperture area
        let rx_area = core::f64::consts::PI * (rx_aperture_m / 2.0).powi(2);
        
        // Fraction of power received
        let power_fraction = rx_area / divergence_loss;
        
        // Received signal power
        let received_power = transmit_power_w * power_fraction;
        
        // SNR
        if noise_power_w <= 0.0 {
            return f64::INFINITY;
        }
        received_power / noise_power_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_capacity() {
        let channel = ChannelCapacity::new(1e9, 100.0).unwrap();
        let capacity = channel.capacity_bps();
        
        // C = 1e9 * log2(101) ≈ 6.66 Gbps
        assert!(capacity > 6e9 && capacity < 7e9);
    }

    #[test]
    fn test_rs_params() {
        let rs = RSParams::standard();
        assert_eq!(rs.n, 255);
        assert_eq!(rs.k, 223);
        assert!(rs.code_rate() > 0.85);
        assert_eq!(rs.error_correction_symbols(), 16);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = ShannonHartleyEncoder::new(1e9, 100.0, FecScheme::ReedSolomon(RSParams::standard()));
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_beam_divergence() {
        let encoder = ShannonHartleyEncoder::new(1e9, 100.0, FecScheme::None).unwrap();
        
        // 1 micron wavelength, 1m aperture, 1 light year distance
        let wavelength = 1e-6;
        let aperture = 1.0;
        let distance = 9.461e15; // 1 light year in meters
        
        let loss = encoder.beam_divergence_loss(distance, wavelength, aperture);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }
}
