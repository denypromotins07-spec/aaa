//! Microfluidic Pump Controller for Bioreactor Perfusion
//! 
//! Interfaces with physical perfusion pumps to deliver biochemical
//! agents (dopamine, serotonin, cortisol analogues) to the organoid.

/// Maximum number of pump channels
pub const MAX_PUMP_CHANNELS: usize = 16;

/// Default flow rate limits (nL/min)
const MIN_FLOW_RATE: f32 = 0.1;
const MAX_FLOW_RATE: f32 = 1000.0;

/// Biochemical agent types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum BiochemicalAgent {
    Dopamine = 0,
    Serotonin = 1,
    Norepinephrine = 2,
    Cortisol = 3,
    Glutamate = 4,
    GABA = 5,
    Acetylcholine = 6,
    Saline = 7,
    Custom = 15,
}

/// Pump operation modes
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum PumpMode {
    Continuous = 0,
    Pulsed = 1,
    Gradient = 2,
    Bolus = 3,
}

/// Error types for pump operations
#[derive(Debug, Clone, Copy)]
pub enum PumpError {
    InvalidChannel,
    FlowRateOutOfRange,
    PressureFault,
    AirBubbleDetected,
    ReservoirEmpty,
    CommunicationFailure,
    NotInitialized,
}

/// Pump status flags
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct PumpStatus {
    /// Pump is running
    pub running: bool,
    /// Flow rate in nL/min
    pub flow_rate: f32,
    /// Cumulative volume delivered (nL)
    pub volume_delivered: f32,
    /// Back pressure reading (kPa)
    pub pressure: f32,
    /// Temperature (Celsius)
    pub temperature: f32,
    /// Air bubble detected
    pub air_detected: bool,
    /// Reservoir level (0-1)
    pub reservoir_level: f32,
    /// Error flags
    pub error_flags: u32,
}

impl Default for PumpStatus {
    fn default() -> Self {
        Self {
            running: false,
            flow_rate: 0.0,
            volume_delivered: 0.0,
            pressure: 0.0,
            temperature: 37.0, // Body temp
            air_detected: false,
            reservoir_level: 1.0,
            error_flags: 0,
        }
    }
}

/// Configuration for a single pump channel
#[repr(C, align(32))]
pub struct PumpConfig {
    /// Assigned biochemical agent
    pub agent: BiochemicalAgent,
    /// Target flow rate (nL/min)
    pub target_flow_rate: f32,
    /// Minimum allowed flow rate
    pub min_flow_rate: f32,
    /// Maximum allowed flow rate
    pub max_flow_rate: f32,
    /// Pulse period for pulsed mode (ms)
    pub pulse_period_ms: u32,
    /// Duty cycle for pulsed mode (0-1)
    pub duty_cycle: f32,
}

impl Default for PumpConfig {
    fn default() -> Self {
        Self {
            agent: BiochemicalAgent::Saline,
            target_flow_rate: 10.0,
            min_flow_rate: MIN_FLOW_RATE,
            max_flow_rate: MAX_FLOW_RATE,
            pulse_period_ms: 1000,
            duty_cycle: 0.5,
        }
    }
}

/// Digital Microfluidic Pump Controller
pub struct MicrofluidicPumpController {
    /// Pump configurations
    configs: [PumpConfig; MAX_PUMP_CHANNELS],
    /// Current pump statuses
    statuses: [PumpStatus; MAX_PUMP_CHANNELS],
    /// Number of active channels
    num_channels: usize,
    /// System enabled flag
    enabled: bool,
    /// Global emergency stop flag
    emergency_stop: bool,
    /// Last update timestamp (ns)
    last_update_ns: u64,
}

impl MicrofluidicPumpController {
    /// Create a new pump controller
    pub fn new(num_channels: usize) -> Self {
        let mut controller = Self {
            configs: [PumpConfig::default(); MAX_PUMP_CHANNELS],
            statuses: [PumpStatus::default(); MAX_PUMP_CHANNELS],
            num_channels: num_channels.min(MAX_PUMP_CHANNELS),
            enabled: false,
            emergency_stop: false,
            last_update_ns: 0,
        };

        // Initialize all channels with saline
        for i in 0..controller.num_channels {
            controller.configs[i].agent = BiochemicalAgent::Saline;
        }

        controller
    }

    /// Configure a pump channel
    pub fn configure_channel(
        &mut self,
        channel: usize,
        agent: BiochemicalAgent,
        flow_rate: f32,
    ) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        if flow_rate < MIN_FLOW_RATE || flow_rate > MAX_FLOW_RATE {
            return Err(PumpError::FlowRateOutOfRange);
        }

        let config = &mut self.configs[channel];
        config.agent = agent;
        config.target_flow_rate = flow_rate.clamp(MIN_FLOW_RATE, MAX_FLOW_RATE);

        Ok(())
    }

    /// Start a pump channel
    pub fn start_pump(&mut self, channel: usize) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        if self.emergency_stop {
            return Err(PumpError::CommunicationFailure); // Blocked by E-stop
        }

        let status = &mut self.statuses[channel];
        let config = &self.configs[channel];

        // Check prerequisites
        if status.reservoir_level < 0.1 {
            return Err(PumpError::ReservoirEmpty);
        }

        if status.air_detected {
            return Err(PumpError::AirBubbleDetected);
        }

        status.running = true;
        status.flow_rate = config.target_flow_rate;

        Ok(())
    }

    /// Stop a pump channel
    pub fn stop_pump(&mut self, channel: usize) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        let status = &mut self.statuses[channel];
        status.running = false;
        status.flow_rate = 0.0;

        Ok(())
    }

    /// Set flow rate for a channel
    pub fn set_flow_rate(&mut self, channel: usize, flow_rate: f32) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        if flow_rate < MIN_FLOW_RATE || flow_rate > MAX_FLOW_RATE {
            return Err(PumpError::FlowRateOutOfRange);
        }

        self.configs[channel].target_flow_rate = flow_rate;
        
        if self.statuses[channel].running {
            self.statuses[channel].flow_rate = flow_rate;
        }

        Ok(())
    }

    /// Get current status of a channel
    #[inline]
    pub fn get_status(&self, channel: usize) -> Option<&PumpStatus> {
        if channel < self.num_channels {
            Some(&self.statuses[channel])
        } else {
            None
        }
    }

    /// Update pump states (call periodically from main loop)
    pub fn update(&mut self, timestamp_ns: u64) {
        if self.emergency_stop {
            // All pumps stopped during E-stop
            for status in &mut self.statuses[..self.num_channels] {
                status.running = false;
                status.flow_rate = 0.0;
            }
            return;
        }

        let delta_ns = timestamp_ns.saturating_sub(self.last_update_ns);
        self.last_update_ns = timestamp_ns;

        // Update volume delivered for running pumps
        for status in &mut self.statuses[..self.num_channels] {
            if status.running && status.flow_rate > 0.0 {
                // Convert flow rate (nL/min) to volume (nL) based on time elapsed
                let delta_min = delta_ns as f32 / 60_000_000_000.0;
                status.volume_delivered += status.flow_rate * delta_min;
            }
        }
    }

    /// Trigger emergency stop (immediate halt of all pumps)
    pub fn emergency_stop(&mut self) {
        self.emergency_stop = true;
        self.enabled = false;
        
        // Immediately stop all pumps
        for status in &mut self.statuses[..self.num_channels] {
            status.running = false;
            status.flow_rate = 0.0;
        }
    }

    /// Reset from emergency stop (requires manual confirmation)
    pub fn reset_emergency_stop(&mut self, confirmation_code: u32) -> Result<(), PumpError> {
        if confirmation_code != 0xDEAD_BEEF {
            return Err(PumpError::CommunicationFailure);
        }
        
        self.emergency_stop = false;
        Ok(())
    }

    /// Simulate sensor readings (in production, read from hardware)
    pub fn update_sensor_readings(
        &mut self,
        channel: usize,
        pressure: f32,
        temperature: f32,
        reservoir_level: f32,
        air_detected: bool,
    ) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        let status = &mut self.statuses[channel];
        status.pressure = pressure;
        status.temperature = temperature;
        status.reservoir_level = reservoir_level.clamp(0.0, 1.0);
        status.air_detected = air_detected;

        // Check for fault conditions
        if pressure > 500.0 {
            status.error_flags |= 0x01; // High pressure flag
        }
        if temperature < 35.0 || temperature > 40.0 {
            status.error_flags |= 0x02; // Temp out of range
        }

        Ok(())
    }

    /// Deliver a bolus dose
    pub fn deliver_bolus(
        &mut self,
        channel: usize,
        volume_nl: f32,
        max_flow_rate: f32,
    ) -> Result<(), PumpError> {
        if channel >= self.num_channels {
            return Err(PumpError::InvalidChannel);
        }

        if volume_nl <= 0.0 || volume_nl > 100_000.0 {
            return Err(PumpError::FlowRateOutOfRange);
        }

        let status = &mut self.statuses[channel];
        if status.reservoir_level * 1_000_000.0 < volume_nl {
            return Err(PumpError::ReservoirEmpty);
        }

        // Calculate delivery time at max flow rate
        let flow_rate = max_flow_rate.min(MAX_FLOW_RATE).max(MIN_FLOW_RATE);
        let delivery_time_ms = (volume_nl / flow_rate * 60_000.0) as u32;

        // Start pump for calculated duration
        status.running = true;
        status.flow_rate = flow_rate;

        // In production, this would use a timer interrupt to stop after delivery_time_ms
        // For now, we just track the intended volume
        status.volume_delivered += volume_nl;

        Ok(())
    }

    /// Get total volume delivered across all channels
    pub fn total_volume_delivered(&self) -> f32 {
        self.statuses[..self.num_channels]
            .iter()
            .map(|s| s.volume_delivered)
            .sum()
    }

    /// Enable the system
    pub fn enable(&mut self) -> Result<(), PumpError> {
        if self.emergency_stop {
            return Err(PumpError::CommunicationFailure);
        }
        self.enabled = true;
        Ok(())
    }

    /// Check if system is ready
    pub fn is_ready(&self) -> bool {
        self.enabled 
            && !self.emergency_stop 
            && self.statuses.iter().all(|s| !s.air_detected && s.reservoir_level > 0.1)
    }
}

/// Perfusion protocol definitions
#[derive(Debug, Clone)]
pub struct PerfusionProtocol {
    /// Protocol name
    pub name: &'static str,
    /// Steps: (channel, flow_rate, duration_ms)
    pub steps: [(usize, f32, u32); 8],
    /// Number of valid steps
    pub num_steps: usize,
}

impl PerfusionProtocol {
    /// Create a dopamine perfusion protocol
    pub fn dopamine_protocol() -> Self {
        Self {
            name: "Dopamine Infusion",
            steps: [
                (0, 5.0, 60_000),   // Channel 0: 5 nL/min for 1 min
                (0, 10.0, 120_000), // Ramp up
                (0, 20.0, 300_000), // Maintain
                (0, 10.0, 60_000),  // Ramp down
                (0, 0.0, 0),
                (0, 0.0, 0),
                (0, 0.0, 0),
                (0, 0.0, 0),
            ],
            num_steps: 4,
        }
    }

    /// Create a cortisol analogue (stress response) protocol
    pub fn cortisol_protocol() -> Self {
        Self {
            name: "Cortisol Analogue",
            steps: [
                (3, 50.0, 30_000),  // Rapid bolus
                (3, 20.0, 60_000),  // Sustain
                (3, 10.0, 120_000), // Taper
                (3, 5.0, 60_000),   // Baseline
                (0, 0.0, 0),
                (0, 0.0, 0),
                (0, 0.0, 0),
                (0, 0.0, 0),
            ],
            num_steps: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pump_initialization() {
        let controller = MicrofluidicPumpController::new(8);
        assert!(!controller.emergency_stop);
        assert!(!controller.enabled);
    }

    #[test]
    fn test_pump_configuration() {
        let mut controller = MicrofluidicPumpController::new(8);
        
        let result = controller.configure_channel(0, BiochemicalAgent::Dopamine, 10.0);
        assert!(result.is_ok());
        
        // Test invalid flow rate
        let result = controller.configure_channel(0, BiochemicalAgent::Dopamine, 0.0);
        assert!(matches!(result, Err(PumpError::FlowRateOutOfRange)));
    }

    #[test]
    fn test_emergency_stop() {
        let mut controller = MicrofluidicPumpController::new(8);
        controller.enable().unwrap();
        controller.start_pump(0).unwrap();
        
        assert!(controller.statuses[0].running);
        
        controller.emergency_stop();
        
        assert!(!controller.statuses[0].running);
        assert!(controller.emergency_stop);
    }

    #[test]
    fn test_bolus_delivery() {
        let mut controller = MicrofluidicPumpController::new(8);
        controller.configure_channel(0, BiochemicalAgent::Dopamine, 10.0).unwrap();
        
        let initial_volume = controller.statuses[0].volume_delivered;
        controller.deliver_bolus(0, 100.0, 50.0).unwrap();
        
        assert!(controller.statuses[0].volume_delivered > initial_volume);
    }
}
