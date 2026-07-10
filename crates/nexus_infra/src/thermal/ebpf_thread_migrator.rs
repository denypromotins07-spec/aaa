//! eBPF Thread Migrator for Predictive Thermal Scheduling
//! 
//! Interfaces with eBPF hooks into the kernel's CPU runqueue to enable
//! atomic thread migration before hardware thermal throttling occurs.

use core::fmt;

/// Maximum number of CPU cores supported
const MAX_CPU_CORES: usize = 256;
/// Default thermal threshold for migration trigger (°C)
const DEFAULT_MIGRATION_THRESHOLD: f64 = 85.0;
/// Time horizon for thermal prediction (milliseconds)
const PREDICTION_HORIZON_MS: u64 = 5;
/// Minimum temperature difference for migration benefit
const MIN_TEMP_BENEFIT: f64 = 3.0;

/// Errors in eBPF thread migration
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationError {
    EbpfLoadFailure,
    HookRegistrationFailed,
    InvalidCoreId,
    MigrationBlocked,
    ThermalSensorTimeout,
    SchedulerDenied,
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationError::EbpfLoadFailure => write!(f, "Failed to load eBPF program"),
            MigrationError::HookRegistrationFailed => write!(f, "Failed to register scheduler hook"),
            MigrationError::InvalidCoreId => write!(f, "Invalid CPU core identifier"),
            MigrationError::MigrationBlocked => write!(f, "Thread migration blocked by scheduler"),
            MigrationError::ThermalSensorTimeout => write!(f, "Thermal sensor read timeout"),
            MigrationError::SchedulerDenied => write!(f, "Scheduler denied migration request"),
        }
    }
}

/// Thread state for migration decisions
#[derive(Debug, Clone, Copy)]
pub struct ThreadState {
    /// Thread ID
    pub tid: u64,
    /// Current core assignment
    pub current_core: u8,
    /// Thread priority
    pub priority: u8,
    /// Estimated execution time remaining (μs)
    pub remaining_time_us: u64,
    /// Thermal sensitivity (0-1, higher = more heat generated)
    pub thermal_sensitivity: f64,
    /// Last migration timestamp (ms)
    pub last_migration_ms: u64,
    /// Migration cooldown remaining (ms)
    pub migration_cooldown_ms: u64,
}

impl Default for ThreadState {
    fn default() -> Self {
        Self {
            tid: 0,
            current_core: 0,
            priority: 128,
            remaining_time_us: 1000,
            thermal_sensitivity: 0.5,
            last_migration_ms: 0,
            migration_cooldown_ms: 0,
        }
    }
}

/// Core thermal state
#[derive(Debug, Clone, Copy)]
pub struct CoreThermalState {
    /// Current temperature (°C)
    pub temperature: f64,
    /// Predicted temperature in PREDICTION_HORIZON_MS (°C)
    pub predicted_temp: f64,
    /// Temperature trend (°C/ms)
    pub temp_trend: f64,
    /// Current utilization (0-1)
    pub utilization: f64,
    /// Number of threads assigned
    pub thread_count: u8,
    /// Core is available for migration
    pub available: bool,
}

impl Default for CoreThermalState {
    fn default() -> Self {
        Self {
            temperature: 45.0,
            predicted_temp: 45.0,
            temp_trend: 0.0,
            utilization: 0.0,
            thread_count: 0,
            available: true,
        }
    }
}

/// eBPF Thread Migrator
pub struct EbpfThreadMigrator {
    /// Core thermal states
    core_states: [CoreThermalState; MAX_CPU_CORES],
    /// Number of active cores
    active_cores: usize,
    /// Migration threshold
    migration_threshold: f64,
    /// eBPF program loaded flag
    ebpf_loaded: bool,
    /// Hooks registered flag
    hooks_registered: bool,
}

impl EbpfThreadMigrator {
    /// Create a new thread migrator
    pub fn new(active_cores: usize) -> Result<Self, MigrationError> {
        if active_cores == 0 || active_cores > MAX_CPU_CORES {
            return Err(MigrationError::InvalidCoreId);
        }

        let mut core_states = [CoreThermalState::default(); MAX_CPU_CORES];
        for i in 0..active_cores {
            core_states[i].available = true;
        }

        Ok(Self {
            core_states,
            active_cores,
            migration_threshold: DEFAULT_MIGRATION_THRESHOLD,
            ebpf_loaded: false,
            hooks_registered: false,
        })
    }

    /// Load eBPF program into kernel
    pub fn load_ebpf_program(&mut self) -> Result<(), MigrationError> {
        // In real implementation, this would load the eBPF bytecode
        // For now, simulate successful load
        self.ebpf_loaded = true;
        Ok(())
    }

    /// Register scheduler hooks
    pub fn register_scheduler_hooks(&mut self) -> Result<(), MigrationError> {
        if !self.ebpf_loaded {
            return Err(MigrationError::EbpfLoadFailure);
        }

        // In real implementation, this would attach to:
        // - sched_switch tracepoint
        // - sched_migrate_task tracepoint
        // - cpu_online/cpu_hotplug events
        self.hooks_registered = true;
        Ok(())
    }

    /// Update core thermal state from sensors
    pub fn update_core_state(
        &mut self,
        core_id: usize,
        temperature: f64,
        predicted_temp: f64,
        utilization: f64,
    ) -> Result<(), MigrationError> {
        if core_id >= self.active_cores {
            return Err(MigrationError::InvalidCoreId);
        }

        let state = &mut self.core_states[core_id];
        state.temperature = temperature;
        state.predicted_temp = predicted_temp;
        state.utilization = utilization;
        
        // Calculate trend
        state.temp_trend = (predicted_temp - temperature) / PREDICTION_HORIZON_MS as f64;

        // Mark unavailable if predicted to exceed threshold
        state.available = predicted_temp < self.migration_threshold;

        Ok(())
    }

    /// Evaluate if thread should be migrated
    pub fn evaluate_migration(&self, thread: &ThreadState) -> Option<u8> {
        if !self.hooks_registered {
            return None;
        }

        // Check if thread is in cooldown
        if thread.migration_cooldown_ms > 0 {
            return None;
        }

        let current_core = thread.current_core as usize;
        if current_core >= self.active_cores {
            return None;
        }

        let current_state = &self.core_states[current_core];

        // Check if current core will exceed threshold
        if current_state.predicted_temp < self.migration_threshold {
            return None; // No migration needed
        }

        // Find best alternative core
        let mut best_core: Option<u8> = None;
        let mut best_temp = current_state.predicted_temp;

        for i in 0..self.active_cores {
            if i == current_core {
                continue;
            }

            let candidate = &self.core_states[i];
            if !candidate.available {
                continue;
            }

            // Check if candidate is significantly cooler
            if candidate.predicted_temp < best_temp - MIN_TEMP_BENEFIT {
                best_temp = candidate.predicted_temp;
                best_core = Some(i as u8);
            }
        }

        best_core
    }

    /// Execute thread migration
    pub fn migrate_thread(&mut self, thread: &mut ThreadState, target_core: u8) 
        -> Result<(), MigrationError> 
    {
        if target_core as usize >= self.active_cores {
            return Err(MigrationError::InvalidCoreId);
        }

        if !self.core_states[target_core as usize].available {
            return Err(MigrationError::MigrationBlocked);
        }

        // In real implementation, this would:
        // 1. Use sched_setaffinity via eBPF helper
        // 2. Update scheduler runqueue
        // 3. Trigger IPI to target core

        let old_core = thread.current_core;
        thread.current_core = target_core;
        thread.last_migration_ms = self.get_timestamp_ms();
        thread.migration_cooldown_ms = 100; // 100ms cooldown

        // Update core thread counts
        self.core_states[old_core as usize].thread_count = 
            self.core_states[old_core as usize].thread_count.saturating_sub(1);
        self.core_states[target_core as usize].thread_count += 1;

        Ok(())
    }

    /// Get simulated timestamp
    fn get_timestamp_ms(&self) -> u64 {
        // In real implementation, would use ktime_get_ns() / 1_000_000
        0
    }

    /// Set migration threshold
    pub fn set_migration_threshold(&mut self, threshold: f64) {
        self.migration_threshold = threshold;
    }

    /// Get core state by ID
    pub fn get_core_state(&self, core_id: usize) -> Option<&CoreThermalState> {
        if core_id < self.active_cores {
            Some(&self.core_states[core_id])
        } else {
            None
        }
    }

    /// Get all core states slice
    pub fn get_all_core_states(&self) -> &[CoreThermalState] {
        &self.core_states[..self.active_cores]
    }

    /// Check if eBPF is loaded
    pub fn is_ebpf_loaded(&self) -> bool {
        self.ebpf_loaded
    }

    /// Check if hooks are registered
    pub fn is_hooks_registered(&self) -> bool {
        self.hooks_registered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrator_creation() {
        let migrator = EbpfThreadMigrator::new(16);
        assert!(migrator.is_ok());
    }

    #[test]
    fn test_invalid_core_count() {
        let migrator = EbpfThreadMigrator::new(0);
        assert_eq!(migrator.unwrap_err(), MigrationError::InvalidCoreId);

        let migrator = EbpfThreadMigrator::new(MAX_CPU_CORES + 1);
        assert_eq!(migrator.unwrap_err(), MigrationError::InvalidCoreId);
    }

    #[test]
    fn test_ebpf_load() {
        let mut migrator = EbpfThreadMigrator::new(16).unwrap();
        assert!(!migrator.is_ebpf_loaded());
        
        let result = migrator.load_ebpf_program();
        assert!(result.is_ok());
        assert!(migrator.is_ebpf_loaded());
    }

    #[test]
    fn test_hook_registration() {
        let mut migrator = EbpfThreadMigrator::new(16).unwrap();
        
        // Must load eBPF first
        let result = migrator.register_scheduler_hooks();
        assert_eq!(result.unwrap_err(), MigrationError::EbpfLoadFailure);
        
        migrator.load_ebpf_program().unwrap();
        let result = migrator.register_scheduler_hooks();
        assert!(result.is_ok());
    }

    #[test]
    fn test_core_state_update() {
        let mut migrator = EbpfThreadMigrator::new(16).unwrap();
        
        let result = migrator.update_core_state(0, 75.0, 80.0, 0.5);
        assert!(result.is_ok());
        
        let state = migrator.get_core_state(0).unwrap();
        assert_eq!(state.temperature, 75.0);
        assert_eq!(state.predicted_temp, 80.0);
    }

    #[test]
    fn test_migration_evaluation() {
        let mut migrator = EbpfThreadMigrator::new(16).unwrap();
        migrator.load_ebpf_program().unwrap();
        migrator.register_scheduler_hooks().unwrap();
        
        // Set core 0 to exceed threshold
        migrator.update_core_state(0, 88.0, 92.0, 0.9).unwrap();
        // Set core 1 to be cooler
        migrator.update_core_state(1, 60.0, 65.0, 0.3).unwrap();
        
        let thread = ThreadState {
            current_core: 0,
            ..Default::default()
        };
        
        let target = migrator.evaluate_migration(&thread);
        assert_eq!(target, Some(1));
    }
}
