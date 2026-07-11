//! Discrete Lattice Mapper
//! 
//! Models the exchange matching engine as a 2D discrete spacetime grid (Cellular Automata).
//! Maps price (x-axis) and time (t-axis) onto a lattice with finite resolution.

use core::fmt;

/// Represents a cell in the discrete spacetime lattice of the matching engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeCell {
    /// Price index (tick number from reference)
    pub price_idx: i64,
    /// Time index (clock cycle from reference)
    pub time_idx: i64,
    /// Order count at this cell
    pub order_count: u32,
    /// Volume at this cell
    pub volume: i64,
}

impl LatticeCell {
    pub const fn new(price_idx: i64, time_idx: i64) -> Self {
        Self {
            price_idx,
            time_idx,
            order_count: 0,
            volume: 0,
        }
    }

    #[inline]
    pub fn add_order(&mut self, size: i64) {
        self.order_count = self.order_count.saturating_add(1);
        self.volume = self.volume.saturating_add(size);
    }
}

/// Configuration for the discrete lattice mapper
#[derive(Debug, Clone, Copy)]
pub struct LatticeConfig {
    /// Minimum price increment (tick size) in base currency units
    pub tick_size: f64,
    /// Minimum time increment (clock cycle) in nanoseconds
    pub time_granularity_ns: u64,
    /// Reference price for lattice origin
    pub reference_price: f64,
    /// Reference timestamp for lattice origin (nanos since epoch)
    pub reference_time_ns: u64,
    /// Maximum lattice dimensions to prevent OOM
    pub max_price_cells: usize,
    pub max_time_cells: usize,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            time_granularity_ns: 100, // 100ns typical FPGA cycle
            reference_price: 100.0,
            reference_time_ns: 0,
            max_price_cells: 10_000,
            max_time_cells: 10_000,
        }
    }
}

/// Maps continuous market data to discrete lattice coordinates
pub struct DiscreteLatticeMapper {
    config: LatticeConfig,
}

impl DiscreteLatticeMapper {
    pub const fn new(config: LatticeConfig) -> Self {
        Self { config }
    }

    /// Convert a continuous price to a discrete lattice index
    #[inline]
    pub fn price_to_index(&self, price: f64) -> Result<i64, LatticeError> {
        let delta = price - self.config.reference_price;
        let idx = (delta / self.config.tick_size).round() as i64;
        
        if idx.unsigned_abs() >= self.config.max_price_cells {
            return Err(LatticeError::IndexOutOfBounds);
        }
        Ok(idx)
    }

    /// Convert a lattice index back to continuous price
    #[inline]
    pub fn index_to_price(&self, idx: i64) -> f64 {
        self.config.reference_price + (idx as f64 * self.config.tick_size)
    }

    /// Convert a continuous timestamp to discrete lattice time index
    #[inline]
    pub fn time_to_index(&self, timestamp_ns: u64) -> Result<i64, LatticeError> {
        if timestamp_ns < self.config.reference_time_ns {
            return Err(LatticeError::TimestampBeforeReference);
        }
        
        let delta_ns = timestamp_ns - self.config.reference_time_ns;
        let idx = (delta_ns / self.config.time_granularity_ns) as i64;
        
        if idx.unsigned_abs() >= self.config.max_time_cells {
            return Err(LatticeError::IndexOutOfBounds);
        }
        Ok(idx)
    }

    /// Convert lattice time index back to continuous timestamp
    #[inline]
    pub fn index_to_time(&self, idx: i64) -> Option<u64> {
        if idx < 0 {
            return None;
        }
        self.config
            .reference_time_ns
            .checked_add((idx as u64) * self.config.time_granularity_ns)
    }

    /// Map a market event (price, time, volume) to lattice coordinates
    #[inline]
    pub fn map_event(
        &self,
        price: f64,
        timestamp_ns: u64,
        volume: i64,
    ) -> Result<LatticeCell, LatticeError> {
        let price_idx = self.price_to_index(price)?;
        let time_idx = self.time_to_index(timestamp_ns)?;
        
        let mut cell = LatticeCell::new(price_idx, time_idx);
        cell.add_order(volume);
        
        Ok(cell)
    }

    /// Detect cellular automata transition rules from observed lattice states
    /// Returns the CA rule number (Wolfram notation) if a match is found
    pub fn detect_ca_rule(&self, prev_state: &[LatticeCell], curr_state: &[LatticeCell]) -> Option<u8> {
        if prev_state.is_empty() || curr_state.is_empty() {
            return None;
        }
        
        // Simplified CA rule detection based on neighbor patterns
        // In production, this would use full neighborhood analysis
        let mut rule_candidates = [0u8; 256];
        let mut valid_transitions = 0usize;
        
        for window in prev_state.windows(3) {
            let left = window[0].order_count > 0;
            let center = window[1].order_count > 0;
            let right = window[2].order_count > 0;
            
            let pattern = (left as u8) << 2 | (center as u8) << 1 | (right as u8);
            
            // Find corresponding state in current slice
            if let Some(&curr) = curr_state.get(window[1].time_idx as usize) {
                let next_state = curr.order_count > 0;
                rule_candidates[pattern as usize] += if next_state { 1 } else { 0 };
                valid_transitions += 1;
            }
        }
        
        if valid_transitions == 0 {
            return None;
        }
        
        // Determine most likely rule
        let mut rule_bits = 0u8;
        for (i, &count) in rule_candidates.iter().enumerate() {
            if count > (valid_transitions / 2) as u32 {
                rule_bits |= 1 << i;
            }
        }
        
        Some(rule_bits)
    }

    /// Calculate the "spacetime interval" between two lattice events
    /// Uses discrete metric: ds² = c²*dt² - dx² (with c = 1 tick per time_unit)
    pub fn spacetime_interval(&self, cell1: &LatticeCell, cell2: &LatticeCell) -> i128 {
        let dt = (cell2.time_idx - cell1.time_idx) as i128;
        let dx = (cell2.price_idx - cell1.price_idx) as i128;
        
        // Discrete interval (simplified, c=1)
        dt.pow(2) - dx.pow(2)
    }
}

/// Errors that can occur during lattice mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeError {
    IndexOutOfBounds,
    TimestampBeforeReference,
    InvalidPrice,
    Overflow,
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::IndexOutOfBounds => write!(f, "Lattice index out of bounds"),
            LatticeError::TimestampBeforeReference => write!(f, "Timestamp before reference time"),
            LatticeError::InvalidPrice => write!(f, "Invalid price value"),
            LatticeError::Overflow => write!(f, "Arithmetic overflow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_mapping() {
        let config = LatticeConfig::default();
        let mapper = DiscreteLatticeMapper::new(config);
        
        let price = 100.05;
        let idx = mapper.price_to_index(price).unwrap();
        assert_eq!(mapper.index_to_price(idx), price);
    }

    #[test]
    fn test_spacetime_interval() {
        let config = LatticeConfig::default();
        let mapper = DiscreteLatticeMapper::new(config);
        
        let cell1 = LatticeCell::new(0, 0);
        let cell2 = LatticeCell::new(3, 5);
        
        // ds² = 5² - 3² = 25 - 9 = 16
        let interval = mapper.spacetime_interval(&cell1, &cell2);
        assert_eq!(interval, 16);
    }
}
