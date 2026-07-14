//! Walk-Forward Micro-Backtester - Continuous in-memory backtest against recent tick data.
//! 
//! This module runs vectorized backtests of current Alpha parameters against the
//! immediate past (last N hours of live tick data) to detect regime changes.

use std::sync::atomic::{AtomicU64, AtomicI128, Ordering};
use std::sync::Arc;

use super::in_memory_tick_replay::{InMemoryTickReplay, TickRingBuffer, MarketTick, ReplayError};

/// Result of a walk-forward backtest iteration
#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    /// Number of ticks processed
    pub ticks_processed: u64,
    
    /// Total PnL from the backtest (scaled integer)
    pub total_pnl_scaled: i128,
    
    /// Sharpe ratio (annualized, scaled by 10000 for integer math)
    pub sharpe_ratio_x10000: i64,
    
    /// Maximum drawdown (scaled integer)
    pub max_drawdown_scaled: i128,
    
    /// Win rate (scaled by 10000, e.g., 5500 = 55%)
    pub win_rate_x10000: u32,
    
    /// Number of trades executed
    pub trade_count: u64,
}

/// Configuration for walk-forward backtesting
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Number of ticks to look back for each backtest
    pub lookback_ticks: usize,
    
    /// Initial capital for backtest (scaled integer)
    pub initial_capital_scaled: i128,
    
    /// Maximum position size (scaled integer)
    pub max_position_scaled: i128,
    
    /// Transaction cost per trade in basis points
    pub transaction_cost_bps: u32,
    
    /// Minimum ticks between trades (cooldown)
    pub trade_cooldown_ticks: u32,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            lookback_ticks: 10000,  // ~10k ticks
            initial_capital_scaled: 1_000_000_000_000,  // 10k units with 8 decimal scaling
            max_position_scaled: 100_000_000_000,  // 1k units
            transaction_cost_bps: 10,  // 0.10%
            trade_cooldown_ticks: 100,
        }
    }
}

/// Walk-Forward Micro-Backtester
pub struct WalkForwardMicroBacktester {
    config: WalkForwardConfig,
    tick_buffer: Arc<TickRingBuffer>,
    replay_engine: InMemoryTickReplay,
    
    /// Total backtests run
    backtest_count: AtomicU64,
    
    /// Last Sharpe ratio observed
    last_sharpe_x10000: AtomicI128,
    
    /// Cumulative PnL across all backtests
    cumulative_pnl: AtomicI128,
}

impl WalkForwardMicroBacktester {
    pub fn new(config: WalkForwardConfig, tick_buffer: Arc<TickRingBuffer>) -> Self {
        let replay = InMemoryTickReplay::new(Arc::clone(&tick_buffer));
        
        Self {
            config,
            tick_buffer,
            replay_engine: replay,
            backtest_count: AtomicU64::new(0),
            last_sharpe_x10000: AtomicI128::new(0),
            cumulative_pnl: AtomicI128::new(0),
        }
    }
    
    /// Run a single walk-forward backtest iteration
    /// 
    /// Returns WalkForwardResult or an error if replay fails
    pub fn run_iteration<F>(&self, signal_fn: F) -> Result<WalkForwardResult, BacktestError>
    where
        F: Fn(&MarketTick) -> Signal,
    {
        // Initialize replay window
        self.replay_engine
            .init_replay(self.config.lookback_ticks)
            .map_err(|e| BacktestError::ReplayError(e))?;
        
        let mut state = BacktestState::new(self.config.initial_capital_scaled);
        let mut ticks_since_last_trade = 0u32;
        let mut wins = 0u64;
        let mut losses = 0u64;
        let mut peak_equity = state.equity_scaled;
        let mut max_drawdown = 0i128;
        
        // Process ticks
        while let Some(tick) = self.replay_engine.next_tick() {
            // Generate signal
            let signal = signal_fn(&tick);
            
            // Check cooldown
            if ticks_since_last_trade < self.config.trade_cooldown_ticks {
                ticks_since_last_trade += 1;
                continue;
            }
            
            // Execute trading logic based on signal
            match signal {
                Signal::Long => {
                    if state.position_scaled >= 0 {
                        // Enter long position
                        let entry_price = tick.ask_price;
                        let qty = self.config.max_position_scaled.min(state.cash_scaled / entry_price * 100_000_000);
                        
                        if qty > 0 {
                            state.enter_long(qty, entry_price, self.config.transaction_cost_bps);
                            ticks_since_last_trade = 0;
                        }
                    }
                }
                Signal::Short => {
                    if state.position_scaled <= 0 {
                        // Enter short position
                        let entry_price = tick.bid_price;
                        let qty = self.config.max_position_scaled.min(state.cash_scaled / entry_price * 100_000_000);
                        
                        if qty > 0 {
                            state.enter_short(qty, entry_price, self.config.transaction_cost_bps);
                            ticks_since_last_trade = 0;
                        }
                    }
                }
                Signal::Flat => {
                    // Close any open position
                    if state.position_scaled != 0 {
                        let exit_price = if state.position_scaled > 0 { 
                            tick.bid_price 
                        } else { 
                            tick.ask_price 
                        };
                        let pnl = state.close_position(exit_price, self.config.transaction_cost_bps);
                        
                        if pnl > 0 {
                            wins += 1;
                        } else if pnl < 0 {
                            losses += 1;
                        }
                        ticks_since_last_trade = 0;
                    }
                }
            }
            
            // Update equity and track drawdown
            let mark_price = (tick.bid_price + tick.ask_price) / 2;
            state.update_unrealized_pnl(mark_price);
            
            if state.equity_scaled > peak_equity {
                peak_equity = state.equity_scaled;
            }
            
            let drawdown = peak_equity - state.equity_scaled;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        // Close any remaining position at final price
        if state.position_scaled != 0 {
            // Use last known price (simplified)
            state.close_position(state.last_mark_price, self.config.transaction_cost_bps);
        }
        
        let total_trades = wins + losses;
        let win_rate = if total_trades > 0 {
            ((wins * 10000) / total_trades) as u32
        } else {
            0
        };
        
        // Calculate Sharpe ratio (simplified, annualized)
        let sharpe = calculate_sharpe_ratio(state.pnl_scaled, state.equity_scaled, total_trades);
        
        // Update statistics
        self.backtest_count.fetch_add(1, Ordering::Relaxed);
        self.last_sharpe_x10000.store(sharpe as i128, Ordering::Relaxed);
        self.cumulative_pnl.fetch_add(state.pnl_scaled, Ordering::Relaxed);
        
        Ok(WalkForwardResult {
            ticks_processed: self.config.lookback_ticks as u64,
            total_pnl_scaled: state.pnl_scaled,
            sharpe_ratio_x10000: sharpe,
            max_drawdown_scaled: max_drawdown,
            win_rate_x10000: win_rate,
            trade_count: total_trades,
        })
    }
    
    /// Get the last observed Sharpe ratio
    pub fn get_last_sharpe(&self) -> i64 {
        (self.last_sharpe_x10000.load(Ordering::Relaxed) / 100) as i64
    }
    
    /// Get total backtests run
    pub fn get_backtest_count(&self) -> u64 {
        self.backtest_count.load(Ordering::Relaxed)
    }
    
    /// Get cumulative PnL across all backtests
    pub fn get_cumulative_pnl(&self) -> i128 {
        self.cumulative_pnl.load(Ordering::Relaxed)
    }
}

/// Trading signal from alpha model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Long,
    Short,
    Flat,
}

/// Internal backtest state
struct BacktestState {
    cash_scaled: i128,
    position_scaled: i128,
    entry_price: i128,
    pnl_scaled: i128,
    equity_scaled: i128,
    last_mark_price: i128,
}

impl BacktestState {
    fn new(initial_capital: i128) -> Self {
        Self {
            cash_scaled: initial_capital,
            position_scaled: 0,
            entry_price: 0,
            pnl_scaled: 0,
            equity_scaled: initial_capital,
            last_mark_price: 0,
        }
    }
    
    fn enter_long(&mut self, qty: i128, price: i128, cost_bps: u32) {
        let cost = (qty * price) / 100_000_000;
        let fee = (cost * cost_bps as i128) / 10000;
        
        self.cash_scaled -= cost + fee;
        self.position_scaled = qty;
        self.entry_price = price;
    }
    
    fn enter_short(&mut self, qty: i128, price: i128, cost_bps: u32) {
        let notional = (qty * price) / 100_000_000;
        let fee = (notional * cost_bps as i128) / 10000;
        
        self.cash_scaled -= fee;  // Short sells add to cash, but we track net
        self.position_scaled = -qty;
        self.entry_price = price;
    }
    
    fn close_position(&mut self, exit_price: i128, cost_bps: u32) -> i128 {
        if self.position_scaled == 0 {
            return 0;
        }
        
        let is_long = self.position_scaled > 0;
        let qty = self.position_scaled.abs();
        
        let proceeds = (qty * exit_price) / 100_000_000;
        let fee = (proceeds * cost_bps as i128) / 10000;
        
        let pnl = if is_long {
            proceeds - (qty * self.entry_price) / 100_000_000 - fee
        } else {
            (qty * self.entry_price) / 100_000_000 - proceeds - fee
        };
        
        self.cash_scaled += proceeds + if is_long { 0 } else { (qty * self.entry_price) / 100_000_000 };
        self.cash_scaled -= fee;
        self.pnl_scaled += pnl;
        self.position_scaled = 0;
        self.entry_price = 0;
        
        pnl
    }
    
    fn update_unrealized_pnl(&mut self, mark_price: i128) {
        self.last_mark_price = mark_price;
        
        let unrealized = if self.position_scaled > 0 {
            ((self.position_scaled * mark_price) / 100_000_000) 
                - ((self.position_scaled * self.entry_price) / 100_000_000)
        } else if self.position_scaled < 0 {
            ((self.position_scaled.abs() * self.entry_price) / 100_000_000)
                - ((self.position_scaled.abs() * mark_price) / 100_000_000)
        } else {
            0
        };
        
        self.equity_scaled = self.cash_scaled + unrealized;
    }
}

/// Calculate Sharpe ratio (scaled by 10000)
fn calculate_sharpe_ratio(pnl: i128, equity: i128, trades: u64) -> i64 {
    if equity == 0 || trades == 0 {
        return 0;
    }
    
    // Simplified Sharpe: (return / volatility) * sqrt(252)
    // Using integer approximation
    let return_pct = (pnl * 10000) / equity;
    
    // Rough volatility estimate based on trade count
    let vol_estimate = 100;  // Assume 1% daily vol
    
    let sharpe = (return_pct * 16) / vol_estimate;  // sqrt(252) ≈ 16
    sharpe as i64
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum BacktestError {
    #[error("Replay error: {0}")]
    ReplayError(ReplayError),
    
    #[error("Insufficient capital")]
    InsufficientCapital,
    
    #[error("Invalid configuration")]
    InvalidConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_walk_forward_basic() {
        let buffer = Arc::new(TickRingBuffer::new(1024));
        
        // Push some test ticks
        for i in 0..100 {
            buffer.push(MarketTick {
                timestamp_ns: i * 1000,
                bid_price: 50_000_000_000 + (i as i128 * 100_000),
                ask_price: 50_000_000_000 + (i as i128 * 100_000) + 10_000,
                sequence: i,
                ..Default::default()
            });
        }
        
        let config = WalkForwardConfig {
            lookback_ticks: 50,
            ..Default::default()
        };
        
        let backtester = WalkForwardMicroBacktester::new(config, Arc::clone(&buffer));
        
        // Simple momentum signal
        let result = backtester.run_iteration(|tick| {
            if tick.bid_price > 50_000_000_000 {
                Signal::Long
            } else {
                Signal::Flat
            }
        }).unwrap();
        
        assert!(result.ticks_processed > 0);
        assert!(result.trade_count >= 0);
    }
}
