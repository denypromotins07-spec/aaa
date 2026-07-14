//! Backtest module - Walk-forward micro-backtesting and parameter decay scoring
pub mod in_memory_tick_replay;
pub mod walk_forward_micro_bt;
pub mod parameter_decay_scorer;

pub use in_memory_tick_replay::{InMemoryTickReplay, TickRingBuffer, MarketTick, ReplayError};
pub use walk_forward_micro_bt::{WalkForwardMicroBacktester, WalkForwardConfig, WalkForwardResult, Signal, BacktestError};
pub use parameter_decay_scorer::{ParameterDecayScorer, DecayScorerConfig, DecayState, DecayAction, DecayStats};
