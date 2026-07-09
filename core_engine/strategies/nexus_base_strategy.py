"""
Nexus Base Strategy - NautilusTrader Integration

This module provides the NexusBaseStrategy class that inherits from NautilusTrader's
Strategy class and integrates with the Rust alpha engine via zero-copy PyO3 buffers.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any
from collections import deque

from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.live.config import LiveConfig
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId, StrategyId, PositionId
from nautilus_trader.model.orders import MarketOrder, LimitOrder
from nautilus_trader.strategy.strategy import Strategy

from core_engine.strategies.rust_alpha_bridge import RustAlphaBridge
from core_engine.strategies.signal_dispatcher import SignalDispatcher


class NexusBaseStrategy(Strategy):
    """
    Base strategy class for NEXUS-OMEGA trading system.
    
    Integrates with the Rust alpha engine to receive conviction scores
    and execute trades based on fused signals. Designed for minimal GIL
    contention by using zero-copy buffers for data exchange.
    """

    def __init__(
        self,
        config: LiveConfig,
        instrument_ids: list[InstrumentId],
        conviction_threshold: float = 0.6,
        max_position_size: int = 1000,
        rust_engine_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the Nexus strategy.
        
        Parameters
        ----------
        config : LiveConfig
            Configuration for the strategy.
        instrument_ids : list[InstrumentId]
            List of instrument IDs to trade.
        conviction_threshold : float, default 0.6
            Minimum conviction score required to execute a trade.
        max_position_size : int, default 1000
            Maximum position size in base units.
        rust_engine_path : str, optional
            Path to the Rust alpha engine shared library.
        """
        super().__init__(config=config)
        
        PyCondition.not_empty(instrument_ids, "instrument_ids")
        PyCondition.positive_float(conviction_threshold, "conviction_threshold")
        PyCondition.positive_int(max_position_size, "max_position_size")
        
        self._instrument_ids = instrument_ids
        self._conviction_threshold = conviction_threshold
        self._max_position_size = max_position_size
        
        # Rust alpha engine bridge
        self._rust_bridge: Optional[RustAlphaBridge] = None
        if rust_engine_path:
            self._rust_bridge = RustAlphaBridge(rust_engine_path)
        
        # Signal dispatcher for order generation
        self._dispatcher = SignalDispatcher(
            conviction_threshold=conviction_threshold,
            max_size=max_position_size,
        )
        
        # Conviction score history for analysis
        self._conviction_history: deque[float] = deque(maxlen=1000)
        
        # Current conviction scores per instrument
        self._current_conviction: Dict[InstrumentId, float] = {}
        
        # Active positions
        self._positions: Dict[InstrumentId, Any] = {}
        
        # Performance metrics
        self._trades_executed = 0
        self._total_pnl = 0.0
        
    def on_start(self) -> None:
        """
        Actions to perform when the strategy is started.
        """
        self.log.info(f"Starting NexusBaseStrategy with {len(self._instrument_ids)} instruments")
        
        # Subscribe to market data for all instruments
        for instrument_id in self._instrument_ids:
            self.subscribe_quote_ticks(instrument_id)
            self.subscribe_trade_ticks(instrument_id)
            
        # Initialize Rust bridge if available
        if self._rust_bridge:
            self._rust_bridge.initialize()
            self.log.info("Rust alpha engine initialized")
            
    def on_stop(self) -> None:
        """
        Actions to perform when the strategy is stopped.
        """
        self.log.info("Stopping NexusBaseStrategy")
        
        # Close any open positions (optional, configurable)
        # for position in self.positions:
        #     self.close_position(position)
            
        if self._rust_bridge:
            self._rust_bridge.shutdown()
            
    def on_resume(self) -> None:
        """
        Actions to perform when the strategy is resumed.
        """
        self.log.info("Resuming NexusBaseStrategy")
        
    def on_reset(self) -> None:
        """
        Actions to perform when the strategy is reset.
        """
        self._conviction_history.clear()
        self._current_conviction.clear()
        self._positions.clear()
        self._trades_executed = 0
        self._total_pnl = 0.0
        self.log.info("Reset NexusBaseStrategy")
        
    def on_instrument(self, instrument: Any) -> None:
        """
        Actions to perform when an instrument is received.
        """
        pass
        
    def on_data(self, data: Any) -> None:
        """
        Actions to perform when data is received.
        """
        pass
        
    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to perform when a quote tick is received.
        
        This is the primary entry point for processing market data
        and receiving alpha signals from the Rust engine.
        """
        if self._rust_bridge is None:
            return
            
        # Process tick through Rust engine (zero-copy)
        conviction_result = self._rust_bridge.process_quote_tick(
            instrument_id=tick.instrument_id,
            bid_price=tick.bid_price,
            ask_price=tick.ask_price,
            bid_size=tick.bid_size,
            ask_size=tick.ask_size,
            ts_event=tick.ts_event,
        )
        
        if conviction_result is not None:
            # Update current conviction
            self._current_conviction[tick.instrument_id] = conviction_result.conviction
            self._conviction_history.append(conviction_result.conviction)
            
            # Check if we should act on this signal
            self._evaluate_signal(
                instrument_id=tick.instrument_id,
                conviction=conviction_result.conviction,
                conviction_std=conviction_result.conviction_std,
                regime=conviction_result.regime,
                ts_event=tick.ts_event,
            )
            
    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Actions to perform when a trade tick is received.
        """
        if self._rust_bridge is None:
            return
            
        # Process trade through Rust engine
        conviction_result = self._rust_bridge.process_trade_tick(
            instrument_id=tick.instrument_id,
            price=tick.price,
            size=tick.size,
            aggressor_side=tick.aggressor_side,
            ts_event=tick.ts_event,
        )
        
        if conviction_result is not None:
            self._current_conviction[tick.instrument_id] = conviction_result.conviction
            
            # Evaluate signal with slightly different logic for trades
            self._evaluate_signal(
                instrument_id=tick.instrument_id,
                conviction=conviction_result.conviction,
                conviction_std=conviction_result.conviction_std,
                regime=conviction_result.regime,
                ts_event=tick.ts_event,
            )
            
    def on_bar(self, bar: Bar) -> None:
        """
        Actions to perform when a bar is received.
        """
        # Bars can be used for slower-timeframe signals
        pass
        
    def on_order_filled(self, event: OrderFilled) -> None:
        """
        Actions to perform when an order is filled.
        """
        self._trades_executed += 1
        
        # Update PnL tracking
        if event.pnl is not None:
            self._total_pnl += float(event.pnl)
            
        self.log.info(
            f"Order filled: {event.order_id}, pnl={event.pnl}, "
            f"total_trades={self._trades_executed}, total_pnl={self._total_pnl:.2f}"
        )
        
    def _evaluate_signal(
        self,
        instrument_id: InstrumentId,
        conviction: float,
        conviction_std: float,
        regime: int,
        ts_event: int,
    ) -> None:
        """
        Evaluate whether to execute a trade based on conviction score.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument being evaluated.
        conviction : float
            The conviction score (-1.0 to +1.0).
        conviction_std : float
            Standard deviation of the conviction estimate.
        regime : int
            Current market regime (0-4).
        ts_event : int
            Event timestamp in nanoseconds.
        """
        # Calculate z-score adjusted conviction
        zscore_threshold = self._conviction_threshold
        if conviction_std > 0:
            z_score = abs(conviction) / conviction_std
            # Require higher raw conviction if uncertainty is high
            effective_threshold = zscore_threshold * (1.0 + 1.0 / (z_score + 0.1))
        else:
            effective_threshold = zscore_threshold
            
        # Check if conviction exceeds threshold
        if abs(conviction) < effective_threshold:
            return
            
        # Determine order side
        if conviction > 0:
            side = OrderSide.BUY
        elif conviction < 0:
            side = OrderSide.SELL
        else:
            return
            
        # Get current position for this instrument
        current_position = self._positions.get(instrument_id)
        
        # Generate order via dispatcher
        order_spec = self._dispatcher.generate_order(
            instrument_id=instrument_id,
            side=side,
            conviction=conviction,
            current_position=current_position,
            ts_event=ts_event,
        )
        
        if order_spec is not None:
            self._submit_order(order_spec, instrument_id)
            
    def _submit_order(self, order_spec: Dict[str, Any], instrument_id: InstrumentId) -> None:
        """
        Submit an order to the market.
        
        Parameters
        ----------
        order_spec : Dict[str, Any]
            Order specification from the dispatcher.
        instrument_id : InstrumentId
            The instrument to trade.
        """
        side = order_spec["side"]
        quantity = order_spec["quantity"]
        
        # Create market order (can be extended to limit orders)
        order = MarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.strategy_id,
            instrument_id=instrument_id,
            side=side,
            quantity=self._make_qty(quantity),
            init_id=self._generate_order_id(),
            ts_init=self.clock.timestamp_ns(),
        )
        
        self.submit_order(order)
        
        self.log.info(
            f"Submitted {side} order for {quantity} units of {instrument_id}, "
            f"conviction={order_spec['conviction']:.3f}"
        )
        
    def get_current_conviction(self, instrument_id: InstrumentId) -> float:
        """
        Get the current conviction score for an instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to query.
            
        Returns
        -------
        float
            Current conviction score, or 0.0 if not available.
        """
        return self._current_conviction.get(instrument_id, 0.0)
        
    def get_conviction_history(self) -> list[float]:
        """
        Get the history of conviction scores.
        
        Returns
        -------
        list[float]
            List of historical conviction scores.
        """
        return list(self._conviction_history)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance statistics.
        """
        return {
            "trades_executed": self._trades_executed,
            "total_pnl": self._total_pnl,
            "avg_conviction": sum(self._conviction_history) / len(self._conviction_history) 
                if self._conviction_history else 0.0,
            "instruments": [str(i) for i in self._instrument_ids],
        }
