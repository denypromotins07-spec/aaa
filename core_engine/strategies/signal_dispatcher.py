"""
Signal Dispatcher - Translates Conviction Scores to Order Commands

This module handles the translation of fused conviction scores from the
Rust engine into actionable order commands for NautilusTrader. Implements
intelligent position sizing and risk management based on conviction levels.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId


class SignalStrength(IntEnum):
    """Signal strength classification based on conviction."""
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class OrderSpecification:
    """
    Specification for an order to be submitted.
    
    Attributes
    ----------
    side : OrderSide
        The order side (BUY or SELL).
    quantity : int
        The order quantity in base units.
    conviction : float
        The conviction score that generated this order.
    signal_strength : SignalStrength
        Classified signal strength.
    ts_event : int
        Event timestamp in nanoseconds.
    metadata : Dict[str, Any]
        Additional metadata for the order.
    """
    side: OrderSide
    quantity: int
    conviction: float
    signal_strength: SignalStrength
    ts_event: int
    metadata: Dict[str, Any]


class SignalDispatcher:
    """
    Dispatches trading signals based on conviction scores.
    
    Translates conviction scores (-1.0 to +1.0) into order specifications
    with appropriate position sizing and risk management.
    """
    
    # Conviction thresholds for signal strength classification
    THRESHOLDS = {
        SignalStrength.WEAK: 0.3,
        SignalStrength.MODERATE: 0.5,
        SignalStrength.STRONG: 0.7,
        SignalStrength.VERY_STRONG: 0.85,
    }
    
    # Position size multipliers by signal strength
    SIZE_MULTIPLIERS = {
        SignalStrength.NONE: 0.0,
        SignalStrength.WEAK: 0.25,
        SignalStrength.MODERATE: 0.50,
        SignalStrength.STRONG: 0.75,
        SignalStrength.VERY_STRONG: 1.0,
    }
    
    def __init__(
        self,
        conviction_threshold: float = 0.6,
        max_size: int = 1000,
        min_size: int = 10,
        risk_per_trade: float = 0.02,
    ) -> None:
        """
        Initialize the signal dispatcher.
        
        Parameters
        ----------
        conviction_threshold : float, default 0.6
            Minimum absolute conviction to generate a signal.
        max_size : int, default 1000
            Maximum position size in base units.
        min_size : int, default 10
            Minimum position size in base units.
        risk_per_trade : float, default 0.02
            Risk per trade as fraction of portfolio (0.02 = 2%).
        """
        self._conviction_threshold = conviction_threshold
        self._max_size = max_size
        self._min_size = min_size
        self._risk_per_trade = risk_per_trade
        
        # Track recent signals for analysis
        self._signal_history: list[Dict[str, Any]] = []
        
        # Current positions by instrument (simplified tracking)
        self._positions: Dict[InstrumentId, Dict[str, Any]] = {}
        
    def classify_signal_strength(self, conviction: float) -> SignalStrength:
        """
        Classify signal strength based on conviction magnitude.
        
        Parameters
        ----------
        conviction : float
            Conviction score (-1.0 to +1.0).
            
        Returns
        -------
        SignalStrength
            Classified signal strength.
        """
        abs_conviction = abs(conviction)
        
        if abs_conviction >= self.THRESHOLDS[SignalStrength.VERY_STRONG]:
            return SignalStrength.VERY_STRONG
        elif abs_conviction >= self.THRESHOLDS[SignalStrength.STRONG]:
            return SignalStrength.STRONG
        elif abs_conviction >= self.THRESHOLDS[SignalStrength.MODERATE]:
            return SignalStrength.MODERATE
        elif abs_conviction >= self.THRESHOLDS[SignalStrength.WEAK]:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NONE
            
    def calculate_position_size(
        self,
        conviction: float,
        signal_strength: SignalStrength,
        current_position: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Calculate position size based on conviction and signal strength.
        
        Parameters
        ----------
        conviction : float
            Conviction score.
        signal_strength : SignalStrength
            Classified signal strength.
        current_position : dict, optional
            Current position information.
            
        Returns
        -------
        int
            Position size in base units.
        """
        # Base size from signal strength multiplier
        base_multiplier = self.SIZE_MULTIPLIERS[signal_strength]
        
        # Adjust by conviction magnitude within strength class
        threshold = self.THRESHOLDS.get(signal_strength, 0.0)
        next_threshold = self.THRESHOLDS.get(
            SignalStrength(signal_strength.value + 1) if signal_strength.value < 4 else signal_strength,
            1.0
        )
        
        # Interpolate within the strength band
        if next_threshold > threshold:
            conviction_factor = (abs(conviction) - threshold) / (next_threshold - threshold)
            conviction_factor = max(0.0, min(1.0, conviction_factor))
        else:
            conviction_factor = 1.0
            
        # Calculate raw size
        raw_size = self._max_size * base_multiplier * (0.5 + 0.5 * conviction_factor)
        
        # Apply risk adjustment based on existing position
        if current_position:
            existing_size = current_position.get("size", 0)
            existing_side = current_position.get("side")
            
            # Determine if we're adding to or reducing position
            new_side = OrderSide.BUY if conviction > 0 else OrderSide.SELL
            
            if existing_side == new_side:
                # Adding to existing position - cap total
                remaining_capacity = self._max_size - existing_size
                raw_size = min(raw_size, remaining_capacity)
            else:
                # Reducing or reversing position
                raw_size = min(raw_size, existing_size * 1.5)  # Allow up to 150% reversal
                
        # Enforce minimum size
        if raw_size < self._min_size and raw_size > 0:
            raw_size = self._min_size
            
        # Round to integer and enforce maximum
        size = int(round(raw_size))
        return max(0, min(size, self._max_size))
        
    def generate_order(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        conviction: float,
        current_position: Optional[Dict[str, Any]] = None,
        ts_event: int = 0,
    ) -> Optional[OrderSpecification]:
        """
        Generate an order specification from a trading signal.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to trade.
        side : OrderSide
            The order side (BUY or SELL).
        conviction : float
            The conviction score.
        current_position : dict, optional
            Current position information.
        ts_event : int, default 0
            Event timestamp in nanoseconds.
            
        Returns
        -------
        OrderSpecification, optional
            Order specification, or None if signal is too weak.
        """
        # Check conviction threshold
        if abs(conviction) < self._conviction_threshold:
            return None
            
        # Classify signal strength
        signal_strength = self.classify_signal_strength(conviction)
        
        if signal_strength == SignalStrength.NONE:
            return None
            
        # Calculate position size
        quantity = self.calculate_position_size(
            conviction=conviction,
            signal_strength=signal_strength,
            current_position=current_position,
        )
        
        if quantity <= 0:
            return None
            
        # Create order specification
        spec = OrderSpecification(
            side=side,
            quantity=quantity,
            conviction=conviction,
            signal_strength=signal_strength,
            ts_event=ts_event,
            metadata={
                "instrument_id": str(instrument_id),
                "signal_strength_name": signal_strength.name,
                "threshold_used": self._conviction_threshold,
                "risk_per_trade": self._risk_per_trade,
            },
        )
        
        # Record signal for analysis
        self._record_signal(instrument_id, spec)
        
        return spec
        
    def _record_signal(
        self,
        instrument_id: InstrumentId,
        spec: OrderSpecification,
    ) -> None:
        """Record signal for performance analysis."""
        record = {
            "instrument_id": str(instrument_id),
            "side": spec.side.name,
            "quantity": spec.quantity,
            "conviction": spec.conviction,
            "signal_strength": spec.signal_strength.name,
            "ts_event": spec.ts_event,
        }
        
        self._signal_history.append(record)
        
        # Keep history bounded
        if len(self._signal_history) > 10000:
            self._signal_history = self._signal_history[-5000:]
            
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about dispatched signals.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing signal statistics.
        """
        if not self._signal_history:
            return {
                "total_signals": 0,
                "avg_conviction": 0.0,
                "avg_quantity": 0.0,
                "strength_distribution": {},
            }
            
        buy_signals = [s for s in self._signal_history if s["side"] == "BUY"]
        sell_signals = [s for s in self._signal_history if s["side"] == "SELL"]
        
        # Strength distribution
        strength_dist = {}
        for strength in SignalStrength:
            count = sum(
                1 for s in self._signal_history 
                if s["signal_strength"] == strength.name
            )
            strength_dist[strength.name] = count
            
        return {
            "total_signals": len(self._signal_history),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_conviction": sum(s["conviction"] for s in self._signal_history) / len(self._signal_history),
            "avg_quantity": sum(s["quantity"] for s in self._signal_history) / len(self._signal_history),
            "strength_distribution": strength_dist,
        }
        
    def update_position(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        size: int,
    ) -> None:
        """
        Update tracked position for an instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument identifier.
        side : OrderSide
            The position side.
        size : int
            The position size.
        """
        if size <= 0:
            self._positions.pop(instrument_id, None)
        else:
            self._positions[instrument_id] = {
                "side": side,
                "size": size,
            }
            
    def get_current_position(self, instrument_id: InstrumentId) -> Optional[Dict[str, Any]]:
        """
        Get current tracked position for an instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument identifier.
            
        Returns
        -------
        dict, optional
            Position information, or None if no position.
        """
        return self._positions.get(instrument_id)
        
    def reset(self) -> None:
        """Reset the dispatcher state."""
        self._signal_history.clear()
        self._positions.clear()
