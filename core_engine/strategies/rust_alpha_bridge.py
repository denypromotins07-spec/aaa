"""
Rust Alpha Bridge - Zero-Copy FFI Interface to Rust Engine

This module provides the bridge between Python/NautilusTrader and the Rust
alpha engine using PyO3 for zero-copy data exchange. Minimizes GIL contention
by batching operations and using shared memory buffers.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum

from nautilus_trader.model.identifiers import InstrumentId


# Conviction result structure (matches Rust ConvictionResult)
@dataclass
class ConvictionResult:
    """Result from Rust conviction calculation."""
    conviction: float
    conviction_std: float
    effective_leverage: float
    regime: int
    num_signals: int
    ts: int


class MarketRegime(IntEnum):
    """Market regime identifiers matching Rust enum."""
    MEAN_REVERTING = 0
    TRENDING = 1
    HIGH_VOLATILITY = 2
    HIGH_TOXICITY = 3
    LOW_LIQUIDITY = 4


class RustAlphaBridge:
    """
    Bridge to Rust alpha engine via PyO3 FFI.
    
    Provides zero-copy access to the Rust alpha calculation engine,
    allowing high-throughput processing of market data with minimal
    Python GIL overhead.
    """
    
    def __init__(self, library_path: str) -> None:
        """
        Initialize the Rust bridge.
        
        Parameters
        ----------
        library_path : str
            Path to the compiled Rust shared library (.so or .dll).
        """
        self._library_path = library_path
        self._lib: Optional[ctypes.CDLL] = None
        self._engine_ptr: Optional[ctypes.c_void_p] = None
        self._initialized = False
        
        # Configure logging
        self._log = logging.getLogger(__name__)
        
        # Result buffer for zero-copy reads
        self._result_buffer: Optional[ctypes.c_char_p] = None
        
    def initialize(self) -> bool:
        """
        Initialize the Rust engine.
        
        Returns
        -------
        bool
            True if initialization succeeded.
        """
        if self._initialized:
            return True
            
        try:
            # Load the Rust library
            self._lib = ctypes.CDLL(self._library_path)
            
            # Set up function signatures
            self._setup_function_signatures()
            
            # Create engine instance
            self._engine_ptr = self._lib.nexus_engine_create()
            
            if not self._engine_ptr:
                self._log.error("Failed to create Rust engine instance")
                return False
                
            self._initialized = True
            self._log.info(f"Rust engine initialized from {self._library_path}")
            return True
            
        except Exception as e:
            self._log.error(f"Failed to initialize Rust engine: {e}")
            return False
            
    def _setup_function_signatures(self) -> None:
        """Set up ctypes function signatures for the Rust library."""
        if self._lib is None:
            return
            
        # nexus_engine_create() -> *mut NexusEngine
        self._lib.nexus_engine_create.restype = ctypes.c_void_p
        
        # nexus_engine_destroy(*mut NexusEngine)
        self._lib.nexus_engine_destroy.argtypes = [ctypes.c_void_p]
        
        # nexus_process_quote_tick(...) -> ConvictionResult
        self._lib.nexus_process_quote_tick.argtypes = [
            ctypes.c_void_p,  # engine pointer
            ctypes.c_uint64,  # instrument_id hash
            ctypes.c_double,  # bid_price
            ctypes.c_double,  # ask_price
            ctypes.c_double,  # bid_size
            ctypes.c_double,  # ask_size
            ctypes.c_uint64,  # ts_event
        ]
        self._lib.nexus_process_quote_tick.restype = ConvictionResultStruct
        
        # nexus_process_trade_tick(...) -> ConvictionResult
        self._lib.nexus_process_trade_tick.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_double,  # price
            ctypes.c_double,  # size
            ctypes.c_uint8,   # aggressor_side
            ctypes.c_uint64,
        ]
        self._lib.nexus_process_trade_tick.restype = ConvictionResultStruct
        
        # nexus_get_conviction_zscore(...) -> c_double
        self._lib.nexus_get_conviction_zscore.argtypes = [ctypes.c_void_p]
        self._lib.nexus_get_conviction_zscore.restype = ctypes.c_double
        
    def shutdown(self) -> None:
        """Shutdown the Rust engine and release resources."""
        if not self._initialized:
            return
            
        try:
            if self._engine_ptr and self._lib:
                self._lib.nexus_engine_destroy(self._engine_ptr)
                self._engine_ptr = None
                
            self._initialized = False
            self._log.info("Rust engine shut down")
            
        except Exception as e:
            self._log.error(f"Error shutting down Rust engine: {e}")
            
    def process_quote_tick(
        self,
        instrument_id: InstrumentId,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        ts_event: int,
    ) -> Optional[ConvictionResult]:
        """
        Process a quote tick through the Rust engine.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument identifier.
        bid_price : float
            Bid price.
        ask_price : float
            Ask price.
        bid_size : float
            Bid size.
        ask_size : float
            Ask size.
        ts_event : int
            Event timestamp in nanoseconds.
            
        Returns
        -------
        ConvictionResult, optional
            The conviction result, or None if processing failed.
        """
        if not self._initialized or self._engine_ptr is None or self._lib is None:
            return None
            
        try:
            # Get hash of instrument ID for Rust
            inst_hash = hash(str(instrument_id)) & 0xFFFFFFFFFFFFFFFF
            
            # Call Rust function
            result_struct = self._lib.nexus_process_quote_tick(
                self._engine_ptr,
                inst_hash,
                bid_price,
                ask_price,
                bid_size,
                ask_size,
                ts_event,
            )
            
            # Convert to Python ConvictionResult
            return ConvictionResult(
                conviction=result_struct.conviction,
                conviction_std=result_struct.conviction_std,
                effective_leverage=result_struct.effective_leverage,
                regime=result_struct.regime,
                num_signals=result_struct.num_signals,
                ts=result_struct.ts,
            )
            
        except Exception as e:
            self._log.error(f"Error processing quote tick: {e}")
            return None
            
    def process_trade_tick(
        self,
        instrument_id: InstrumentId,
        price: float,
        size: float,
        aggressor_side: int,
        ts_event: int,
    ) -> Optional[ConvictionResult]:
        """
        Process a trade tick through the Rust engine.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument identifier.
        price : float
            Trade price.
        size : float
            Trade size.
        aggressor_side : int
            Aggressor side (1=buy, 2=sell).
        ts_event : int
            Event timestamp in nanoseconds.
            
        Returns
        -------
        ConvictionResult, optional
            The conviction result, or None if processing failed.
        """
        if not self._initialized or self._engine_ptr is None or self._lib is None:
            return None
            
        try:
            inst_hash = hash(str(instrument_id)) & 0xFFFFFFFFFFFFFFFF
            
            result_struct = self._lib.nexus_process_trade_tick(
                self._engine_ptr,
                inst_hash,
                price,
                size,
                aggressor_side,
                ts_event,
            )
            
            return ConvictionResult(
                conviction=result_struct.conviction,
                conviction_std=result_struct.conviction_std,
                effective_leverage=result_struct.effective_leverage,
                regime=result_struct.regime,
                num_signals=result_struct.num_signals,
                ts=result_struct.ts,
            )
            
        except Exception as e:
            self._log.error(f"Error processing trade tick: {e}")
            return None
            
    def get_conviction_zscore(self) -> float:
        """
        Get the current conviction z-score from the Rust engine.
        
        Returns
        -------
        float
            Current conviction z-score.
        """
        if not self._initialized or self._engine_ptr is None or self._lib is None:
            return 0.0
            
        try:
            return self._lib.nexus_get_conviction_zscore(self._engine_ptr)
        except Exception:
            return 0.0
            
    @property
    def is_initialized(self) -> bool:
        """Check if the bridge is initialized."""
        return self._initialized


# C-compatible structure for ConvictionResult
class ConvictionResultStruct(ctypes.Structure):
    """C-compatible conviction result structure."""
    _fields_ = [
        ("conviction", ctypes.c_double),
        ("conviction_std", ctypes.c_double),
        ("effective_leverage", ctypes.c_double),
        ("regime", ctypes.c_uint8),
        ("num_signals", ctypes.c_uint8),
        ("_padding", ctypes.c_uint8 * 6),  # Alignment padding
        ("ts", ctypes.c_uint64),
    ]
