#!/usr/bin/env python3
"""
Locational Marginal Pricing (LMP) Stochastic Model for Energy Market Trading

Implements stochastic modeling of day-ahead and real-time energy markets
for hedging NEXUS-OMEGA's electricity consumption via virtual bidding.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class MarketType(Enum):
    DAY_AHEAD = "day_ahead"
    REAL_TIME = "real_time"
    VIRTUAL_BID = "virtual_bid"


@dataclass
class LMPComponent:
    """Components of Locational Marginal Price"""
    energy_component: float  # System marginal price ($/MWh)
    congestion_component: float  # Congestion cost ($/MWh)
    loss_component: float  # Transmission losses ($/MWh)
    
    @property
    def total_lmp(self) -> float:
        return self.energy_component + self.congestion_component + self.loss_component


@dataclass
class LoadForecast:
    """Electricity load forecast"""
    timestamp: int  # Unix timestamp
    load_mw: float  # Forecasted load (MW)
    confidence_interval: Tuple[float, float]  # 95% CI
    source: str  # Forecast source


class LMPStochasticModel:
    """
    Stochastic model for Locational Marginal Pricing.
    
    Uses mean-reverting Ornstein-Uhlenbeck process with regime switching
    to model LMP dynamics across different market conditions.
    """
    
    def __init__(
        self,
        historical_lmp: np.ndarray,
        historical_load: np.ndarray,
        regime_threshold: float = 100.0
    ):
        """
        Initialize LMP stochastic model.
        
        Args:
            historical_lmp: Historical LMP values ($/MWh)
            historical_load: Historical load values (MW)
            regime_threshold: Price threshold for high-price regime
        """
        self.historical_lmp = np.array(historical_lmp, dtype=np.float64)
        self.historical_load = np.array(historical_load, dtype=np.float64)
        self.regime_threshold = regime_threshold
        
        # Estimate OU process parameters using maximum likelihood
        self._estimate_parameters()
        
        # Regime probabilities
        self.current_regime = self._determine_current_regime()
        
    def _estimate_parameters(self):
        """Estimate Ornstein-Uhlenbeck process parameters"""
        if len(self.historical_lmp) < 2:
            raise ValueError("Insufficient historical data")
        
        # Calculate returns
        returns = np.diff(self.historical_lmp)
        
        # Mean reversion speed (theta)
        # Using regression: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        x_t = self.historical_lmp[:-1]
        dx = returns
        
        # OLS estimation
        if np.var(x_t) > 0:
            beta = np.cov(dx, x_t)[0, 1] / np.var(x_t)
            self.theta = max(-beta, 0.01)  # Ensure positive mean reversion
        else:
            self.theta = 0.1
        
        # Long-term mean (mu)
        self.mu = np.mean(self.historical_lmp)
        
        # Volatility (sigma)
        self.sigma = np.std(returns) * np.sqrt(24)  # Annualize
        
        # Regime transition probabilities
        high_regime = self.historical_lmp > self.regime_threshold
        transitions = np.diff(high_regime.astype(int))
        
        # P(low -> high) and P(high -> low)
        low_to_high = np.sum(transitions == 1) / max(np.sum(~high_regime[:-1]), 1)
        high_to_low = np.sum(transitions == -1) / max(np.sum(high_regime[:-1]), 1)
        
        self.p_low_to_high = min(max(low_to_high, 0.01), 0.5)
        self.p_high_to_low = min(max(high_to_low, 0.01), 0.5)
        
    def _determine_current_regime(self) -> str:
        """Determine current market regime"""
        if len(self.historical_lmp) == 0:
            return "low"
        return "high" if self.historical_lmp[-1] > self.regime_threshold else "low"
    
    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1.0/24.0,  # Hourly steps
        initial_price: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate LMP paths using Euler-Maruyama scheme.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size (days)
            initial_price: Starting price (uses last historical if None)
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        if initial_price is None:
            initial_price = self.historical_lmp[-1] if len(self.historical_lmp) > 0 else self.mu
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = initial_price
        
        # Determine initial regimes
        regimes = np.array([self.current_regime] * n_paths)
        
        for t in range(n_steps):
            # Regime switching
            rand = np.random.uniform(size=n_paths)
            switch_to_high = (regimes == "low") & (rand < self.p_low_to_high)
            switch_to_low = (regimes == "high") & (rand < self.p_high_to_low)
            
            regimes[switch_to_high] = "high"
            regimes[switch_to_low] = "low"
            
            # Regime-dependent parameters
            mu_t = np.where(regimes == "high", self.mu * 1.5, self.mu)
            sigma_t = np.where(regimes == "high", self.sigma * 1.5, self.sigma)
            
            # OU process: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
            dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
            drift = self.theta * (mu_t - paths[:, t]) * dt
            diffusion = sigma_t * dW
            
            paths[:, t + 1] = paths[:, t] + drift + diffusion
            
            # Ensure non-negative prices
            paths[:, t + 1] = np.maximum(paths[:, t + 1], 0.0)
        
        return paths[:, 1:]  # Exclude initial value
    
    def calculate_value_at_risk(
        self,
        position_mw: float,
        confidence_level: float = 0.95,
        horizon_hours: int = 24
    ) -> float:
        """
        Calculate Value at Risk for energy position.
        
        Args:
            position_mw: Position size (MW)
            confidence_level: Confidence level for VaR
            horizon_hours: Risk horizon (hours)
            
        Returns:
            VaR in dollars
        """
        n_steps = horizon_hours
        n_paths = 10000
        
        paths = self.simulate_paths(n_paths, n_steps)
        
        # Calculate P&L for each path
        initial_prices = np.full(n_paths, self.historical_lmp[-1] if len(self.historical_lmp) > 0 else self.mu)
        final_prices = paths[:, -1]
        
        pnl = (final_prices - initial_prices) * position_mw * horizon_hours / 1000  # $ thousands
        
        # VaR is the loss at the confidence level
        var = -np.percentile(pnl, (1 - confidence_level) * 100)
        
        return var
    
    def predict_day_ahead_lmp(
        self,
        forecast_load: LoadForecast
    ) -> LMPComponent:
        """
        Predict day-ahead LMP based on load forecast.
        
        This implements a simple linear relationship between load and LMP,
        calibrated from historical data. **NO look-ahead bias**: only uses
        data available at day-ahead market close.
        
        Args:
            forecast_load: Day-ahead load forecast
            
        Returns:
            Predicted LMP components
        """
        # Calibrated coefficients (would be estimated from historical data)
        base_energy_price = 35.0  # $/MWh
        load_sensitivity = 0.5  # $/MWh per MW
        
        # Energy component based on load forecast
        energy_component = base_energy_price + load_sensitivity * (forecast_load.load_mw - 1000)
        
        # Congestion component (simplified model)
        # Higher load -> higher congestion probability
        congestion_factor = max(0, (forecast_load.load_mw - 1200) / 200)
        congestion_component = congestion_factor * 15.0
        
        # Loss component (typically 2-5% of energy price)
        loss_component = energy_component * 0.03
        
        return LMPComponent(
            energy_component=max(energy_component, 0),
            congestion_component=max(congestion_component, 0),
            loss_component=max(loss_component, 0)
        )
    
    def calculate_virtual_bid_spread(
        self,
        day_ahead_lmp: LMPComponent,
        expected_real_time_lmp: LMPComponent,
        transaction_fee: float = 0.5
    ) -> Tuple[str, float]:
        """
        Calculate optimal virtual bid direction and size.
        
        Virtual bids exploit spreads between day-ahead and real-time markets.
        
        Args:
            day_ahead_lmp: Day-ahead LMP
            expected_real_time_lmp: Expected real-time LMP
            transaction_fee: Transaction fee ($/MWh)
            
        Returns:
            Tuple of (bid_direction, expected_profit_per_mwh)
        """
        da_price = day_ahead_lmp.total_lmp
        rt_price = expected_real_time_lmp.total_lmp
        
        spread = da_price - rt_price
        
        if spread > transaction_fee:
            # DEC bid: sell DA, buy back RT
            profit = spread - transaction_fee
            return ("DEC", profit)
        elif spread < -transaction_fee:
            # INC bid: buy DA, sell RT
            profit = -spread - transaction_fee
            return ("INC", profit)
        else:
            # No profitable opportunity
            return ("NONE", 0.0)


class ComputeEnergyHedger:
    """
    Hedger for NEXUS-OMEGA's electricity consumption.
    
    Manages virtual bids in energy markets to hedge physical
    electricity consumption across data center locations.
    """
    
    def __init__(
        self,
        lmp_models: dict,  # {location: LMPStochasticModel}
        compute_load_profiles: dict  # {location: List[float]} MW by hour
    ):
        """
        Initialize hedger.
        
        Args:
            lmp_models: LMP models for each location
            compute_load_profiles: Expected compute load profiles by location
        """
        self.lmp_models = lmp_models
        self.compute_load_profiles = compute_load_profiles
        self.active_bids: List[dict] = []
        
    def calculate_hedge_ratio(
        self,
        location: str,
        risk_tolerance: float = 0.1
    ) -> float:
        """
        Calculate optimal hedge ratio for a location.
        
        Args:
            location: Data center location
            risk_tolerance: Maximum acceptable VaR fraction
            
        Returns:
            Hedge ratio (0-1)
        """
        if location not in self.lmp_models:
            raise ValueError(f"No LMP model for location: {location}")
        
        model = self.lmp_models[location]
        load_profile = self.compute_load_profiles.get(location, [10.0])
        avg_load = np.mean(load_profile)
        
        # Calculate unhedged VaR
        unhedged_var = model.calculate_value_at_risk(avg_load, confidence_level=0.95)
        
        # Target VaR based on risk tolerance
        expected_cost = avg_load * model.mu * 24  # Daily cost
        target_var = expected_cost * risk_tolerance
        
        # Hedge ratio to achieve target VaR
        if unhedged_var > target_var:
            hedge_ratio = 1 - (target_var / unhedged_var)
        else:
            hedge_ratio = 0.0
        
        return min(max(hedge_ratio, 0.0), 1.0)
    
    def generate_virtual_bids(
        self,
        locations: List[str],
        max_bid_mw: float = 100.0
    ) -> List[dict]:
        """
        Generate virtual bid recommendations.
        
        Args:
            locations: Locations to consider for bidding
            max_bid_mw: Maximum bid size per location
            
        Returns:
            List of bid recommendations
        """
        bids = []
        
        for location in locations:
            if location not in self.lmp_models:
                continue
                
            model = self.lmp_models[location]
            load_profile = self.compute_load_profiles.get(location, [10.0])
            
            # Get day-ahead forecast (would come from market operator)
            forecast_load = LoadForecast(
                timestamp=0,
                load_mw=np.mean(load_profile),
                confidence_interval=(np.mean(load_profile) * 0.9, np.mean(load_profile) * 1.1),
                source="internal"
            )
            
            da_lmp = model.predict_day_ahead_lmp(forecast_load)
            
            # Simulate expected RT LMP
            rt_paths = model.simulate_paths(1000, 24)
            expected_rt_price = np.mean(rt_paths[:, -1])
            
            rt_lmp = LMPComponent(
                energy_component=expected_rt_price * 0.95,
                congestion_component=da_lmp.congestion_component * 0.8,
                loss_component=da_lmp.loss_component
            )
            
            # Calculate bid
            direction, profit_per_mwh = model.calculate_virtual_bid_spread(da_lmp, rt_lmp)
            
            if direction != "NONE" and profit_per_mwh > 0:
                # Calculate optimal bid size
                hedge_ratio = self.calculate_hedge_ratio(location)
                bid_size = min(hedge_ratio * np.mean(load_profile), max_bid_mw)
                
                bids.append({
                    "location": location,
                    "direction": direction,
                    "size_mw": bid_size,
                    "expected_profit": profit_per_mwh * bid_size * 24,
                    "da_price": da_lmp.total_lmp,
                    "rt_price": rt_lmp.total_lmp
                })
        
        return sorted(bids, key=lambda x: x["expected_profit"], reverse=True)
    
    def execute_hedge(
        self,
        bid: dict,
        execution_price_slippage: float = 0.02
    ) -> dict:
        """
        Execute a virtual bid (simulation).
        
        Args:
            bid: Bid recommendation
            execution_price_slippage: Expected slippage
            
        Returns:
            Execution result
        """
        # In production, this would submit to market operator
        executed_size = bid["size_mw"] * (1 - execution_price_slippage)
        executed_profit = bid["expected_profit"] * (1 - execution_price_slippage)
        
        result = {
            "status": "executed",
            "location": bid["location"],
            "direction": bid["direction"],
            "requested_mw": bid["size_mw"],
            "executed_mw": executed_size,
            "expected_profit": executed_profit
        }
        
        self.active_bids.append(result)
        return result


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic historical data
    historical_lmp = np.random.normal(50, 15, 365 * 24)  # 1 year hourly
    historical_load = np.random.normal(1000, 100, 365 * 24)
    
    # Create model
    model = LMPStochasticModel(historical_lmp, historical_load)
    
    # Simulate paths
    paths = model.simulate_paths(100, 48)
    print(f"Simulated {paths.shape[0]} paths for {paths.shape[1]} hours")
    
    # Calculate VaR
    var = model.calculate_value_at_risk(50.0)  # 50 MW position
    print(f"24-hour VaR (95%): ${var:.2f}K")
    
    # Generate forecast
    forecast = LoadForecast(
        timestamp=0,
        load_mw=1100.0,
        confidence_interval=(1000.0, 1200.0),
        source="internal"
    )
    
    da_lmp = model.predict_day_ahead_lmp(forecast)
    print(f"Day-ahead LMP: ${da_lmp.total_lmp:.2f}/MWh")
    
    # Create hedger
    hedger = ComputeEnergyHedger(
        lmp_models={"us-east": model},
        compute_load_profiles={"us-east": [10.0] * 24}
    )
    
    hedge_ratio = hedger.calculate_hedge_ratio("us-east")
    print(f"Optimal hedge ratio: {hedge_ratio:.2%}")
    
    bids = hedger.generate_virtual_bids(["us-east"])
    print(f"Generated {len(bids)} virtual bid(s)")
