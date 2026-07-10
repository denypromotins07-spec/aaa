#!/usr/bin/env python3
"""
Compute-Energy Hedger for NEXUS-OMEGA

Coordinates workload shifting with energy market hedging to minimize
total compute + energy costs while maintaining performance SLAs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class WorkloadPriority(Enum):
    CRITICAL = 1      # HFT trading - never pause
    HIGH = 2          # Market making - minimal interruption
    MEDIUM = 3        # Research/ML training - can be paused
    LOW = 4           # Background tasks - first to shift


@dataclass
class ComputeNode:
    """Represents a compute node in the cluster"""
    node_id: str
    location: str
    power_capacity_mw: float
    current_load_mw: float
    workloads: List[dict]
    pue: float = 1.15  # Power Usage Effectiveness


@dataclass
class EnergyContract:
    """Energy purchase contract"""
    contract_id: str
    location: str
    volume_mwh: float
    price_per_mwh: float
    start_time: int
    end_time: int
    contract_type: str  # "fixed", "index", "virtual"


class ComputeEnergyHedger:
    """
    Main hedger class that coordinates compute workload management
    with energy market positions.
    """
    
    def __init__(
        self,
        nodes: List[ComputeNode],
        energy_contracts: List[EnergyContract],
        lmp_model=None,  # LMPStochasticModel from lmp_stochastic_model
        sla_max_latency_ms: float = 1.0
    ):
        """
        Initialize the compute-energy hedger.
        
        Args:
            nodes: List of compute nodes
            energy_contracts: Existing energy contracts
            lmp_model: LMP stochastic model for predictions
            sla_max_latency_ms: Maximum acceptable latency for critical workloads
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.energy_contracts = energy_contracts
        self.lmp_model = lmp_model
        self.sla_max_latency_ms = sla_max_latency_ms
        
        # Performance tracking
        self.total_energy_cost = 0.0
        self.total_compute_value = 0.0
        self.hedge_effectiveness = 0.0
        
    def calculate_node_marginal_cost(self, node_id: str) -> float:
        """
        Calculate marginal cost of compute at a node including energy.
        
        Returns:
            Marginal cost in $/compute-unit
        """
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")
        
        node = self.nodes[node_id]
        
        # Find applicable energy contract
        applicable_contract = None
        current_time = int(time.time())
        
        for contract in self.energy_contracts:
            if (contract.location == node.location and 
                contract.start_time <= current_time <= contract.end_time):
                applicable_contract = contract
                break
        
        if applicable_contract:
            energy_cost_per_mwh = applicable_contract.price_per_mwh
        else:
            # Use spot price estimate
            energy_cost_per_mwh = 50.0  # Default spot price
        
        # Calculate total marginal cost
        # Energy cost per MW-hour * PUE / compute capacity
        energy_cost_per_compute = (energy_cost_per_mwh * node.pue) / 1000.0
        
        return energy_cost_per_compute
    
    def identify_shiftable_workloads(
        self,
        max_pause_duration_hours: int = 6
    ) -> List[Tuple[str, dict]]:
        """
        Identify workloads that can be shifted or paused.
        
        Args:
            max_pause_duration_hours: Maximum acceptable pause duration
            
        Returns:
            List of (node_id, workload) tuples
        """
        shiftable = []
        
        for node_id, node in self.nodes.items():
            for workload in node.workloads:
                priority = workload.get("priority", WorkloadPriority.MEDIUM.value)
                
                # Only shift MEDIUM and LOW priority workloads
                if priority >= WorkloadPriority.MEDIUM.value:
                    # Check if workload can tolerate pause
                    tolerable_pause = workload.get("max_pause_hours", 0)
                    if tolerable_pause >= max_pause_duration_hours:
                        shiftable.append((node_id, workload))
        
        return shiftable
    
    def optimize_workload_distribution(
        self,
        target_locations: List[str],
        energy_prices: Dict[str, float]
    ) -> List[dict]:
        """
        Optimize workload distribution across locations based on energy prices.
        
        Args:
            target_locations: Available locations for workload placement
            energy_prices: Current/predicted energy prices by location
            
        Returns:
            List of migration recommendations
        """
        migrations = []
        
        # Get shiftable workloads
        shiftable = self.identify_shiftable_workloads()
        
        # Sort by energy price
        sorted_locations = sorted(
            [(loc, price) for loc, price in energy_prices.items() if loc in target_locations],
            key=lambda x: x[1]
        )
        
        for node_id, workload in shiftable:
            current_node = self.nodes[node_id]
            current_price = energy_prices.get(current_node.location, 50.0)
            
            # Find cheaper location with capacity
            for target_loc, target_price in sorted_locations:
                if target_price < current_price * 0.85:  # 15% savings threshold
                    # Check capacity at target
                    target_nodes = [n for n in self.nodes.values() if n.location == target_loc]
                    for target_node in target_nodes:
                        available = target_node.power_capacity_mw - target_node.current_load_mw
                        if available >= workload.get("power_mw", 1.0):
                            migrations.append({
                                "workload_id": workload["id"],
                                "from_node": node_id,
                                "to_location": target_loc,
                                "estimated_savings": (current_price - target_price) * workload.get("power_mw", 1.0),
                                "priority": workload.get("priority")
                            })
                            break
                    if migrations and migrations[-1]["to_location"] == target_loc:
                        break
        
        # Sort by savings
        migrations.sort(key=lambda x: x["estimated_savings"], reverse=True)
        return migrations
    
    def execute_hedge_strategy(
        self,
        risk_budget: float = 100000.0,
        hedge_ratio: float = 0.7
    ) -> Dict[str, float]:
        """
        Execute combined compute + energy hedge strategy.
        
        Args:
            risk_budget: Maximum acceptable daily P&L variance
            hedge_ratio: Target fraction of energy exposure to hedge
            
        Returns:
            Strategy execution summary
        """
        summary = {
            "workload_migrations": 0,
            "energy_hedges": 0,
            "total_savings": 0.0,
            "risk_reduced": 0.0
        }
        
        # Step 1: Shift workloads to low-cost regions
        current_prices = self._get_current_energy_prices()
        migrations = self.optimize_workload_distribution(
            list(set(n.location for n in self.nodes.values())),
            current_prices
        )
        
        for migration in migrations[:5]:  # Limit concurrent migrations
            if self._execute_migration(migration):
                summary["workload_migrations"] += 1
                summary["total_savings"] += migration["estimated_savings"]
        
        # Step 2: Execute energy hedges for remaining exposure
        if self.lmp_model:
            hedge_recommendations = self._calculate_hedge_recommendations(hedge_ratio)
            for hedge in hedge_recommendations:
                if hedge["notional"] <= risk_budget:
                    self._execute_energy_hedge(hedge)
                    summary["energy_hedges"] += 1
                    summary["risk_reduced"] += hedge["var_reduction"]
        
        return summary
    
    def _get_current_energy_prices(self) -> Dict[str, float]:
        """Get current energy prices for all locations"""
        prices = {}
        current_time = int(time.time())
        
        for node in self.nodes.values():
            if node.location in prices:
                continue
                
            # Find applicable contract
            for contract in self.energy_contracts:
                if (contract.location == node.location and
                    contract.start_time <= current_time <= contract.end_time):
                    prices[node.location] = contract.price_per_mwh
                    break
            else:
                # Use model prediction or default
                if self.lmp_model:
                    prices[node.location] = self.lmp_model.mu
                else:
                    prices[node.location] = 50.0
        
        return prices
    
    def _execute_migration(self, migration: dict) -> bool:
        """Execute a workload migration (simulation)"""
        # In production, this would:
        # 1. Snapshot workload state
        # 2. Transfer to target
        # 3. Resume execution
        # 4. Update routing
        return True
    
    def _calculate_hedge_recommendations(self, hedge_ratio: float) -> List[dict]:
        """Calculate energy hedge recommendations"""
        recommendations = []
        
        if not self.lmp_model:
            return recommendations
        
        # Calculate total energy exposure by location
        exposures = {}
        for node in self.nodes.values():
            loc = node.location
            exposures[loc] = exposures.get(loc, 0) + node.current_load_mw
        
        for location, load_mw in exposures.items():
            # Calculate VaR
            var = self.lmp_model.calculate_value_at_risk(load_mw)
            
            # Recommend hedge
            hedge_notional = load_mw * hedge_ratio
            var_reduction = var * hedge_ratio
            
            recommendations.append({
                "location": location,
                "notional_mw": hedge_notional,
                "var_reduction": var_reduction
            })
        
        return recommendations
    
    def _execute_energy_hedge(self, hedge: dict) -> bool:
        """Execute an energy hedge (simulation)"""
        # In production, this would submit virtual bids to market operator
        return True
    
    def get_hedge_effectiveness(self) -> float:
        """
        Calculate hedge effectiveness ratio.
        
        Returns:
            Ratio of risk reduced to cost incurred
        """
        if self.total_energy_cost == 0:
            return 0.0
        
        return self.hedge_effectiveness / self.total_energy_cost
    
    def generate_daily_report(self) -> dict:
        """Generate daily hedging report"""
        return {
            "total_energy_cost": self.total_energy_cost,
            "total_compute_value": self.total_compute_value,
            "hedge_effectiveness": self.get_hedge_effectiveness(),
            "active_contracts": len(self.energy_contracts),
            "active_nodes": len(self.nodes),
            "shiftable_workloads": len(self.identify_shiftable_workloads())
        }


def create_sample_environment() -> Tuple[ComputeEnergyHedger, dict]:
    """Create sample environment for testing"""
    
    # Create sample compute nodes
    nodes = [
        ComputeNode(
            node_id="us-east-1",
            location="us-east",
            power_capacity_mw=50.0,
            current_load_mw=35.0,
            workloads=[
                {"id": "wl-001", "priority": 1, "power_mw": 10.0, "max_pause_hours": 0},
                {"id": "wl-002", "priority": 3, "power_mw": 15.0, "max_pause_hours": 4},
                {"id": "wl-003", "priority": 4, "power_mw": 10.0, "max_pause_hours": 8},
            ]
        ),
        ComputeNode(
            node_id="us-west-1",
            location="us-west",
            power_capacity_mw=40.0,
            current_load_mw=25.0,
            workloads=[
                {"id": "wl-004", "priority": 2, "power_mw": 15.0, "max_pause_hours": 1},
                {"id": "wl-005", "priority": 4, "power_mw": 10.0, "max_pause_hours": 6},
            ]
        ),
        ComputeNode(
            node_id="eu-central-1",
            location="eu-central",
            power_capacity_mw=30.0,
            current_load_mw=20.0,
            workloads=[
                {"id": "wl-006", "priority": 3, "power_mw": 12.0, "max_pause_hours": 3},
            ]
        ),
    ]
    
    # Create sample energy contracts
    current_time = int(time.time())
    contracts = [
        EnergyContract(
            contract_id="ctr-001",
            location="us-east",
            volume_mwh=500,
            price_per_mwh=45.0,
            start_time=current_time - 3600,
            end_time=current_time + 86400,
            contract_type="fixed"
        ),
        EnergyContract(
            contract_id="ctr-002",
            location="us-west",
            volume_mwh=400,
            price_per_mwh=55.0,
            start_time=current_time - 3600,
            end_time=current_time + 86400,
            contract_type="index"
        ),
    ]
    
    hedger = ComputeEnergyHedger(nodes, contracts)
    
    context = {
        "nodes": nodes,
        "contracts": contracts,
        "energy_prices": {
            "us-east": 45.0,
            "us-west": 55.0,
            "eu-central": 65.0
        }
    }
    
    return hedger, context


if __name__ == "__main__":
    hedger, context = create_sample_environment()
    
    print("=== NEXUS-OMEGA Compute-Energy Hedger ===\n")
    
    # Show current state
    print("Current Node Status:")
    for node_id, node in hedger.nodes.items():
        utilization = node.current_load_mw / node.power_capacity_mw * 100
        print(f"  {node_id}: {node.current_load_mw:.1f}/{node.power_capacity_mw:.1f} MW ({utilization:.1f}%)")
    
    print("\nShiftable Workloads:")
    shiftable = hedger.identify_shiftable_workloads()
    for node_id, wl in shiftable:
        print(f"  {wl['id']} on {node_id}: Priority {wl['priority']}, Max Pause {wl['max_pause_hours']}h")
    
    print("\nOptimization Recommendations:")
    migrations = hedger.optimize_workload_distribution(
        ["us-east", "us-west", "eu-central"],
        context["energy_prices"]
    )
    for m in migrations:
        print(f"  Move {m['workload_id']} to {m['to_location']}: ${m['estimated_savings']:.2f}/hr savings")
    
    print("\nExecuting Hedge Strategy...")
    summary = hedger.execute_hedge_strategy(risk_budget=50000.0, hedge_ratio=0.6)
    print(f"  Workload migrations: {summary['workload_migrations']}")
    print(f"  Energy hedges: {summary['energy_hedges']}")
    print(f"  Total savings: ${summary['total_savings']:.2f}")
    
    print("\nDaily Report:")
    report = hedger.generate_daily_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
