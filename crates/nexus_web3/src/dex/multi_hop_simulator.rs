//! Multi-Hop DEX Simulator
//! Calculates optimal swap routes across multiple pools with U256 precision

use thiserror::Error;
use alloc::vec::Vec;
use uint::U256;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Empty route")]
    EmptyRoute,
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    #[error("Invalid pool state")]
    InvalidPoolState,
}

pub type Result<T> = core::result::Result<T, SimulationError>;

#[derive(Clone, Debug)]
pub struct Pool {
    pub address: [u8; 20],
    pub token0: [u8; 20],
    pub token1: [u8; 20],
    pub fee_bps: u32,
    pub sqrt_price_x96: U256,
    pub liquidity: U256,
    pub tick: i32,
}

#[derive(Clone, Debug)]
pub struct SwapStep {
    pub pool_index: usize,
    pub amount_in: U256,
    pub amount_out: U256,
    pub price_impact_bps: u32,
}

pub struct MultiHopSimulator {
    pools: Vec<Pool>,
}

impl MultiHopSimulator {
    pub fn new(pools: Vec<Pool>) -> Self {
        Self { pools }
    }

    pub fn simulate_multi_hop(
        &self,
        route: &[usize],
        amount_in: U256,
    ) -> Result<Vec<SwapStep>> {
        if route.is_empty() {
            return Err(SimulationError::EmptyRoute);
        }

        let mut current_amount = amount_in;
        let mut steps = Vec::with_capacity(route.len());

        for &pool_idx in route {
            if pool_idx >= self.pools.len() {
                return Err(SimulationError::InvalidPoolState);
            }

            let pool = &self.pools[pool_idx];
            
            // Calculate output with fee
            let fee = current_amount * U256::from(pool.fee_bps) / U256::from(10000);
            let amount_after_fee = current_amount - fee;
            
            // Simple constant product simulation (real implementation would use TickMath)
            let k = pool.liquidity * pool.sqrt_price_x96;
            let new_sqrt_price = k / (pool.liquidity + amount_after_fee);
            
            let amount_out = if new_sqrt_price < pool.sqrt_price_x96 {
                pool.sqrt_price_x96 - new_sqrt_price
            } else {
                U256::zero()
            };

            let impact = if !amount_in.is_zero() {
                ((amount_out * U256::from(10000)) / amount_in).as_u64() as u32
            } else {
                0
            };

            steps.push(SwapStep {
                pool_index: pool_idx,
                amount_in: current_amount,
                amount_out,
                price_impact_bps: impact,
            });

            current_amount = amount_out;
        }

        Ok(steps)
    }

    pub fn find_best_route(
        &self,
        token_in: [u8; 20],
        token_out: [u8; 20],
        amount_in: U256,
        max_hops: usize,
    ) -> Option<(Vec<usize>, U256)> {
        // BFS to find all possible routes
        let mut best_route: Option<Vec<usize>> = None;
        let mut best_output = U256::zero();

        let mut queue = vec![(vec![], token_in, amount_in)];

        while let Some((route, current_token, current_amount)) = queue.pop() {
            if route.len() >= max_hops {
                continue;
            }

            for (idx, pool) in self.pools.iter().enumerate() {
                if route.contains(&idx) {
                    continue;
                }

                let next_token = if pool.token0 == current_token {
                    pool.token1
                } else if pool.token1 == current_token {
                    pool.token0
                } else {
                    continue;
                };

                let mut new_route = route.clone();
                new_route.push(idx);

                if next_token == token_out {
                    // Found complete route
                    if let Ok(steps) = self.simulate_multi_hop(&new_route, amount_in) {
                        if let Some(last_step) = steps.last() {
                            if last_step.amount_out > best_output {
                                best_output = last_step.amount_out;
                                best_route = Some(new_route);
                            }
                        }
                    }
                } else {
                    // Continue searching
                    queue.push((new_route, next_token, current_amount));
                }
            }
        }

        best_route.map(|r| (r, best_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_single_hop() {
        let pool = Pool {
            address: [1u8; 20],
            token0: [2u8; 20],
            token1: [3u8; 20],
            fee_bps: 30,
            sqrt_price_x96: U256::from(1_000_000_000_000_000_000u128),
            liquidity: U256::from(1_000_000_000_000_000_000u128),
            tick: 0,
        };

        let simulator = MultiHopSimulator::new(vec![pool]);
        let result = simulator.simulate_multi_hop(&[0], U256::from(1_000_000_000u128));
        
        assert!(result.is_ok());
    }
}
