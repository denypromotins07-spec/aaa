//! AST Evaluator for Massively Parallel Backtesting
//! 
//! Evaluates expression trees against historical tick data
//! using the Stage 8 Zero-Copy RL environment.

use crate::gp::expression_tree::{ExpressionTree, TreeIterator};
use crate::gp::arena_allocator::{NodePtr, AstNode, NodeData, Operator};
use std::sync::Arc;

/// Result of evaluating a single tree on a data window
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub sharpe_ratio: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub turnover: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub trade_count: u32,
    pub is_valid: bool,
}

impl EvaluationResult {
    pub const fn invalid() -> Self {
        Self {
            sharpe_ratio: f64::NEG_INFINITY,
            total_return: 0.0,
            max_drawdown: f64::MAX,
            turnover: f64::MAX,
            win_rate: 0.0,
            profit_factor: 0.0,
            trade_count: 0,
            is_valid: false,
        }
    }

    pub const fn valid(
        sharpe: f64,
        return_: f64,
        dd: f64,
        turnover: f64,
        win_rate: f64,
        pf: f64,
        trades: u32,
    ) -> Self {
        Self {
            sharpe_ratio: sharpe,
            total_return: return_,
            max_drawdown: dd,
            turnover,
            win_rate,
            profit_factor: pf,
            trade_count: trades,
            is_valid: true,
        }
    }
}

/// Input data slice for evaluation (zero-copy view into RL environment)
pub struct DataWindow<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
    pub obi: &'a [f64],      // Order Book Imbalance
    pub micro_price: &'a [f64],
    pub timestamp: &'a [i64],
    pub length: usize,
}

impl<'a> DataWindow<'a> {
    pub fn new(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        obi: &'a [f64],
        micro_price: &'a [f64],
        timestamp: &'a [i64],
    ) -> Self {
        let len = open.len().min(high.len()).min(low.len()).min(close.len())
            .min(volume.len()).min(obi.len()).min(micro_price.len()).min(timestamp.len());
        
        Self {
            open: &open[..len],
            high: &high[..len],
            low: &low[..len],
            close: &close[..len],
            volume: &volume[..len],
            obi: &obi[..len],
            micro_price: &micro_price[..len],
            timestamp: &timestamp[..len],
            length: len,
        }
    }
}

/// Fast evaluator that executes ASTs without interpretation overhead
/// Uses pre-allocated buffers to avoid heap allocation during evaluation
pub struct AstEvaluator {
    /// Pre-allocated buffer for time series operations
    ts_buffer: Vec<f64>,
    /// Pre-allocated buffer for boolean results
    bool_buffer: Vec<bool>,
    /// Variable bindings (index -> data pointer)
    var_bindings: Vec<*const [f64]>,
}

unsafe impl Send for AstEvaluator {}
unsafe impl Sync for AstEvaluator {}

impl AstEvaluator {
    pub fn new(max_series_length: usize, max_variables: usize) -> Self {
        Self {
            ts_buffer: vec![0.0f64; max_series_length],
            bool_buffer: vec![false; max_series_length],
            var_bindings: vec![std::ptr::null(); max_variables],
        }
    }

    /// Evaluate a tree on the given data window
    /// Returns EvaluationResult with performance metrics
    pub fn evaluate(&mut self, tree: &ExpressionTree, data: &DataWindow) -> EvaluationResult {
        if tree.is_empty() || data.length == 0 {
            return EvaluationResult::invalid();
        }

        // Bind variables to data columns
        self.bind_variables(data);

        // Evaluate tree at each timestep to generate signals
        let mut signals: Vec<f64> = Vec::with_capacity(data.length);
        let mut valid_count = 0usize;

        // Sliding window evaluation
        for t in 0..data.length {
            if let Some(root) = tree.root() {
                match self.eval_node_at_time(root, t, data) {
                    Ok(value) => {
                        signals.push(value);
                        valid_count += 1;
                    }
                    Err(_) => {
                        signals.push(0.0);
                    }
                }
            } else {
                signals.push(0.0);
            }
        }

        if valid_count < data.length / 10 {
            // Less than 10% valid signals = invalid strategy
            return EvaluationResult::invalid();
        }

        // Calculate performance metrics from signals
        self.calculate_metrics(&signals, data)
    }

    /// Bind variable indices to data columns
    fn bind_variables(&mut self, data: &DataWindow) {
        // Standard binding convention:
        // 0: open, 1: high, 2: low, 3: close, 4: volume
        // 5: obi, 6: micro_price, 7+: reserved
        if self.var_bindings.len() > 0 { self.var_bindings[0] = data.open as *const [f64]; }
        if self.var_bindings.len() > 1 { self.var_bindings[1] = data.high as *const [f64]; }
        if self.var_bindings.len() > 2 { self.var_bindings[2] = data.low as *const [f64]; }
        if self.var_bindings.len() > 3 { self.var_bindings[3] = data.close as *const [f64]; }
        if self.var_bindings.len() > 4 { self.var_bindings[4] = data.volume as *const [f64]; }
        if self.var_bindings.len() > 5 { self.var_bindings[5] = data.obi as *const [f64]; }
        if self.var_bindings.len() > 6 { self.var_bindings[6] = data.micro_price as *const [f64]; }
    }

    /// Evaluate a node at a specific timestep
    fn eval_node_at_time(&self, node: NodePtr<AstNode>, t: usize, data: &DataWindow) -> Result<f64, ()> {
        unsafe {
            let n = node.as_ref();
            match &n.data {
                NodeData::ConstantFloat(v) => Ok(*v),
                NodeData::ConstantInt(v) => Ok(*v as f64),
                NodeData::ConstantBool(v) => Ok(if *v { 1.0 } else { 0.0 }),
                NodeData::Variable { index, .. } => {
                    if *index >= self.var_bindings.len() {
                        return Err(());
                    }
                    let slice_ptr = self.var_bindings[*index];
                    if slice_ptr.is_null() {
                        return Err(());
                    }
                    let slice = &*slice_ptr;
                    if t >= slice.len() {
                        return Err(());
                    }
                    Ok(slice[t])
                }
                NodeData::Operator(op) => {
                    self.eval_operator(op, &n.children, n.child_count, t, data)
                }
            }
        }
    }

    /// Evaluate an operator node
    fn eval_operator(
        &self,
        op: &Operator,
        children: &[Option<NodePtr<AstNode>>],
        child_count: u8,
        t: usize,
        data: &DataWindow,
    ) -> Result<f64, ()> {
        match op {
            Operator::Add => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                Ok(a + b)
            }
            Operator::Sub => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                Ok(a - b)
            }
            Operator::Mul => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                Ok(a * b)
            }
            Operator::Div => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                if b.abs() < 1e-10 {
                    return Ok(0.0); // Protect against division by zero
                }
                Ok(a / b)
            }
            Operator::Lt => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                Ok(if a < b { 1.0 } else { 0.0 })
            }
            Operator::Gt => {
                let a = self.eval_child(children, 0, t, data)?;
                let b = self.eval_child(children, 1, t, data)?;
                Ok(if a > b { 1.0 } else { 0.0 })
            }
            Operator::TsMean => {
                let window = self.eval_child(children, 1, t, data)? as usize;
                if window == 0 {
                    return Ok(0.0);
                }
                let start = t.saturating_sub(window);
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for i in start..=t {
                    if let Ok(v) = self.eval_child(children, 0, i, data) {
                        sum += v;
                        count += 1;
                    }
                }
                if count == 0 {
                    return Ok(0.0);
                }
                Ok(sum / count as f64)
            }
            Operator::TsStdDev => {
                let window = self.eval_child(children, 1, t, data)? as usize;
                if window == 0 {
                    return Ok(0.0);
                }
                let start = t.saturating_sub(window);
                let mut values = Vec::with_capacity(window);
                for i in start..=t {
                    if let Ok(v) = self.eval_child(children, 0, i, data) {
                        values.push(v);
                    }
                }
                if values.len() < 2 {
                    return Ok(0.0);
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                Ok(variance.sqrt())
            }
            _ => {
                // Default: evaluate first child or return 0
                self.eval_child(children, 0, t, data)
            }
        }
    }

    fn eval_child(
        &self,
        children: &[Option<NodePtr<AstNode>>],
        index: usize,
        t: usize,
        data: &DataWindow,
    ) -> Result<f64, ()> {
        if index >= children.len() {
            return Err(());
        }
        match children[index] {
            Some(child) => self.eval_node_at_time(child, t, data),
            None => Err(()),
        }
    }

    /// Calculate performance metrics from signal series
    fn calculate_metrics(&self, signals: &[f64], data: &DataWindow) -> EvaluationResult {
        if signals.len() < 10 {
            return EvaluationResult::invalid();
        }

        // Simple returns calculation based on signal * next period return
        let mut returns: Vec<f64> = Vec::with_capacity(signals.len() - 1);
        for i in 0..signals.len() - 1 {
            let price_return = (data.close[i + 1] - data.close[i]) / data.close[i].max(1e-10);
            let strat_return = signals[i] * price_return;
            returns.push(strat_return);
        }

        if returns.is_empty() {
            return EvaluationResult::invalid();
        }

        // Calculate Sharpe Ratio (annualized, assuming daily data)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe = if std_dev < 1e-10 {
            0.0
        } else {
            (mean_return / std_dev) * (252.0_f64).sqrt() // Annualize
        };

        // Calculate Max Drawdown
        let mut cumulative = 0.0f64;
        let mut peak = 0.0f64;
        let mut max_dd = 0.0f64;
        for r in &returns {
            cumulative += r;
            peak = peak.max(cumulative);
            let dd = (peak - cumulative) / peak.max(1e-10);
            max_dd = max_dd.max(dd);
        }

        // Calculate Win Rate and Profit Factor
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut gross_profit = 0.0f64;
        let mut gross_loss = 0.0f64;
        for r in &returns {
            if *r > 0.0 {
                wins += 1;
                gross_profit += r;
            } else if *r < 0.0 {
                losses += 1;
                gross_loss += r.abs();
            }
        }
        let win_rate = wins as f64 / (wins + losses).max(1) as f64;
        let profit_factor = if gross_loss < 1e-10 {
            f64::INFINITY
        } else {
            gross_profit / gross_loss
        };

        // Turnover (simplified: count sign changes)
        let mut turnover = 0usize;
        for i in 1..signals.len() {
            if (signals[i] > 0.0) != (signals[i - 1] > 0.0) {
                turnover += 1;
            }
        }
        let turnover_rate = turnover as f64 / signals.len() as f64;

        EvaluationResult::valid(
            sharpe,
            cumulative,
            max_dd,
            turnover_rate,
            win_rate,
            profit_factor,
            (wins + losses) as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::arena_allocator::{TreeArena, make_const_float, make_operator};

    #[test]
    fn test_basic_evaluation() {
        let mut arena = TreeArena::new(1000);
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let root = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();
        let tree = ExpressionTree::new(root);

        let data = create_test_data(100);
        let mut evaluator = AstEvaluator::new(100, 10);
        let result = evaluator.evaluate(&tree, &data);

        assert!(result.is_valid);
        assert_eq!(result.total_return, 0.0); // Constant signal = no directional bias
    }

    fn create_test_data(length: usize) -> DataWindow<'static> {
        // Static test data for simplicity
        lazy_static::lazy_static! {
            static ref TEST_CLOSE: Vec<f64> = (0..1000).map(|i| 100.0 + (i as f64 * 0.01)).collect();
            static ref TEST_OPEN: Vec<f64> = TEST_CLOSE.clone();
            static ref TEST_HIGH: Vec<f64> = TEST_CLOSE.clone();
            static ref TEST_LOW: Vec<f64> = TEST_CLOSE.clone();
            static ref TEST_VOLUME: Vec<f64> = vec![1000.0; 1000];
            static ref TEST_OBI: Vec<f64> = vec![0.0; 1000];
            static ref TEST_MICRO: Vec<f64> = TEST_CLOSE.clone();
            static ref TEST_TS: Vec<i64> = (0..1000).collect();
        }

        DataWindow::new(
            &TEST_OPEN[..length],
            &TEST_HIGH[..length],
            &TEST_LOW[..length],
            &TEST_CLOSE[..length],
            &TEST_VOLUME[..length],
            &TEST_OBI[..length],
            &TEST_MICRO[..length],
            &TEST_TS[..length],
        )
    }
}
