//! Integration tests for Information Theory module

use nexus_timeseries::prelude::*;

#[test]
fn test_shannon_entropy_uniform() {
    let mut entropy = ShannonEntropy::new(10);
    
    // Uniform distribution
    for i in 0..1000 {
        entropy.update((i % 10) as f64);
    }
    
    let h = entropy.entropy().unwrap();
    assert!(h > 2.0); // log(10) ≈ 2.3
}

#[test]
fn test_shannon_entropy_concentrated() {
    let mut entropy = ShannonEntropy::new(10);
    
    // All same value
    for _ in 0..1000 {
        entropy.update(5.0);
    }
    
    let h = entropy.entropy().unwrap();
    assert!(h < 0.1); // Near zero
}

#[test]
fn test_kd_tree_arena_operations() {
    let mut arena = KdTreeArena::new(100, 3);
    
    // Add points
    for i in 0..50 {
        arena.allocate(vec![i as f64, i as f64 * 2.0, i as f64 * 3.0]);
    }
    
    assert!(arena.count() > 0);
    
    // Test ring buffer eviction
    for i in 0..100 {
        arena.allocate(vec![i as f64; 3]);
    }
    
    // Should not exceed max capacity
    assert!(arena.count() <= 100);
}

#[test]
fn test_transfer_entropy_computation() {
    let mut te = TransferEntropyKsg::new(3, 2, 2, 1, 500);
    
    // Correlated series with lag
    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 - 0.3).sin()).collect();
    
    let result = te.compute(&x, &y);
    assert!(result.is_some());
}
