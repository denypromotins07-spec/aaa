//! Integration tests for FracDiff module

use nexus_timeseries::prelude::*;

#[test]
fn test_fracdiff_fixed_window_full_pipeline() {
    let mut diff = FixedWindowFracDiff::new(0.4, 50).expect("Failed to create FracDiff");
    
    // Generate synthetic price series with memory
    let mut prices = Vec::new();
    let mut price = 100.0;
    for i in 0..200 {
        price *= 1.0 + (i as f64 * 0.01).sin() * 0.001;
        prices.push(price);
        diff.update(price);
    }
    
    // Should produce output after warmup
    assert!(diff.update(101.0).is_some());
}

#[test]
fn test_adf_stationarity_detection() {
    let mut test = StreamingAdfTest::new(100, 0);
    
    // Stationary AR(1) process
    let mut x = 0.0;
    for i in 0..150 {
        x = 0.5 * x + (i as f64 * 0.1).sin() * 0.1;
        test.update(x);
    }
    
    let result = test.run_test();
    assert!(result.is_some());
}

#[test]
fn test_weight_cache_efficiency() {
    use nexus_timeseries::fracdiff::simd_weight_convolution::WeightCache;
    
    let d_values = [0.2, 0.4, 0.6, 0.8];
    let cache = WeightCache::new(100, &d_values);
    
    assert_eq!(cache.all_weights().len(), 4);
    
    // Should find closest match
    let closest = cache.get_closest(0.41);
    assert!(closest.is_some());
    if let Some(c) = closest {
        assert!((c.d() - 0.4).abs() < 0.05);
    }
}
