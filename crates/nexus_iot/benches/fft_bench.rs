use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nexus_iot::dsp::simd_fft_vibration::SimdFftProcessor;

fn bench_fft_1024(c: &mut Criterion) {
    let fft = SimdFftProcessor::new(1024).unwrap();
    let mut real = vec![0.0; 1024];
    let mut imag = vec![0.0; 1024];
    
    // Generate sine wave
    for i in 0..1024 {
        let t = i as f64 / 10000.0;
        real[i] = (2.0 * std::f64::consts::PI * 500.0 * t).sin();
    }
    
    c.bench_function("FFT 1024 points", |b| {
        b.iter(|| {
            let mut r = real.clone();
            let mut i = imag.clone();
            fft.compute_fft(black_box(&mut r), black_box(&mut i)).unwrap();
        })
    });
}

criterion_group!(benches, bench_fft_1024);
criterion_main!(benches);
