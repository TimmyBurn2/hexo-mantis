use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn scaffold_smoke(c: &mut Criterion) {
    c.bench_function("scaffold_fold_64", |b| {
        b.iter(|| black_box((0u64..64).fold(0u64, u64::wrapping_add)))
    });
}
criterion_group!(benches, scaffold_smoke);
criterion_main!(benches);
