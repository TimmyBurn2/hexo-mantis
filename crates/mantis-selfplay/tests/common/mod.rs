//! Helpers shared by the integration tests and benches; a subdirectory, so cargo builds no
//! binary for it, and a bench pulls it in with `#[path]`.

#![allow(dead_code)]

/// One SplitMix64 step: the deterministic stream every seeded corpus here draws from.
pub fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
