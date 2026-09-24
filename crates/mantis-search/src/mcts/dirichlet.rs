/// Dirichlet noise sampling for MCTS root exploration.
///
/// Used by the Rust training path to inject exploration noise into root priors
/// after the first NN expansion, matching the AlphaZero recipe
/// (Silver et al. 2018 §2.1).
///
/// Approach: draw n independent Gamma(alpha, 1.0) samples, normalize by sum.
/// This is mathematically equivalent to sampling from Dir(alpha, …, alpha).
use rand::Rng;
use rand_distr::{Distribution, Gamma};

/// An alpha no Gamma(alpha, 1) can be built from: zero, negative or NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidDirichletAlpha {
    /// The refused concentration.
    pub alpha: f32,
}

impl std::fmt::Display for InvalidDirichletAlpha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InvalidDirichletAlpha: Dirichlet alpha must be > 0, got {}",
            self.alpha
        )
    }
}

impl std::error::Error for InvalidDirichletAlpha {}

/// Sample a symmetric Dirichlet(alpha) of length `n` with the caller's per-worker RNG; Gamma
/// draws that all underflow (sum < 1e-8, very small alpha) fall back to uniform.
/// # Errors
/// `InvalidDirichletAlpha` when `alpha` is not > 0; the RNG is not advanced.
pub fn sample_dirichlet(
    alpha: f32,
    n: usize,
    rng: &mut impl Rng,
) -> Result<Vec<f32>, InvalidDirichletAlpha> {
    debug_assert!(n > 0, "Dirichlet n must be positive");

    let dist = Gamma::new(alpha as f64, 1.0).map_err(|_| InvalidDirichletAlpha { alpha })?;
    let mut samples: Vec<f32> = (0..n).map(|_| dist.sample(rng) as f32).collect();

    let sum: f32 = samples.iter().sum();
    if sum > 1e-8 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Degenerate: all Gamma draws underflowed to zero. Fall back to uniform.
        let u = 1.0 / n as f32;
        samples.fill(u);
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_positive_or_nan_alpha_is_a_named_error_not_a_panic() {
        let mut rng = rand::rng();
        for bad in [0.0f32, -0.3, f32::NAN] {
            let err = sample_dirichlet(bad, 25, &mut rng).expect_err("alpha must be refused");
            assert_eq!(err.alpha.to_bits(), bad.to_bits(), "{err}");
            assert!(err.to_string().contains("alpha must be > 0"), "{err}");
        }
    }

    #[test]
    fn test_dirichlet_different_draws() {
        // Two independent draws from the same alpha/n should almost certainly differ.
        let mut rng = rand::rng();
        let v1 = sample_dirichlet(0.3, 25, &mut rng).expect("valid alpha");
        let v2 = sample_dirichlet(0.3, 25, &mut rng).expect("valid alpha");
        let max_diff: f32 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Two sequential Dirichlet draws were identical (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_dirichlet_sparse_at_low_alpha() {
        // At alpha=0.3, n=25, the effective support (exp(H)) should be sparse.
        // Concretely: average over 100 trials should have fewer than 10 components
        // receiving > 1/n weight (i.e. > 0.04).
        let mut rng = rand::rng();
        let n = 25usize;
        let threshold = 1.0 / n as f32; // 0.04
        let trials = 100;

        let total_above: f32 = (0..trials)
            .map(|_| {
                let v = sample_dirichlet(0.3, n, &mut rng).expect("valid alpha");
                v.iter().filter(|&&s| s > threshold).count() as f32
            })
            .sum();
        let avg_above = total_above / trials as f32;

        assert!(
            avg_above < 10.0,
            "alpha=0.3 n=25: expected avg effective support < 10, got {avg_above:.2}"
        );
    }
}
