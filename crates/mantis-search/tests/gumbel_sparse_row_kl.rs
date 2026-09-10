//! What the SPARSE Gumbel row costs against Mctx's exact target.
//!
//! R8 justify: the driven game, the two target constructions and the divergence are one
//! measurement; split, the reconstruction could change without the drive that produced the row.
//!
//! A row stores the m sampled candidates' exact entries plus ONE scalar, the tail mass alpha:
//! under Sequential Halving every UNVISITED root child completes to the same mixed value, so the
//! tail is the recording prior times one shared scalar. THE ALGEBRA: with the row's OWN prior the
//! reconstruction must reproduce the exact target to floating-point noise. THE DRIFT: with a
//! MOVED prior, KL grows with the move, over a ladder so a reader gets the curve. The exact
//! target is recomputed here from the tree; no production row carries it.

use mantis_core::Board;
use mantis_search::{MCTSTree, MctxRootState, SearchKind};

/// 19-window stride with a pass slot.
const N_ACTIONS: usize = 19 * 19 + 1;
/// The minted candidate count (R347(b)); it is also the sparse row's slot bound.
const GUMBEL_M: usize = 16;
/// Small on purpose: the divergence does not depend on the budget.
const SIMS: usize = 64;
const PLIES: usize = 10;
/// The minted Q-scale 1.0 and the value it replaced, 0.1, BOTH driven: at 1.0 the tail mass is
/// numerically zero, so only 0.1 is a regime in which the reconstruction could be wrong.
const C_VISIT: f32 = 50.0;
const C_SCALES: [f32; 2] = [1.0, 0.1];

/// The algebra bar: floating-point noise, not a tolerance — the same arithmetic reordered.
const EXACT_BAR: f64 = 1e-6;
/// The drift bar: median KL below 0.01 nats.
const DRIFT_BAR: f64 = 0.01;
/// The rung the bar is read at: a tenth of a nat of per-action logit drift.
const DRIFT_BAR_LAMBDA: f64 = 0.10;
/// The drift ladder in NATS of logit displacement; `0.0` re-derives the algebra as a control.
const DRIFT_LADDER: [f64; 6] = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00];

fn r8_board() -> Board {
    let mut board = Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    board
}

/// A skewed but everywhere-positive prior: a uniform one hides a flat fallback.
fn stub_policy() -> Vec<f32> {
    let raw: Vec<f32> = (0..N_ACTIONS)
        .map(|i| 1.0 + (i % 13) as f32 * 0.25)
        .collect();
    let total: f32 = raw.iter().sum();
    raw.into_iter().map(|x| x / total).collect()
}

/// One Gumbel search over `board`, driven exactly as the self-play drive does.
fn search(board: &Board, policy: &[f32], seed: u64, c_scale: f32) -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, C_VISIT, c_scale);
    tree.new_game(board.clone());

    let root = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(root.len(), 1);
    tree.expand_and_backup(&[policy.to_vec()], &[0.1]);

    let budget = SIMS - 1;
    let state = MctxRootState::new_seeded(&tree, GUMBEL_M, budget, seed);
    let mut spent = 0usize;
    while spent < budget {
        let mut round = state.round_batch(&tree, C_VISIT, c_scale);
        if round.is_empty() {
            break;
        }
        round.truncate(budget - spent);
        let Ok(leaves) = tree.select_leaves_forced(&round) else {
            break;
        };
        if leaves.is_empty() {
            break;
        }
        let policies: Vec<Vec<f32>> = (0..leaves.len()).map(|_| policy.to_vec()).collect();
        let values: Vec<f32> = (0..leaves.len())
            .map(|i| 0.3 - 0.05 * ((spent + i) % 7) as f32)
            .collect();
        tree.expand_and_backup(&policies, &values);
        spent += leaves.len();
    }
    tree
}

/// `((q, r), recording prior, visits)` per root child, read off the pool rather than through a
/// getter that would grow the tree's public surface for one measurement.
fn root_children(tree: &MCTSTree) -> Vec<((i32, i32), f32, u32)> {
    let root = &tree.pool[0];
    if !root.is_expanded() {
        return Vec::new();
    }
    let first = root.first_child as usize;
    (first..first + root.n_children as usize)
        .map(|i| {
            let node = &tree.pool[i];
            let val = node.action_idx;
            let cell = ((val >> 16) as i32 - 32768, (val & 0xFFFF) as i32 - 32768);
            (cell, node.prior, node.n_visits)
        })
        .collect()
}

struct Row {
    ply: usize,
    n_legal: usize,
    n_explicit: usize,
    alpha: f64,
    /// KL(exact || reconstructed) at each rung of [`DRIFT_LADDER`], in order.
    kl: Vec<f64>,
    /// Exact mass representable only at [`RECON_FLOOR`] — what the sparse row truly loses.
    floored: f64,
}

/// The floor a reconstructed probability is read at: the f32 smallest normal, since alpha is an
/// `f32` and a tail below it cannot be stored. KL against a hard zero is `+inf` for a difference
/// no CE loss observes; the floored mass is reported alongside, so the floor hides no hole.
const RECON_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// `(KL(p || q), exact mass floored)` over aligned distributions.
fn kl_divergence(p: &[f64], q: &[f64]) -> (f64, f64) {
    let mut acc = 0.0;
    let mut floored = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi <= 0.0 {
            continue;
        }
        let qf = if qi < RECON_FLOOR {
            floored += pi;
            RECON_FLOOR
        } else {
            qi
        };
        acc += pi * (pi / qf).ln();
    }
    (acc, floored)
}

/// Rebuild the row: explicit entries verbatim, `alpha` spread over the rest by `prior`.
fn reconstruct(exact: &[f64], explicit: &[bool], prior: &[f64], alpha: f64) -> Vec<f64> {
    let tail_prior_total: f64 = prior
        .iter()
        .zip(explicit)
        .filter(|(_, &e)| !e)
        .map(|(&p, _)| p)
        .sum();
    exact
        .iter()
        .zip(explicit)
        .zip(prior)
        .map(|((&x, &e), &p)| {
            if e {
                x
            } else if tail_prior_total > 0.0 {
                alpha * p / tail_prior_total
            } else {
                0.0
            }
        })
        .collect()
}

/// The prior a trainer holds after its logits have drifted `lambda` NATS per action.
///
/// LOGIT space, and NOT aligned with the target: on the tail the exact target IS the prior times
/// a constant, so a drift toward it leaves the shape unchanged and the ladder goes flat. The mean
/// displacement is removed so `lambda` moves the SHAPE, not the normalizer.
fn drifted_prior(prior: &[f64], lambda: f64) -> Vec<f64> {
    let n = prior.len();
    let disp: Vec<f64> = (0..n)
        .map(|i| {
            // A deterministic hash into [-1, 1]: no RNG dependency and no seed to lose.
            let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let mean: f64 = disp.iter().sum::<f64>() / n as f64;
    let raw: Vec<f64> = prior
        .iter()
        .zip(&disp)
        .map(|(&p, &d)| p * ((d - mean) * lambda).exp())
        .collect();
    let total: f64 = raw.iter().sum();
    if total <= 0.0 {
        return prior.to_vec();
    }
    raw.into_iter().map(|r| r / total).collect()
}

fn bar_rung() -> usize {
    DRIFT_LADDER
        .iter()
        .position(|&l| (l - DRIFT_BAR_LAMBDA).abs() < 1e-12)
        .expect("the bar's drift must be a rung of the reported ladder")
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return f64::NAN;
    }
    values[values.len() / 2]
}

fn measure_game(c_scale: f32) -> Vec<Row> {
    let policy = stub_policy();
    let mut board = r8_board();
    let mut rows = Vec::new();

    for ply in 0..PLIES {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        let tree = search(&board, &policy, 20_260_910 + ply as u64, c_scale);
        let children = root_children(&tree);
        if children.is_empty() {
            break;
        }
        let target = tree.get_improved_policy_ls(N_ACTIONS, C_VISIT, c_scale);
        let (bcq, bcr) = board.window_center();
        let trunk = board.cluster_window_size() as i32;
        let half = (trunk - 1) / 2;

        // Aligned in root-child order: exact target, recording prior, visited-or-not.
        let exact: Vec<f64> = children
            .iter()
            .map(|&((q, r), _, _)| f64::from(target.get(q, r, bcq, bcr, trunk, half, 0.0)))
            .collect();
        // Normalized first: `exact` returns through an f32 container whose sum misses 1 by
        // noise, which would make KL negative and test precision, not the tail.
        let exact_sum: f64 = exact.iter().sum();
        let exact: Vec<f64> = exact.iter().map(|&x| x / exact_sum).collect();
        let prior: Vec<f64> = children.iter().map(|&(_, p, _)| f64::from(p)).collect();
        let explicit: Vec<bool> = children.iter().map(|&(_, _, v)| v > 0).collect();

        let n_explicit = explicit.iter().filter(|&&e| e).count();
        // Sequential Halving visits at most m candidates, so the row claims at most m.
        assert!(
            n_explicit <= GUMBEL_M,
            "ply {ply}: {n_explicit} visited root children against m={GUMBEL_M} — the sparse \
             row's slot bound is not a bound"
        );
        let explicit_mass: f64 = exact
            .iter()
            .zip(&explicit)
            .filter(|(_, &e)| e)
            .map(|(&x, _)| x)
            .sum();
        let alpha = (1.0 - explicit_mass).clamp(0.0, 1.0);

        let mut kl: Vec<f64> = Vec::with_capacity(DRIFT_LADDER.len());
        let mut floored_at_bar = 0.0;
        for (i, &lambda) in DRIFT_LADDER.iter().enumerate() {
            let p_cur = drifted_prior(&prior, lambda);
            let recon = reconstruct(&exact, &explicit, &p_cur, alpha);
            let (k, floored) = kl_divergence(&exact, &recon);
            kl.push(k);
            if i == bar_rung() {
                floored_at_bar = floored;
            }
        }

        rows.push(Row {
            ply,
            n_legal: legal.len(),
            n_explicit,
            alpha,
            kl,
            floored: floored_at_bar,
        });

        let mv = *legal.first().expect("the legal set was checked non-empty");
        board.apply_move(mv.0, mv.1).expect("a legal move applies");
    }
    rows
}

#[test]
fn the_sparse_row_reproduces_mctxs_exact_target_and_its_drift_is_bounded() {
    println!("R347(a) sparse-row witness — driven r8 game, m = {GUMBEL_M}, {SIMS} sims");
    let mut saw_a_real_tail = false;

    for c_scale in C_SCALES {
        let rows = measure_game(c_scale);
        assert!(
            rows.len() >= 4,
            "c_scale {c_scale}: the drive produced {} rows; the medians below are not a \
             measurement on fewer",
            rows.len()
        );

        // Printed, not merely asserted: "under the bar" cannot be quoted at a re-mint.
        println!("\n  c_scale = {c_scale} ({} rows)", rows.len());
        println!("    ply  n_legal  m_explicit  alpha         floored       KL by drift lambda");
        for r in &rows {
            let kls: Vec<String> = r.kl.iter().map(|k| format!("{k:.2e}")).collect();
            println!(
                "    {:>3}  {:>7}  {:>10}  {:.6e}  {:.3e}  {}",
                r.ply,
                r.n_legal,
                r.n_explicit,
                r.alpha,
                r.floored,
                kls.join(" ")
            );
        }
        let alphas: Vec<f64> = rows.iter().map(|r| r.alpha).collect();
        let alpha_median = median(alphas.clone());
        println!(
            "    alpha: min {:.3e} median {:.3e} max {:.3e}",
            alphas.iter().copied().fold(f64::INFINITY, f64::min),
            alpha_median,
            alphas.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        let mut widest_under_bar: Option<f64> = None;
        for (i, &lambda) in DRIFT_LADDER.iter().enumerate() {
            let m = median(rows.iter().map(|r| r.kl[i]).collect());
            if m < DRIFT_BAR {
                widest_under_bar = Some(lambda);
            }
            println!("    lambda {lambda:.2}: median KL(exact || reconstructed) = {m:.3e} nats");
        }
        match widest_under_bar {
            Some(l) => println!(
                "    R347(a) bar ({DRIFT_BAR} nats) holds out to lambda {l:.2} on this ladder"
            ),
            None => println!("    R347(a) bar ({DRIFT_BAR} nats) is crossed at every rung"),
        }

        // At zero drift the reconstruction IS the exact target.
        let exact_median = median(rows.iter().map(|r| r.kl[0]).collect());
        assert!(
            exact_median.abs() < EXACT_BAR,
            "c_scale {c_scale}: median KL at zero drift is {exact_median:.3e} nats, past the \
             numerical-noise bar {EXACT_BAR:.0e} — the completed-Q target on the UNVISITED \
             legal actions is not the recording prior times one scalar, which is the \
             arithmetic the sparse row rests on"
        );

        // The drift bar, at a declared drift.
        let drift_median = median(rows.iter().map(|r| r.kl[bar_rung()]).collect());
        assert!(
            drift_median < DRIFT_BAR,
            "c_scale {c_scale}: median KL at drift lambda={DRIFT_BAR_LAMBDA} is \
             {drift_median:.3e} nats, above R347(a)'s bar of {DRIFT_BAR} — the tail's shape \
             (current prior vs recording prior) costs more than the ruling admits here"
        );

        // The slot bound over the whole game.
        assert!(
            rows.iter().all(|r| r.n_explicit <= GUMBEL_M),
            "c_scale {c_scale}: a row claimed more than m={GUMBEL_M} explicit entries"
        );

        if alphas.iter().any(|&a| a > 0.01) {
            saw_a_real_tail = true;
            // The ladder must actually measure drift where there is a tail to drift.
            let top = median(rows.iter().map(|r| r.kl[DRIFT_LADDER.len() - 1]).collect());
            assert!(
                top > drift_median,
                "c_scale {c_scale}: KL does not grow with prior drift ({drift_median:.3e} at \
                 {DRIFT_BAR_LAMBDA}, {top:.3e} at {}) — the ladder is not measuring the \
                 deviation it names",
                DRIFT_LADDER[DRIFT_LADDER.len() - 1]
            );
        }
    }

    // Anti-vacuity: at the minted c_scale alpha is zero, so one regime must carry a tail.
    assert!(
        saw_a_real_tail,
        "no driven c_scale produced a row with tail mass above 0.01, so the reconstruction \
         had nothing to rebuild at any regime and every KL above is trivially zero"
    );
}
