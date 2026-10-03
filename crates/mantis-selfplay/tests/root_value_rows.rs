//! A self-play row's root value is its own search's value (bit-equal to the sampled search stats), or a proven root's proof value.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::SearchKind;
use mantis_selfplay::replay::hexg::GraphRecord;
use mantis_selfplay::runner::{GameResultRow, PositionStats, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;

/// The tactics block at test budgets, root offence armed so roots are decided.
const BLOCK: TacticsConfig = TacticsConfig {
    leaf_turns: 2,
    leaf_nodes: 64,
    root_turns: 4,
    root_nodes: 2_000,
    audit: Some(AuditConfig {
        turns: 4,
        nodes: 256,
        k: 4,
        m: 4,
        total_nodes: 4_000,
        mode: AuditMode::Hold,
    }),
};

type Rows = HashMap<&'static str, u64>;

/// What a drive drained: the game rows (each sampled), the ring rows, and the summed tactics rows.
struct Drove {
    games: Vec<GameResultRow>,
    records: Vec<GraphRecord>,
    rows: Rows,
}

/// One worker, every game sampled, both arms drawn, until `done` and then one more finished game (the one in flight).
fn drive(
    kind: SearchKind,
    tactics: Option<TacticsConfig>,
    ply_cap: usize,
    valued: bool,
    done: fn(&Rows, &[GameResultRow]) -> bool,
) -> Drove {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: ply_cap,
        n_simulations: 32,
        n_sims_quick: 8,
        full_search_prob: 0.5,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_stats_every: 1,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        tactics,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let spawn = if valued {
        common::spawn_valued_compact_producer
    } else {
        common::spawn_compact_producer
    };
    let producer = spawn(
        runner.graph_producer(),
        spec.policy_logit_count,
        Arc::new(AtomicUsize::new(0)),
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let totals = |r: &SelfPlayRunner| -> Rows { r.tactics_totals().into_iter().collect() };
    let (mut games, mut records) = (Vec::new(), Vec::new());
    let mut finished_at: Option<usize> = None;
    while Instant::now() < deadline
        && finished_at.is_none_or(|n| games.len() <= n)
        && runner.fatal_defect().is_none()
    {
        if finished_at.is_none() && done(&totals(&runner), &games) {
            finished_at = Some(games.len());
        }
        // Rows first, then results: a game's result is never drained before its rows.
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
        games.extend(runner.drain_game_results().expect("unpoisoned"));
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));
    games.extend(runner.drain_game_results().expect("unpoisoned"));
    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    Drove {
        games,
        records,
        rows: totals(&runner),
    }
}

/// Each finished game's stats beside its rows: one worker finalizes games in order, each game's rows contiguous under one id.
fn paired(d: &Drove) -> Vec<(&GameResultRow, Vec<&GraphRecord>)> {
    let mut by_game: Vec<(i64, Vec<&GraphRecord>)> = Vec::new();
    for r in &d.records {
        match by_game.last_mut() {
            Some((id, rows)) if *id == r.game_id => rows.push(r),
            _ => by_game.push((r.game_id, vec![r])),
        }
    }
    assert_eq!(
        by_game.len(),
        d.games.len(),
        "every finished game drained its rows"
    );
    d.games
        .iter()
        .zip(by_game.into_iter().map(|(_, rows)| rows))
        .collect()
}

/// Every row of every sampled game, beside the search stats' entry for its ply.
fn rows_beside_stats(d: &Drove) -> Vec<(&PositionStats, &GraphRecord)> {
    let mut out = Vec::new();
    for (game, rows) in paired(d) {
        let stats = game
            .9
            .as_ref()
            .expect("search_stats_every = 1 samples every game");
        assert_eq!(stats.len(), rows.len(), "one ring row per searched ply");
        for (s, r) in stats.iter().zip(rows) {
            assert_eq!(
                u32::from(r.ply_index),
                s.0,
                "the stats and the rows walk the same plies"
            );
            out.push((s, r));
        }
    }
    out
}

/// Σ π′·completedQ rebuilt from one stats entry alone in f64: the unvisited children are ONE logit at v_mix.
fn rebuilt_from_stats(s: &PositionStats, cfg: &SelfPlayRunnerConfig) -> f64 {
    let raw = f64::from(s.2.expect("a Gumbel root stores its raw value"));
    let kids: Vec<(f64, f64, f64)> =
        s.3.iter()
            .map(|&(_, n, q, p)| (f64::from(n), f64::from(q), f64::from(p)))
            .collect();
    let floor = f64::from(f32::MIN_POSITIVE);
    let sum_p: f64 = kids.iter().map(|&(_, _, p)| p.max(floor)).sum();
    let wq = kids.iter().map(|&(_, q, p)| p.max(floor) * q).sum::<f64>() / sum_p;
    let n: f64 = kids.iter().map(|&(n, _, _)| n).sum();
    let v_mix = (raw + n * wq) / (n + 1.0);
    let rest = 1.0 - kids.iter().map(|&(_, _, p)| p).sum::<f64>();
    let mut vals: Vec<(f64, f64)> = kids
        .iter()
        .map(|&(_, q, p)| (p.max(1e-8).ln(), q))
        .collect();
    if rest > 1e-6 {
        vals.push((rest.ln(), v_mix));
    }
    let max_n = kids.iter().map(|&(n, _, _)| n).fold(0.0, f64::max);
    let scale = (f64::from(cfg.c_visit) + max_n) * f64::from(cfg.c_scale);
    let (lo, hi) = vals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, v)| {
            (lo.min(v), hi.max(v))
        });
    let sigma = |v: f64| {
        if cfg.q_rescale {
            (v - lo) / (hi - lo).max(1e-8) * scale
        } else {
            v * scale
        }
    };
    let logits: Vec<f64> = vals.iter().map(|&(lp, v)| lp + sigma(v)).collect();
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|l| (l - top).exp()).sum();
    logits
        .iter()
        .zip(&vals)
        .map(|(l, &(_, v))| (l - top).exp() / z * v)
        .sum()
}

/// The cross-check pin: with no proof the row's value IS the stats' search value bit for bit (Σ π′·completedQ, PUCT's W/N).
#[test]
fn every_row_carries_its_own_searchs_root_value_bit_for_bit() {
    for kind in [SearchKind::Gumbel, SearchKind::Puct] {
        let d = drive(kind, None, 12, true, |_, g| g.len() >= 6);
        let pairs = rows_beside_stats(&d);
        assert!(pairs.len() >= 20, "{kind:?}: only {} rows", pairs.len());
        let nonzero = pairs.iter().filter(|(_, r)| r.root_value != 0.0).count();
        assert!(
            10 * nonzero >= 9 * pairs.len(),
            "{kind:?}: {nonzero} of {} values non-zero",
            pairs.len()
        );
        let quick = pairs.iter().filter(|(_, r)| !r.is_full_search).count();
        assert!(
            quick > 0 && quick < pairs.len(),
            "{kind:?}: both arms drew ({quick} quick)"
        );
        let off_wn = pairs
            .iter()
            .filter(|(s, _)| s.4.to_bits() != s.1.to_bits())
            .count();
        match kind {
            SearchKind::Gumbel => {
                assert!(
                    10 * off_wn >= 9 * pairs.len(),
                    "Gumbel: only {off_wn} of {} values leave W/N",
                    pairs.len()
                );
                let cfg = SelfPlayRunnerConfig::default();
                for (s, r) in &pairs {
                    let want = rebuilt_from_stats(s, &cfg);
                    assert!(
                        (f64::from(r.root_value) - want).abs() < 1e-4,
                        "ply {}: the row's {} is not 0a's Σ π′·completedQ {want} off its own stats",
                        r.ply_index,
                        r.root_value
                    );
                }
            }
            SearchKind::Puct => assert_eq!(off_wn, 0, "PUCT has no π′: its value is W/N"),
        }
        for (s, r) in &pairs {
            let stats_value = s.4;
            assert!(
                r.root_value_valid,
                "{kind:?} ply {}: a searched row has a root value",
                r.ply_index
            );
            assert_eq!(
                r.root_value.to_bits(),
                stats_value.to_bits(),
                "{kind:?} ply {}: the row's root value is not its search's",
                r.ply_index
            );
        }
    }
}

/// A decided root reads +1 (a finish, the owed stone, a new proof), a root lost on cover -1; the outcome confirms the sign.
#[test]
fn a_proven_root_carries_the_proofs_value_and_every_other_row_its_searchs() {
    let d = drive(SearchKind::Gumbel, Some(BLOCK), 80, false, |r, g| {
        g.len() >= 4 && r["proof_stones_played"] > 0 && r["decided_lost"] > 0
    });
    let (mut won, mut lost) = (0u64, 0u64);
    for (s, r) in rows_beside_stats(&d) {
        let stats_value = s.4;
        assert!(r.root_value_valid);
        if r.root_value.to_bits() == stats_value.to_bits() {
            continue;
        }
        if r.root_value == 1.0 {
            won += 1;
            if r.value_valid {
                assert_eq!(
                    r.outcome, 1.0,
                    "ply {}: a proven win's mover lost",
                    r.ply_index
                );
            }
        } else if r.root_value == -1.0 {
            lost += 1;
            if r.value_valid {
                assert_eq!(
                    r.outcome, -1.0,
                    "ply {}: a root lost on cover won",
                    r.ply_index
                );
            }
        } else {
            panic!(
                "ply {}: {} is neither the search's {} nor a proof's",
                r.ply_index, r.root_value, stats_value
            );
        }
    }
    let rows = &d.rows;
    println!("overrides +1 {won} / -1 {lost}: {rows:?}");
    // Every decided search writes one row on either arm; a game in flight at `stop()` drops its rows.
    let decided = rows["finishes_played"] + rows["proof_stones_played"] + rows["root_proofs_found"];
    assert!(
        won > 0 && won <= decided,
        "+1 overrides {won} against {decided} decided roots"
    );
    assert!(
        lost > 0 && lost <= rows["decided_lost"],
        "-1 overrides {lost} against {} lost roots",
        rows["decided_lost"]
    );
}
