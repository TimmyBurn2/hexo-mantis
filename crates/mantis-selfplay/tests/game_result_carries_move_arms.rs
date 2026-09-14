//! R353(d): the per-game result row carries every move's `(sims, is_full_search)` beside
//! `move_history` — the graph rows' flag reaches the replay ring, not the game record.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

const PLY_CAP: usize = 6;
const N_SIMS_QUICK: usize = 8;
const N_SIMS_FULL: usize = 24;
const ENCODING: &str = "gnn_axis_r8";

fn spawn_producer(queue: GraphQueue, n_actions: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(4, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let mut ids = Vec::with_capacity(batch.len());
        let mut results = Vec::with_capacity(batch.len());
        for (id, g) in batch {
            let coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| {
                    (
                        g.node_coords[row as usize * 2],
                        g.node_coords[row as usize * 2 + 1],
                    )
                })
                .collect();
            let n = coords.len();
            let probs = vec![1.0f32 / n.max(1) as f32; n];
            ids.push(id);
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, 0.0f32)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

fn drive(random_opening_plies: u32, want_games: usize) -> Vec<GameResultRow> {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: PLY_CAP,
        n_simulations: N_SIMS_QUICK,
        leaf_batch_size: 4,
        random_opening_plies,
        dirichlet_enabled: true,
        search_kind: SearchKind::Gumbel,
        quiescence_enabled: false,
        full_search_prob: 0.5,
        n_sims_quick: N_SIMS_QUICK,
        n_sims_full: N_SIMS_FULL,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(runner.graph_producer(), spec.policy_logit_count, served);
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut games = Vec::new();
    while Instant::now() < deadline {
        games.extend(runner.drain_game_results());
        if games.len() >= want_games || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    games.extend(runner.drain_game_results());
    assert!(defect.is_none(), "latched a fatal defect: {defect:?}");
    assert!(
        games.len() >= want_games,
        "only {} games inside the budget",
        games.len()
    );
    games
}

#[test]
fn every_move_carries_its_arm_and_both_arms_reach_the_row() {
    let games = drive(0, 12);
    let mut full = 0usize;
    let mut quick = 0usize;
    for (plies, _winner, moves, _worker, _term, _mn, _mx, _distinct, arms) in &games {
        assert_eq!(
            arms.len(),
            moves.len(),
            "a game of {plies} plies carries {} arms for {} moves",
            arms.len(),
            moves.len()
        );
        for &(sims, is_full) in arms {
            match (sims as usize, is_full) {
                (N_SIMS_FULL, true) => full += 1,
                (N_SIMS_QUICK, false) => quick += 1,
                other => panic!("an arm that is neither budget: {other:?}"),
            }
        }
    }
    assert!(
        full > 0 && quick > 0,
        "the rows carry only one arm (full={full}, quick={quick}) at p=0.5"
    );
}

#[test]
fn random_opening_plies_carry_no_arm() {
    let games = drive(2, 6);
    for (_plies, _winner, moves, _worker, _term, _mn, _mx, _distinct, arms) in &games {
        assert_eq!(arms.len(), moves.len());
        assert_eq!(&arms[..2], &[(0u32, false), (0u32, false)][..]);
        for &(sims, _) in &arms[2..] {
            assert!(sims > 0, "a searched ply carries sims 0");
        }
    }
}
