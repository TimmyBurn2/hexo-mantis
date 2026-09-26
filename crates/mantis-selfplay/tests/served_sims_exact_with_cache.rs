//! With the production eval cache on, a search still serves EXACTLY `n_simulations` leaves and the GPU sees only the misses.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::queues::EVAL_CACHE_CAPACITY;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;
const SIMS: usize = 64;

fn drive(kind: SearchKind) -> (usize, usize, u64, u64, u64) {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 3,
        n_simulations: SIMS,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        eval_cache_capacity: EVAL_CACHE_CAPACITY,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let answered = Arc::new(AtomicUsize::new(0));
    let producer = common::spawn_uniform_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        answered.clone(),
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline && records.len() < 6 && runner.fatal_defect().is_none() {
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));
    let snap = runner.stats_snapshot();
    assert!(defect.is_none(), "{kind:?}: latched {defect:?}");
    assert!(
        records.len() >= 6,
        "{kind:?}: only {} searches",
        records.len()
    );
    (
        answered.load(Ordering::Relaxed),
        records.len(),
        snap.max_sims_per_search,
        snap.served_leaves_total,
        snap.gpu_evals_total,
    )
}

#[test]
fn a_cached_search_serves_exactly_its_budget_and_the_gpu_sees_only_the_misses() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let (answered, records, max_sims, served, gpu) = drive(kind);
        println!("{kind:?}: {records} searches, widest {max_sims}, served {served}, gpu {gpu}, answered {answered}");
        assert_eq!(
            max_sims, SIMS as u64,
            "{kind:?}: a cache hit changed the served count"
        );
        let expected = (records * SIMS) as u64;
        assert!(
            served >= expected && served < expected + SIMS as u64,
            "{kind:?}: served {served} outside [{expected}, +{SIMS})"
        );
        assert!(
            gpu < served,
            "{kind:?}: the cache never fired ({gpu} of {served})"
        );
        assert!(
            gpu as usize <= answered && answered < gpu as usize + SIMS,
            "{kind:?}: the GPU answered {answered} graphs but {gpu} were counted"
        );
    }
}
