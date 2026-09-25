//! The `dirichlet_*` rows are inert on the Gumbel arm BY CODE — witnessed by the drive's own fire counter (`dirichlet_root_fires`): 0 under Gumbel with the rows ARMED over a fully served search, > 0 under PUCT with the same rows, which is what makes the 0 a reading and not a gap.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{RunnerStatsSnapshot, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const N_SIMS: usize = 64;
const ENCODING: &str = "gnn_axis_r8";

/// One driven worker with the Dirichlet rows ARMED (`enabled: true`, the shipped α/ε) until `want_records` records land; the runner's counter snapshot.
fn drive(kind: SearchKind, want_records: usize) -> RunnerStatsSnapshot {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 2,
        n_simulations: N_SIMS,
        gumbel_m: 16,
        leaf_batch_size: 8,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer =
        common::spawn_uniform_producer(runner.graph_producer(), spec.policy_logit_count, served, 8);

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = 0usize;
    while Instant::now() < deadline {
        records += runner.drain_graph_records().len();
        if records >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    assert!(
        defect.is_none(),
        "{kind:?}: latched a fatal defect: {defect:?}"
    );
    assert!(
        records >= want_records,
        "{kind:?}: only {records} records inside the budget — a drive that searches nothing \
         cannot speak about the lever"
    );
    runner.stats_snapshot()
}

#[test]
fn the_armed_dirichlet_rows_fire_zero_times_on_the_gumbel_arm_and_fire_under_puct() {
    let gumbel = drive(SearchKind::Gumbel, 2);
    assert_eq!(
        gumbel.max_sims_per_search, N_SIMS as u64,
        "the Gumbel search must have been served in full for its 0 to be a reading"
    );
    assert!(gumbel.gumbel_rounds > 0, "no halving round ran");
    assert_eq!(
        gumbel.dirichlet_root_fires, 0,
        "the Gumbel arm mixed root noise {} times with the rows armed — R359(d)'s 'inert by \
         code' no longer holds and run9's dropped rows would matter",
        gumbel.dirichlet_root_fires
    );

    let puct = drive(SearchKind::Puct, 2);
    assert!(
        puct.dirichlet_root_fires > 0,
        "the PUCT arm with the same rows fired 0 times — the counter cannot see the lever, so \
         the Gumbel 0 above proves nothing"
    );
    println!(
        "dirichlet_root_fires: gumbel {} / puct {} (armed rows, {N_SIMS} sims)",
        gumbel.dirichlet_root_fires, puct.dirichlet_root_fires
    );
}
