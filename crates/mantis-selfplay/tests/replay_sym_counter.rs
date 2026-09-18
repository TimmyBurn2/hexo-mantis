//! R358(b)/R266: the LAW-18 draw counter; the control is `augment = false` putting EVERY draw in bin 0.

use mantis_selfplay::replay::hexg::{GraphRecord, HexgBuffer};
use mantis_selfplay::replay::sym::N_SYMS;

const CAP: usize = 256;
const VISIT_CAP: usize = 128;
const N_RECORDS: usize = 64;
const BATCH: usize = 48;
const BATCHES: usize = 25;

fn record(n_stones: i16, i: usize) -> GraphRecord {
    let stones: Vec<(i16, i16, i8)> = (0..n_stones)
        .map(|q| (q, (q % 3) - 1, if q % 2 == 0 { 1 } else { -1 }))
        .collect();
    GraphRecord {
        stones,
        visits: vec![(-1i16, 0i16, 0.6f32), (n_stones, 0, 0.4)],
        tail_mass: 0.0,
        current_player: if i.is_multiple_of(2) { 1 } else { -1 },
        moves_remaining: 2,
        ply_index: (i % 50) as u16,
        is_full_search: true,
        outcome: if i.is_multiple_of(3) { 1.0 } else { -1.0 },
        value_valid: true,
        game_length: 40,
        game_id: -1,
    }
}

fn filled_graph_ring(with_empty_board_rows: bool) -> HexgBuffer {
    let mut buf = HexgBuffer::new(CAP, "gnn_axis_v1", VISIT_CAP).expect("graph ring constructs");
    for i in 0..N_RECORDS {
        let n_stones = if with_empty_board_rows && i % 4 == 0 {
            0
        } else {
            6 + (i % 11) as i16
        };
        buf.push_record_impl(&record(n_stones, i), (10 + i) as i64)
            .expect("push");
    }
    buf.seed_sampler(0xD6D6_0000_0000_0001);
    buf
}

fn draw(buf: &mut HexgBuffer, augment: bool) -> usize {
    let mut n = 0;
    for _ in 0..BATCHES {
        let indices = buf.sample_indices(BATCH, 0.0);
        n += buf.draw_syms(&indices, augment).len();
    }
    n
}

#[test]
fn a_fresh_ring_has_counted_nothing() {
    let buf = filled_graph_ring(false);
    assert_eq!(buf.sym_draw_counts, [0u64; N_SYMS]);
    assert_eq!(buf.sym_empty_skipped, 0);
}

#[test]
fn augment_on_populates_all_twelve_bins_and_bin_zero_is_not_over_twice_the_mean() {
    let mut buf = filled_graph_ring(false);
    let n = draw(&mut buf, true);
    let bins = buf.sym_draw_counts;
    assert!(
        bins.iter().all(|&b| b > 0),
        "every element must be drawn after {n} draws: {bins:?}"
    );
    assert_eq!(
        bins.iter().sum::<u64>(),
        n as u64,
        "the bins must account for every draw"
    );
    let mean = n as f64 / N_SYMS as f64;
    assert!(
        (bins[0] as f64) <= 2.0 * mean,
        "bin 0 ({}) is over twice the mean ({mean:.1}) — the draw is stuck on the identity",
        bins[0]
    );
    assert_eq!(
        buf.sym_empty_skipped, 0,
        "no empty-board row was in the ring"
    );
}

#[test]
fn augment_off_puts_every_draw_in_bin_zero() {
    let mut buf = filled_graph_ring(false);
    let n = draw(&mut buf, false);
    let mut expected = [0u64; N_SYMS];
    expected[0] = n as u64;
    assert_eq!(buf.sym_draw_counts, expected);
    assert_eq!(buf.sym_empty_skipped, 0);
}

#[test]
fn an_empty_board_row_is_counted_as_skipped_not_as_an_identity_draw() {
    let mut buf = filled_graph_ring(true);
    let n = draw(&mut buf, true);
    let bins = buf.sym_draw_counts;
    let skipped = buf.sym_empty_skipped;
    assert!(
        skipped > 0,
        "a quarter of the ring is the empty board; some draw must have hit it"
    );
    assert_eq!(
        bins.iter().sum::<u64>() + skipped,
        n as u64,
        "bins + skips account for every draw"
    );
    let mean = bins.iter().sum::<u64>() as f64 / N_SYMS as f64;
    assert!(
        (bins[0] as f64) <= 2.0 * mean,
        "a skip must not be counted into bin 0: {bins:?}"
    );
}

#[test]
fn the_counters_are_cumulative_across_batches() {
    let mut buf = filled_graph_ring(false);
    let first = draw(&mut buf, true);
    let after_first: u64 = buf.sym_draw_counts.iter().sum();
    let second = draw(&mut buf, true);
    let after_second: u64 = buf.sym_draw_counts.iter().sum();
    assert_eq!(after_first, first as u64);
    assert_eq!(
        after_second,
        (first + second) as u64,
        "the bins reset between batches"
    );
}

#[test]
fn the_ring_counts_the_rows_it_hands_to_the_trainer() {
    // R358(c): the replay ratio's numerator is what the sampler handed out, since boot.
    let mut buf = filled_graph_ring(false);
    assert_eq!(buf.samples_consumed_total, 0);
    buf.sample_graph_batch_impl(BATCH, false, 0.0, 1)
        .expect("sample");
    assert_eq!(buf.samples_consumed_total, BATCH as u64);
    buf.sample_graph_batch_impl(BATCH, true, 0.0, 1)
        .expect("sample");
    assert_eq!(
        buf.samples_consumed_total,
        2 * BATCH as u64,
        "cumulative, not per batch"
    );
    // `sample_indices` alone is a probe, not a consumption: the seed tests draw through it.
    buf.sample_indices(BATCH, 0.0);
    assert_eq!(buf.samples_consumed_total, 2 * BATCH as u64);
}
