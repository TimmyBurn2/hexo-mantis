//! R344(a) — the replay rings' sampler is reproducible from a declared seed.
//!
//! Both rings seed their `StdRng` from OS entropy at construction, so two launches of the
//! same config never drew the same batch sequence and no Python-side `seed_everything` could
//! reach the field. `seed_sampler` is the one way to declare a stream. What it does NOT buy
//! is stop/resume continuity — `CARD-RING-SAMPLER-SEED` records why that is refused rather
//! than deferred — so the tests below assert run-to-run reproducibility and nothing wider.
//!
//! THE CONTROL IS THE POINT. Asserting "same seed, same indices" alone would pass on a ring
//! whose sampler ignored the seed entirely and happened to be deterministic. Each arm
//! therefore carries its negative: a DIFFERENT seed must produce a DIFFERENT draw, and an
//! UNSEEDED pair must disagree — which is also the standing evidence that construction is
//! still entropy-seeded and this method is not decorative.

use mantis_selfplay::replay::hexg::{GraphRecord, HexgBuffer};
use mantis_selfplay::replay::push_config::PushSingleConfig;
use mantis_selfplay::replay::ReplayBuffer;

const CAP: usize = 256;
const VISIT_CAP: usize = 128;
const N_RECORDS: usize = 64;
const BATCH: usize = 48;

fn filled_graph_ring() -> HexgBuffer {
    let mut buf = HexgBuffer::new(CAP, "gnn_axis_v1", VISIT_CAP).expect("graph ring constructs");
    for i in 0..N_RECORDS {
        let n_stones = 6 + (i % 11) as i16;
        let stones: Vec<(i16, i16, i8)> = (0..n_stones)
            .map(|q| (q, (q % 3) - 1, if q % 2 == 0 { 1 } else { -1 }))
            .collect();
        let rec = GraphRecord {
            stones,
            visits: vec![(-1i16, 0i16, 0.6f32), (n_stones, 0, 0.4)],
            current_player: if i % 2 == 0 { 1 } else { -1 },
            moves_remaining: 2,
            ply_index: (i % 50) as u16,
            is_full_search: true,
            outcome: if i % 3 == 0 { 1.0 } else { -1.0 },
            value_valid: true,
            game_length: 40,
            game_id: -1,
        };
        buf.push_record_impl(&rec, (10 + i) as i64).expect("push");
    }
    buf
}

fn graph_draw(buf: &mut HexgBuffer, seed: Option<u64>) -> Vec<usize> {
    if let Some(s) = seed {
        buf.seed_sampler(s);
    }
    buf.sample_indices(BATCH, 0.0)
}

#[test]
fn a_seeded_graph_ring_draws_the_same_batch_twice_and_a_different_seed_does_not() {
    let mut buf = filled_graph_ring();

    let first = graph_draw(&mut buf, Some(0xC0FF_EE00_0000_0001));
    let again = graph_draw(&mut buf, Some(0xC0FF_EE00_0000_0001));
    assert_eq!(
        first, again,
        "same seed must reproduce the batch stream; this is the whole content of R344(a)"
    );
    assert_eq!(first.len(), BATCH, "the draw must be the requested width");

    let other = graph_draw(&mut buf, Some(0xC0FF_EE00_0000_0002));
    assert_ne!(
        first, other,
        "a different seed drew the SAME indices — the seed is being ignored, and the \
         positive assertion above would pass on a sampler that read no seed at all"
    );
}

#[test]
fn two_unseeded_graph_rings_disagree_so_construction_is_still_entropy_seeded() {
    // The standing evidence for the defect this method closes. If this ever reds, either
    // construction acquired a fixed seed (making `seed_sampler` decorative) or the sampler
    // stopped consuming its generator — both worth knowing loudly.
    let mut a = filled_graph_ring();
    let mut b = filled_graph_ring();
    assert_ne!(
        graph_draw(&mut a, None),
        graph_draw(&mut b, None),
        "two freshly-constructed rings agreed on a {BATCH}-wide draw"
    );
}

#[test]
fn the_dense_ring_carries_the_same_contract() {
    // Read through the PUBLIC `sample_batch_core` rather than the crate-private index draw:
    // widening a production surface so a test can see it is how a test starts constraining
    // the thing it was meant to observe. Every record carries a UNIQUE outcome, so the
    // returned `outcomes` vector IS the draw sequence, one f32 per slot.
    let mut buf = ReplayBuffer::new(CAP, "v6").expect("dense ring constructs");
    let spec = buf.encoding;
    let state = vec![0u16; spec.state_stride()];
    let chain = vec![0u16; spec.chain_stride()];
    let mut policy = vec![0.0f32; spec.policy_stride()];
    policy[0] = 1.0; // a real distribution: a zero policy row is a refusable target
    let ownership = vec![1u8; spec.aux_stride()];
    let winning_line = vec![0u8; spec.aux_stride()];
    let state_f16: Vec<half::f16> = state.iter().map(|b| half::f16::from_bits(*b)).collect();
    let chain_f16: Vec<half::f16> = chain.iter().map(|b| half::f16::from_bits(*b)).collect();
    for i in 0..N_RECORDS {
        buf.push_impl(PushSingleConfig {
            state: &state_f16,
            chain_planes: &chain_f16,
            policy: &policy,
            outcome: -1.0 + 2.0 * (i as f32) / (N_RECORDS as f32 - 1.0),
            ownership: &ownership,
            winning_line: &winning_line,
            game_id: (10 + i) as i64,
            game_length: 40,
            is_full_search: true,
            position_index: (i % 50) as u16,
            value_target_valid: true,
        })
        .expect("dense push");
    }

    buf.seed_sampler(0xD1CE_0000_0000_0001);
    let first = buf
        .sample_batch_core(BATCH, false)
        .expect("sample")
        .outcomes;
    buf.seed_sampler(0xD1CE_0000_0000_0001);
    let again = buf
        .sample_batch_core(BATCH, false)
        .expect("sample")
        .outcomes;
    assert_eq!(first, again, "same seed, same dense draw");

    buf.seed_sampler(0xD1CE_0000_0000_0002);
    let other = buf
        .sample_batch_core(BATCH, false)
        .expect("sample")
        .outcomes;
    assert_ne!(
        first, other,
        "a different seed must produce a different dense draw"
    );
}
