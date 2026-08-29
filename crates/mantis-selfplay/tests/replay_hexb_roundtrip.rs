//! R8-justify: the HEXB persist oracle roster (O-1..O-12 + O-34a — the four-version compat matrix, the hard-reject legs, and the frozen v9 byte-golden) binds ONE on-disk format; the version legs share one write-then-load scaffold and the O-numbering is contiguous by design.
//! HEXB (dense) oracle suite — version round-trips (O-1..O-4), v5 hard-reject
//! (O-5), all-grid + max-name round-trips (O-6/O-7), P13 reject leg (O-8),
//! unknown-encoding + n_planes guards (O-10/O-11), the frozen byte-golden
//! (O-12), and the f16-bits save→load preservation leg (O-34a). The O-9
//! accept-on-name-mismatch witness is an in-src `#[cfg(test)]` unit test in
//! `persist/load.rs` (it needs the test-only `with_encoding` ctor). Ported from
//! the predecessor engine's `replay_buffer/persist/mod.rs` tests.

use std::io::Write;

use half::f16;

use mantis_encoding::registry::lookup_or_panic;
use mantis_selfplay::replay::ReplayBuffer;

fn unique_path(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("hexb_{stem}_{pid}_{nanos}_{n}.hexb"))
}

/// v6 stride facts, computed from the spec (used for hand-written headers).
fn v6_strides() -> (usize, usize, usize, usize) {
    let s = lookup_or_panic("v6");
    (
        s.state_stride(),
        s.chain_stride(),
        s.policy_stride(),
        s.aux_stride(),
    )
}

/// v7-format per-row byte size (no position_index, no value_target_valid).
fn v7_entry_bytes() -> usize {
    let (st, ch, po, au) = v6_strides();
    st * 2 + ch * 2 + po * 4 + 4 + 8 + 2 + au + au + 1
}

// ── O-1: HEXB v9 in-memory round-trip ──────────────────────────────────────────

#[test]
fn o1_aux_and_chain_f16_bits_roundtrip() {
    let mut buf = ReplayBuffer::new(8, "v6");
    let slot = 0;
    let aux_stride = buf.encoding.aux_stride();
    let chain_stride = buf.encoding.chain_stride();
    let a_start = slot * aux_stride;
    buf.ownership[a_start + 10] = 2; // P1
    buf.ownership[a_start + 20] = 0; // P2
    buf.ownership[a_start + 30] = 1; // empty
    for i in 0..6 {
        buf.winning_line[a_start + 100 + i] = 1;
    }
    let c_start = slot * chain_stride;
    buf.chain_planes[c_start] = f16::from_f32(0.5).to_bits();
    buf.chain_planes[c_start + 100] = f16::from_f32(1.0).to_bits();
    buf.chain_planes[c_start + buf.encoding.n_cells()] = f16::from_f32(0.25).to_bits();
    buf.outcomes[slot] = 1.0;
    buf.weights[slot] = f16::from_f32(1.0).to_bits();
    buf.is_full_search[slot] = 0; // quick-search
    buf.position_indices[slot] = 42;
    buf.head = 1;
    buf.size = 1;

    let path = unique_path("o1_roundtrip");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    let mut buf2 = ReplayBuffer::new(8, "v6");
    let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 1);

    let a2 = 0;
    assert_eq!(buf2.ownership[a2 + 10], 2);
    assert_eq!(buf2.ownership[a2 + 20], 0);
    assert_eq!(buf2.ownership[a2 + 30], 1);
    for i in 0..6 {
        assert_eq!(buf2.winning_line[a2 + 100 + i], 1);
    }
    assert_eq!(buf2.chain_planes[0], f16::from_f32(0.5).to_bits());
    assert_eq!(buf2.chain_planes[100], f16::from_f32(1.0).to_bits());
    assert_eq!(
        buf2.chain_planes[buf2.encoding.n_cells()],
        f16::from_f32(0.25).to_bits()
    );
    assert_eq!(buf2.is_full_search_at(0), 0);
    assert_eq!(buf2.position_indices[0], 42);
    let _ = std::fs::remove_file(path);
}

#[test]
fn o1_value_target_valid_v9_roundtrip() {
    let mut buf = ReplayBuffer::new(8, "v6");
    buf.push_for_test(0.5, 30, true); // normal — supervise
    buf.push_for_test(0.0, 30, true); // ply-capped — masked below
    buf.value_target_valid[1] = 0;

    let path = unique_path("o1_vv_v9");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    let mut buf2 = ReplayBuffer::new(8, "v6");
    buf2.value_target_valid[0] = 0; // poison — load must WRITE per-row
    let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 2);
    assert_eq!(
        buf2.value_target_valid_at(0),
        1,
        "uncapped row reloads as supervise"
    );
    assert_eq!(
        buf2.value_target_valid_at(1),
        0,
        "ply-capped row stays masked"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn o1_position_index_multirow_roundtrip() {
    let mut buf = ReplayBuffer::new(8, "v6");
    for i in 0..5u16 {
        buf.push_for_test(i as f32, 30, true);
    }
    for i in 0..5u16 {
        buf.position_indices[i as usize] = i * 7 + 3;
    }
    let path = unique_path("o1_pos");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    let mut buf2 = ReplayBuffer::new(8, "v6");
    let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 5);
    for i in 0..5u16 {
        assert_eq!(
            buf2.position_indices[i as usize],
            i * 7 + 3,
            "position_index mismatch row {i}"
        );
    }
    let _ = std::fs::remove_file(path);
}

// ── O-2: v8 file → value_target_valid defaults to 1 (supervise) ─────────────────

#[test]
fn o2_v8_file_defaults_value_target_valid_to_supervise() {
    let path = unique_path("o2_v8");
    let entry_v8 = v7_entry_bytes() + 2; // + position_index
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap(); // magic
        file.write_all(&8u32.to_le_bytes()).unwrap(); // version 8
        file.write_all(&8u32.to_le_bytes()).unwrap(); // n_planes
        file.write_all(&10u64.to_le_bytes()).unwrap(); // capacity
        file.write_all(&2u64.to_le_bytes()).unwrap(); // size
        file.write_all(&2u32.to_le_bytes()).unwrap(); // name_len
        file.write_all(b"v6").unwrap();
        for _ in 0..2 {
            file.write_all(&vec![0u8; entry_v8]).unwrap();
        }
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    buf.value_target_valid[0] = 0; // stale — load must overwrite
    buf.value_target_valid[1] = 0;
    let n = buf.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 2);
    assert_eq!(buf.value_target_valid_at(0), 1);
    assert_eq!(buf.value_target_valid_at(1), 1);
    let _ = std::fs::remove_file(path);
}

// ── O-3: v7 file → position_index defaults to 0 ─────────────────────────────────

#[test]
fn o3_v7_file_defaults_position_index_to_zero() {
    let path = unique_path("o3_v7");
    let entry_v7 = v7_entry_bytes();
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap();
        file.write_all(&7u32.to_le_bytes()).unwrap(); // version 7
        file.write_all(&8u32.to_le_bytes()).unwrap();
        file.write_all(&10u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(b"v6").unwrap();
        file.write_all(&vec![0u8; entry_v7]).unwrap();
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    buf.position_indices[0] = 99; // stale — v7 load must default to 0
    let n = buf.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 1);
    assert_eq!(buf.position_indices[0], 0);
    let _ = std::fs::remove_file(path);
}

// ── O-4: v6 file compat (assumed "v6" + outcomes intact) ────────────────────────

#[test]
fn o4_v6_backward_compat() {
    let path = unique_path("o4_v6");
    let entry_v6 = v7_entry_bytes(); // v6 row layout == v7 row layout
    let (st, ch, po, _au) = v6_strides();
    let outcome_off = st * 2 + ch * 2 + po * 4;
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap();
        file.write_all(&6u32.to_le_bytes()).unwrap(); // version 6 (no name field)
        file.write_all(&8u32.to_le_bytes()).unwrap();
        file.write_all(&10u64.to_le_bytes()).unwrap();
        file.write_all(&5u64.to_le_bytes()).unwrap();
        for i in 0..5 {
            let mut entry = vec![0u8; entry_v6];
            entry[outcome_off..outcome_off + 4].copy_from_slice(&(i as f32).to_le_bytes());
            file.write_all(&entry).unwrap();
        }
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    let n = buf.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 5);
    assert_eq!(buf.size(), 5);
    for slot in 0..5 {
        assert_eq!(
            buf.outcomes[slot], slot as f32,
            "v6 compat outcome mismatch at {slot}"
        );
    }
    let _ = std::fs::remove_file(path);
}

// ── O-5: v5 and earlier HARD-REJECT ─────────────────────────────────────────────

#[test]
fn o5_v5_hard_reject() {
    let path = unique_path("o5_v5");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap();
        file.write_all(&5u32.to_le_bytes()).unwrap();
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    let err = buf.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("not supported"),
        "expected 'not supported', got: {err}"
    );
    let _ = std::fs::remove_file(path);
}

// ── O-6: all grid encodings round-trip (re-anchored to the pruned grid set) ─────

#[test]
fn o6_all_grid_encodings_round_trip() {
    for name in ["v6", "v6w25", "v6_live2_ls"] {
        let mut buf = ReplayBuffer::new(10, name);
        for i in 0..10 {
            buf.push_for_test(i as f32, 10, i % 3 == 0);
        }
        let path = unique_path(&format!("o6_{name}"));
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(10, name);
        let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 10, "encoding {name}: expected 10 loaded");
        for slot in 0..10 {
            assert_eq!(
                buf2.outcomes[slot], slot as f32,
                "{name} outcome mismatch at {slot}"
            );
            assert_eq!(
                buf2.is_full_search_at(slot),
                (slot % 3 == 0) as u8,
                "{name} ifs mismatch at {slot}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}

// ── O-7: max-length registered grid name round-trips ────────────────────────────

#[test]
fn o7_max_name_round_trip() {
    // Longest registered grid encoding name = "v6_live2_ls" (11 bytes).
    let mut buf = ReplayBuffer::new(10, "v6_live2_ls");
    for i in 0..10 {
        buf.push_for_test(i as f32, 10, true);
    }
    let path = unique_path("o7_maxname");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    let mut buf2 = ReplayBuffer::new(10, "v6_live2_ls");
    let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 10);
    for slot in 0..10 {
        assert_eq!(buf2.outcomes[slot], slot as f32);
    }
    let _ = std::fs::remove_file(path);
}

// ── O-8: P13 wire-signature cross-load REJECT leg (re-anchored) ──────────────────

#[test]
fn o8_v6_file_into_v6w25_buffer_rejects() {
    let mut writer = ReplayBuffer::new(8, "v6");
    writer.push_for_test(1.0, 10, true);
    let path = unique_path("o8_v6_to_v6w25");
    writer.save_to_path(path.to_str().unwrap()).unwrap();

    let mut reader = ReplayBuffer::new(10, "v6w25");
    let err = reader.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("encoding mismatch"),
        "expected 'encoding mismatch': {err}"
    );
    assert!(
        err.contains("wire_signature"),
        "expected 'wire_signature' framing: {err}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn o8_v6_file_into_gnn_buffer_rejects() {
    let mut writer = ReplayBuffer::new(8, "v6");
    writer.push_for_test(1.0, 10, true);
    let path = unique_path("o8_v6_to_gnn");
    writer.save_to_path(path.to_str().unwrap()).unwrap();

    // gnn_axis_v1: n_planes=0 → zero-width dense storage, constructs fine; the v6
    // file passes the file-vs-file n_planes header guard, then the wire_signature
    // guard rejects ((0,..) != (8,..)).
    let mut reader = ReplayBuffer::new(8, "gnn_axis_v1");
    let err = reader.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("encoding mismatch"),
        "expected 'encoding mismatch': {err}"
    );
    assert!(
        err.contains("wire_signature"),
        "expected 'wire_signature' framing: {err}"
    );
    let _ = std::fs::remove_file(path);
}

// ── O-10: unknown-encoding reject ───────────────────────────────────────────────

#[test]
fn o10_unknown_encoding_rejects() {
    let path = unique_path("o10_unknown");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap();
        file.write_all(&7u32.to_le_bytes()).unwrap();
        file.write_all(&8u32.to_le_bytes()).unwrap();
        file.write_all(&10u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        let name = b"nonexistent";
        file.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    let err = buf.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("unknown encoding"),
        "expected 'unknown encoding': {err}"
    );
    let _ = std::fs::remove_file(path);
}

// ── O-11: n_planes header guard ─────────────────────────────────────────────────

#[test]
fn o11_n_planes_header_guard() {
    let path = unique_path("o11_nplanes");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0x4845_5842u32.to_le_bytes()).unwrap();
        file.write_all(&7u32.to_le_bytes()).unwrap();
        file.write_all(&11u32.to_le_bytes()).unwrap(); // n_planes != v6's 8
        file.write_all(&10u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(b"v6").unwrap();
    }
    let mut buf = ReplayBuffer::new(10, "v6");
    let err = buf.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("n_planes"), "expected n_planes framing: {err}");
    assert!(
        err.contains("corrupted"),
        "expected 'corrupted' framing: {err}"
    );
    let _ = std::fs::remove_file(path);
}

// ── O-12: frozen HEXB v9 byte-golden — load exact + re-save byte-identity ────────

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/replay/hexb_v9_golden.hexb")
}

#[test]
fn o12_hexb_v9_byte_golden_load_and_resave_identity() {
    let golden = golden_path();
    let golden_bytes = std::fs::read(&golden).expect("frozen hexb golden must exist");
    assert_eq!(golden_bytes.len(), 49218, "golden size drifted");

    let mut buf = ReplayBuffer::new(8, "v6");
    let n = buf.load_from_path(golden.to_str().unwrap()).unwrap();
    assert_eq!(n, 4, "golden holds 4 records");

    // Exact per-row field values (CAPTURE_LOG §C). NB: idx 3 has game_length=0,
    // so push_for_test's `game_length==0 → weight 1.0` special case stored weight
    // 1.0 (NOT schedule.weight_for=0.15); the §C weight annotation for idx 3 is a
    // dispatcher slip — the frozen bytes store 1.0 and the re-save byte-identity
    // below is the authoritative absolute-byte gate.
    let expect_outcome = [0.5f32, -0.25, 1.0, -1.0];
    let expect_ifs = [1u8, 0, 1, 0];
    for i in 0..4 {
        assert_eq!(buf.outcomes[i], expect_outcome[i], "outcome[{i}]");
        assert_eq!(
            buf.is_full_search_at(i),
            expect_ifs[i],
            "is_full_search[{i}]"
        );
        assert_eq!(buf.position_indices[i], 0, "position_index[{i}] default");
        assert_eq!(
            buf.value_target_valid_at(i),
            1,
            "value_target_valid[{i}] default"
        );
    }

    // Re-save the loaded content → byte-identical to the frozen golden (absolute
    // byte parity; equivalent to the frozen sha match, no hasher dep needed).
    let resave = unique_path("o12_resave");
    buf.save_to_path(resave.to_str().unwrap()).unwrap();
    let resaved_bytes = std::fs::read(&resave).unwrap();
    assert_eq!(
        resaved_bytes, golden_bytes,
        "new-engine re-save must reproduce the frozen bytes"
    );
    let _ = std::fs::remove_file(resave);
}

// ── O-34a: f16-bits-as-u16 byte preservation through save→load ──────────────────

#[test]
fn o34a_f16_bits_survive_save_load_no_roundtrip() {
    use mantis_selfplay::replay::push_config::PushSingleConfig;

    let spec = lookup_or_panic("v6");
    let (st, ch, po, au) = (
        spec.state_stride(),
        spec.chain_stride(),
        spec.policy_stride(),
        spec.aux_stride(),
    );

    // Craft adversarial f16 bit patterns: NaN, subnormal, -0, max-normal.
    let patterns: [u16; 4] = [0x7e00, 0x0001, 0x8000, 0x7bff];
    let mut state = vec![f16::from_f32(0.0); st];
    let mut chain = vec![f16::from_f32(0.0); ch];
    for (i, &bits) in patterns.iter().enumerate() {
        state[i] = f16::from_bits(bits);
        chain[i] = f16::from_bits(bits);
    }
    let policy = vec![0.0f32; po];
    let own = vec![1u8; au];
    let wl = vec![0u8; au];

    let mut buf = ReplayBuffer::new(4, "v6");
    buf.push_impl(PushSingleConfig {
        state: &state,
        chain_planes: &chain,
        policy: &policy,
        outcome: 0.0,
        ownership: &own,
        winning_line: &wl,
        game_id: -1,
        game_length: 0,
        is_full_search: true,
        position_index: 0,
        value_target_valid: true,
    })
    .unwrap();

    // In-memory: the stored bits equal the input bits exactly (no f16→f32→f16).
    for (i, &bits) in patterns.iter().enumerate() {
        assert_eq!(buf.states[i], bits, "in-mem state bit pattern {i}");
        assert_eq!(buf.chain_planes[i], bits, "in-mem chain bit pattern {i}");
    }

    let path = unique_path("o34a_f16");
    buf.save_to_path(path.to_str().unwrap()).unwrap();
    let mut buf2 = ReplayBuffer::new(4, "v6");
    buf2.load_from_path(path.to_str().unwrap()).unwrap();
    for (i, &bits) in patterns.iter().enumerate() {
        assert_eq!(buf2.states[i], bits, "post save/load state bit pattern {i}");
        assert_eq!(
            buf2.chain_planes[i], bits,
            "post save/load chain bit pattern {i}"
        );
    }
    let _ = std::fs::remove_file(path);
}
