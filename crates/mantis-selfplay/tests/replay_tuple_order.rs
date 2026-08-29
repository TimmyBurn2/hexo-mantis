//! O-35 — the ⊕ tuple-order pin (versioned v1). Written FIRST against the
//! dispatcher-captured old-side 8/9-tuple order (CAPTURE_LOG §A), passing on the
//! ported return structs. Bites on any reorder, append-instead-of-splice,
//! carrier-type drift, or struct-field rename/removal.

use mantis_selfplay::replay::sample::{
    to_ordered_tags, to_ordered_tags_with_pos, TypeTag, SAMPLE_ORDER_V1, SAMPLE_WITH_POS_ORDER_V1,
};
use mantis_selfplay::replay::ReplayBuffer;
use TypeTag::{F32, U16, U8};

// ── Dispatcher-captured expected values (CAPTURE_LOG §A — the frozen expected
//    order the crate consts are pinned against; editing a crate const bites) ──

const EXPECTED_NAMES_8: [&str; 8] = [
    "states",
    "chain",
    "policies",
    "outcomes",
    "ownership",
    "winning_line",
    "is_full_search",
    "value_target_valid",
];
const EXPECTED_NAMES_9: [&str; 9] = [
    "states",
    "chain",
    "policies",
    "outcomes",
    "ownership",
    "winning_line",
    "is_full_search",
    "position_indices",
    "value_target_valid",
];
// Carrier tags: f16-bits carrier = U16 (states/chain), f32 = F32, u8 = U8,
// genuine-u16 (position_indices) = U16. The float16-vs-uint16 numpy DTYPE
// distinction is a WP7 fact (O-34 gates the f16 path), not a WP5 carrier fact.
const EXPECTED_TAGS_8: [TypeTag; 8] = [U16, U16, F32, F32, U8, U8, U8, U8];
const EXPECTED_TAGS_9: [TypeTag; 9] = [U16, U16, F32, F32, U8, U8, U8, U16, U8];

#[test]
fn crate_consts_equal_dispatcher_captured_order() {
    assert_eq!(
        SAMPLE_ORDER_V1, EXPECTED_NAMES_8,
        "8-tuple field order drifted from capture"
    );
    assert_eq!(
        SAMPLE_WITH_POS_ORDER_V1, EXPECTED_NAMES_9,
        "9-tuple field order drifted from capture"
    );
}

#[test]
fn splice_property_position_indices_before_value_target_valid() {
    // The 9-form is NOT the 8-form with a field appended: position_indices is
    // SPLICED at index 7, pushing value_target_valid to index 8.
    assert_eq!(SAMPLE_WITH_POS_ORDER_V1[7], "position_indices");
    assert_eq!(SAMPLE_WITH_POS_ORDER_V1[8], "value_target_valid");
    assert_eq!(&SAMPLE_WITH_POS_ORDER_V1[..7], &SAMPLE_ORDER_V1[..7]);
    // A naive "append position_indices" would end (..., value_target_valid,
    // position_indices) — this asserts the opposite, so that reorder fails here.
    assert_ne!(SAMPLE_WITH_POS_ORDER_V1[8], "position_indices");
}

#[test]
fn eight_tuple_order_and_carrier_tags_match() {
    let mut buf = ReplayBuffer::new(8, "v6");
    buf.push_for_test(1.0, 10, true);
    let out = buf.sample_batch_core(2, false).expect("sample");

    let ordered = to_ordered_tags(&out);
    let names: Vec<&str> = ordered.iter().map(|(n, _)| *n).collect();
    let tags: Vec<TypeTag> = ordered.iter().map(|(_, t)| *t).collect();

    // Exhaustive-destructure emit order == the const == the capture.
    assert_eq!(names.as_slice(), &SAMPLE_ORDER_V1);
    assert_eq!(names.as_slice(), &EXPECTED_NAMES_8);
    // Carrier-derived tags == expected (a carrier-type drift, e.g.
    // value_target_valid Vec<u8>→Vec<u16>, would flip a tag and fail here).
    assert_eq!(tags.as_slice(), &EXPECTED_TAGS_8);
}

#[test]
fn nine_tuple_order_and_carrier_tags_match_with_splice() {
    let mut buf = ReplayBuffer::new(8, "v6");
    buf.push_for_test(1.0, 10, true);
    let out = buf.sample_batch_with_pos_core(2, false).expect("sample");

    let ordered = to_ordered_tags_with_pos(&out);
    let names: Vec<&str> = ordered.iter().map(|(n, _)| *n).collect();
    let tags: Vec<TypeTag> = ordered.iter().map(|(_, t)| *t).collect();

    assert_eq!(names.as_slice(), &SAMPLE_WITH_POS_ORDER_V1);
    assert_eq!(names.as_slice(), &EXPECTED_NAMES_9);
    assert_eq!(tags.as_slice(), &EXPECTED_TAGS_9);
    // The emitted order splices position_indices at idx 7 (not appended).
    assert_eq!(names[7], "position_indices");
    assert_eq!(names[8], "value_target_valid");
    assert_eq!(tags[7], U16, "position_indices carrier is u16");
    assert_eq!(tags[8], U8, "value_target_valid carrier is u8");
}
