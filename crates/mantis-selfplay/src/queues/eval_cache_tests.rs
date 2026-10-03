//! The eval cache's key completeness, version, capacity and byte-budget arms.

use super::*;
use crate::queues::build_leaf_graph;

fn eval(v: f32, dense: usize) -> CachedEval {
    CachedEval {
        policy: LegalSetPolicy {
            dense: vec![v; dense],
            ..Default::default()
        },
        value: v,
        center: (1, 2),
    }
}

fn key(n: u128) -> LeafKey {
    LeafKey(n)
}

/// One `build_leaf_graph` request, every input it reads.
#[derive(Clone)]
struct Request {
    stones: Vec<(i64, i64, i64)>,
    player: i64,
    left: i64,
    win_length: u8,
    radius: u16,
    trunk: i32,
}

fn request() -> Request {
    Request {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, -1)],
        player: 1,
        left: 2,
        win_length: 6,
        radius: 6,
        trunk: 19,
    }
}

fn leaf_key(r: &Request) -> LeafKey {
    LeafKey::of(&r.stones, r.player, r.left, r.win_length, r.radius, r.trunk)
        .expect("a legal request")
}

/// Request pairs that differ in ONE builder input each, named by it.
fn one_input_apart() -> Vec<(&'static str, Request, Request)> {
    let base = request();
    let with = |f: &dyn Fn(&mut Request)| {
        let mut r = base.clone();
        f(&mut r);
        r
    };
    vec![
        (
            "stone added",
            base.clone(),
            with(&|r| r.stones.push((2, -1, 1))),
        ),
        ("stone moved", base.clone(), with(&|r| r.stones[0].0 += 1)),
        // Off the ±9 table a seed of `q·M1 ^ r·M2` keys a cell like its reflection: the pair that showed it.
        (
            "stone reflected off the table",
            with(&|r| r.stones.push((1, 10, 1))),
            with(&|r| r.stones.push((-1, -10, 1))),
        ),
        ("stone owner", base.clone(), with(&|r| r.stones[0].2 = -1)),
        ("current_player", base.clone(), with(&|r| r.player = -1)),
        ("moves_remaining", base.clone(), with(&|r| r.left = 1)),
        ("win_length", base.clone(), with(&|r| r.win_length = 5)),
        ("radius", base.clone(), with(&|r| r.radius = 8)),
        ("trunk_size", base.clone(), with(&|r| r.trunk = 21)),
    ]
}

/// The first pair `key` gives one key; `None` when it separates them all.
fn first_unseparated(key: &dyn Fn(&Request) -> LeafKey) -> Option<&'static str> {
    one_input_apart()
        .into_iter()
        .find(|(_, a, b)| key(a) == key(b))
        .map(|(name, _, _)| name)
}

#[test]
fn the_key_separates_every_input_the_builder_reads() {
    let build = |r: &Request| {
        build_leaf_graph(&r.stones, r.player, r.left, r.win_length, r.radius, r.trunk)
            .expect("legal")
    };
    for (name, a, b) in one_input_apart() {
        assert_ne!(
            build(&a),
            build(&b),
            "{name} is not an input of the built graph"
        );
    }
    assert_eq!(first_unseparated(&leaf_key), None);
}

/// PLANTED BREAK: a key that omits `moves_remaining` must red on the pair that differs only there.
#[test]
fn a_key_blind_to_moves_remaining_is_caught() {
    let blind = |r: &Request| {
        LeafKey::of(&r.stones, r.player, 0, r.win_length, r.radius, r.trunk).expect("legal")
    };
    assert_eq!(first_unseparated(&blind), Some("moves_remaining"));
}

#[test]
fn one_position_in_any_stone_order_is_one_key() {
    let a = request();
    let mut b = request();
    b.stones.reverse();
    assert_eq!(leaf_key(&a), leaf_key(&b));
}

#[test]
fn the_key_refuses_exactly_what_the_builder_refuses() {
    let base = request();
    let bad = [
        Request {
            player: 0,
            ..base.clone()
        },
        Request {
            left: 256,
            ..base.clone()
        },
        Request {
            stones: vec![(0, 0, 2)],
            ..base.clone()
        },
        Request {
            stones: vec![(i64::from(i32::MAX), 0, 1)],
            ..base
        },
    ];
    for r in &bad {
        let built = build_leaf_graph(&r.stones, r.player, r.left, r.win_length, r.radius, r.trunk)
            .expect_err("the builder refuses");
        let keyed = LeafKey::of(&r.stones, r.player, r.left, r.win_length, r.radius, r.trunk)
            .expect_err("the key refuses");
        assert_eq!(keyed, built);
    }
}

/// The stale-version control: an entry is served only under the version that computed it.
#[test]
fn a_hit_needs_the_same_key_and_version() {
    let cache = EvalCache::new(64, EVAL_CACHE_BYTES);
    cache.put(key(7), 3, eval(0.5, 4));
    assert_eq!(cache.get(key(7), 3).map(|e| e.value), Some(0.5));
    assert!(
        cache.get(key(7), 4).is_none(),
        "another net version must miss"
    );
    assert!(cache.get(key(8), 3).is_none());
}

#[test]
fn a_newer_version_drops_the_older_entries() {
    let cache = EvalCache::new(64, EVAL_CACHE_BYTES);
    cache.put(key(1), 1, eval(0.1, 4));
    cache.put(key(1), 2, eval(0.2, 4));
    assert!(cache.get(key(1), 1).is_none());
    assert_eq!(cache.get(key(1), 2).map(|e| e.value), Some(0.2));
    cache.put(key(1), 1, eval(0.9, 4));
    assert_eq!(
        cache.get(key(1), 2).map(|e| e.value),
        Some(0.2),
        "a stale put must not overwrite"
    );
}

#[test]
fn the_entry_count_never_exceeds_the_capacity() {
    let cache = EvalCache::new(EVAL_CACHE_SHARDS * 4, EVAL_CACHE_BYTES);
    for i in 0..10_000u128 {
        cache.put(key(i), 0, eval(0.0, 4));
    }
    assert!(
        cache.len() <= EVAL_CACHE_SHARDS * 4,
        "held {} entries",
        cache.len()
    );
    assert!(!cache.is_empty());
}

#[test]
fn the_byte_budget_is_enforced_at_insert() {
    let one = eval(0.0, 362).bytes();
    let budget = EVAL_CACHE_SHARDS * one * 3;
    let cache = EvalCache::new(EVAL_CACHE_CAPACITY, budget);
    for i in 0..10_000u128 {
        cache.put(key(i), 0, eval(0.0, 362));
    }
    assert!(
        cache.bytes() <= budget,
        "held {} bytes over {budget}",
        cache.bytes()
    );
    assert_eq!(cache.bytes(), cache.len() * one);
    cache.put(key(u128::MAX), 0, eval(0.0, budget));
    assert!(
        cache.get(key(u128::MAX), 0).is_none(),
        "an entry past a shard's whole budget is refused"
    );
}

#[test]
fn a_zero_capacity_or_budget_cache_is_off() {
    for cache in [EvalCache::new(0, EVAL_CACHE_BYTES), EvalCache::new(64, 0)] {
        cache.put(key(1), 0, eval(0.3, 4));
        assert!(cache.get(key(1), 0).is_none() && cache.is_empty());
    }
}
