//! The player's served-sims witness (protected): `n` new descents a stone or a stop with the leader fixed.

use mantis_player::batch::WebBatch;
use mantis_player::player::{
    Budget, DeployBlock, GameClock, Player, Pull, StoneResult, TurnResult,
};
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};

/// RESEARCH-WEB's one-CPU-thread desk costs.
const DESK_CPU: mantis_player::player::Costs = mantis_player::player::Costs {
    leaf_ms: 58.71,
    root_ms: 93.0,
};

const GOLDENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/graph_build/positions_r8.txt"
);
const SIMS: u32 = 64;

/// Each golden position's move line, the order its stones were played in.
fn golden_moves(n: usize) -> Vec<Vec<(i32, i32)>> {
    let text = std::fs::read_to_string(GOLDENS).expect("the golden positions fixture is present");
    text.lines()
        .filter_map(|line| line.strip_prefix("M "))
        .take(n)
        .map(|fields| {
            let v: Vec<i32> = fields
                .split_whitespace()
                .map(|f| f.parse().expect("an integer"))
                .collect();
            (0..usize::try_from(v[0]).expect("a count"))
                .map(|i| (v[1 + 2 * i], v[2 + 2 * i]))
                .collect()
        })
        .collect()
}

fn block(tactics: bool) -> DeployBlock {
    DeployBlock {
        c_puct: 1.5,
        fpu_reduction: 0.25,
        quiescence_enabled: true,
        quiescence_blend_2: 0.3,
        leaf_batch_size: 8,
        tactics: tactics.then_some(TacticsConfig {
            leaf_turns: 3,
            leaf_nodes: 256,
            root_turns: 8,
            root_nodes: 20_000,
            audit: Some(AuditConfig {
                turns: 8,
                nodes: 2_000,
                k: 4,
                m: 4,
                total_nodes: 40_000,
                mode: AuditMode::Hold,
            }),
        }),
    }
}

/// A peaked stub net: 0.9 on each graph's first legal row, the rest shared; a value from the graph's size.
fn stub(batch: &WebBatch) -> (Vec<f32>, Vec<f32>) {
    let mut probs = Vec::with_capacity(batch.legal.len());
    let mut values = Vec::with_capacity(batch.n_graphs);
    for meta in &batch.leaves {
        let n = meta.policy_dst_slot.len();
        let rest = if n > 1 { 0.1 / (n - 1) as f32 } else { 0.0 };
        probs.extend((0..n).map(|i| {
            if i == 0 && n > 1 {
                0.9
            } else if n == 1 {
                1.0
            } else {
                rest
            }
        }));
        values.push(((n % 7) as f32 - 3.0) / 10.0);
    }
    (probs, values)
}

fn turn(player: &mut Player, budget: Budget) -> TurnResult {
    player.begin_turn(budget, 0.0).expect("a turn begins");
    loop {
        let outputs = match player.next_batch(1.0).expect("the turn advances") {
            Pull::Turn(result) => return result,
            Pull::Batch(batch) => stub(batch),
        };
        player
            .submit(&outputs.0, &outputs.1, 1.0)
            .expect("the outputs fit the batch");
    }
}

fn assert_spent(stone: &StoneResult, label: &str) {
    if stone.decided {
        return;
    }
    if stone.stopped {
        let left = SIMS - stone.sims;
        let gap = match stone.top.as_slice() {
            [_] => u32::MAX,
            [a, b] => a.visits - b.visits,
            _ => panic!("{label}: a stopped stone has a root"),
        };
        assert!(
            left > 0 && gap > left,
            "{label}: the stop fired with the leader fixed ({gap} > {left})"
        );
    } else {
        assert_eq!(
            stone.sims, SIMS,
            "{label}: spends its budget in new descents"
        );
    }
    assert_eq!(
        stone.root_visits,
        stone.inherited + stone.sims,
        "{label}: the root holds the kept and the new"
    );
    if let Some(t) = stone.tactics {
        assert_eq!(
            t.descents,
            u64::from(stone.sims),
            "{label}: the tactics rows count the same descents"
        );
    }
}

#[test]
fn each_stone_spends_its_budget_or_stops_with_the_leader_fixed_and_the_reroot_keeps_visits() {
    for tactics in [false, true] {
        let mut player = Player::new(block(tactics), "gnn_axis_r8", false).expect("a graph row");
        let (mut fired, mut kept, mut second) = (0, 0, 0);
        for (i, moves) in golden_moves(24).iter().enumerate() {
            player
                .new_game(moves, GameClock::Sims(SIMS))
                .expect("the golden replays");
            let result = turn(&mut player, Budget::Sims(SIMS));
            for (s, stone) in result.stones.iter().enumerate() {
                assert_spent(stone, &format!("golden {i} stone {s} tactics {tactics}"));
                fired += u32::from(stone.stopped);
                if s == 1 && !stone.decided && !result.stones[0].decided {
                    second += 1;
                    kept += u32::from(stone.inherited > 0);
                    let first = &result.stones[0];
                    if let (true, Some(child)) = (
                        stone.inherited > 0,
                        first.top.iter().find(|c| c.cell == first.cell),
                    ) {
                        assert_eq!(
                            stone.inherited, child.visits,
                            "golden {i}: the re-root keeps the chosen child's visits"
                        );
                    }
                }
            }
        }
        assert!(
            fired > 0,
            "the peaked stub never let a leader run away, so the stop arm went unread"
        );
        assert!(
            kept >= 3,
            "the re-root kept visits on {kept} of {second} searched second stones"
        );
        let (rows, _) = player.rows();
        assert!(rows.stop_fired == u64::from(fired) && rows.reuse_reroots == u64::from(kept));
    }
}

#[test]
fn the_same_position_and_outputs_play_the_same_stones() {
    let moves = &golden_moves(3)[2];
    let mut stones = Vec::new();
    for _ in 0..2 {
        let mut player = Player::new(block(true), "gnn_axis_r8", false).expect("a graph row");
        player
            .new_game(moves, GameClock::Sims(SIMS))
            .expect("the golden replays");
        stones.push(
            turn(&mut player, Budget::Sims(SIMS))
                .stones
                .iter()
                .map(|s| s.cell)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(stones[0], stones[1]);
}

#[test]
fn a_short_clock_plays_the_cpu_floor_and_flags_it() {
    use mantis_player::budget::{CPU_FLOOR, FLOOR};
    let moves = &golden_moves(1)[0];
    for (cpu, floor) in [(true, CPU_FLOOR), (false, FLOOR)] {
        let mut player = Player::new(block(false), "gnn_axis_r8", cpu).expect("a graph row");
        player.set_costs(DESK_CPU).expect("a device's costs");
        player
            .new_game(moves, GameClock::TurnMs(10_000.0))
            .expect("the golden replays");
        let result = turn(&mut player, Budget::Clock { turn_ms: 1_000.0 });
        assert!(
            result.floor_over_clock && result.sims_set == floor,
            "cpu {cpu}: {result:?}"
        );
    }
}

#[test]
fn a_call_out_of_step_is_refused_and_an_abort_restores_the_board() {
    let moves = &golden_moves(1)[0];
    let mut player = Player::new(block(false), "gnn_axis_r8", false).expect("a graph row");
    player
        .new_game(moves, GameClock::Sims(SIMS))
        .expect("the golden replays");
    assert!(player.next_batch(0.0).is_err(), "no turn begun");
    assert!(
        player.begin_turn(Budget::Sims(SIMS + 1_000), 0.0).is_err(),
        "past the pool"
    );
    let before = player.board().clone();
    player
        .begin_turn(Budget::Sims(SIMS), 0.0)
        .expect("a turn begins");
    let Pull::Batch(batch) = player.next_batch(0.0).expect("the root step needs the net") else {
        panic!("a fresh game's root step is a miss");
    };
    let (probs, values) = stub(batch);
    assert!(player.next_batch(0.0).is_err(), "a batch is outstanding");
    assert!(
        player.submit(&probs[1..], &values, 0.0).is_err(),
        "a short probs is refused"
    );
    assert_eq!(
        player.board().cells_iter().count(),
        before.cells_iter().count(),
        "a refused submit rolls the turn back"
    );
    player
        .begin_turn(Budget::Sims(SIMS), 0.0)
        .expect("the player stays usable");
    player.abort();
    assert_eq!(
        player.board().cells_iter().count(),
        before.cells_iter().count()
    );
}

/// The pool's cap, read through the refusal a budget past it gets.
fn cap_of(player: &mut Player) -> u32 {
    match player.begin_turn(Budget::Sims(100_000), 0.0) {
        Err(mantis_player::player::PlayerError::BudgetPastPool { cap, .. }) => cap,
        other => panic!("a budget past every pool is refused, got {other:?}"),
    }
}

#[test]
fn every_game_clock_sizes_a_pool_of_at_least_the_floor() {
    use mantis_player::budget::FLOOR;
    let mut player = Player::new(block(false), "gnn_axis_r8", true).expect("a graph row");
    player.set_costs(DESK_CPU).expect("a device's costs");
    for (clock, want) in [
        (GameClock::Sims(1), FLOOR),
        (GameClock::Sims(8), FLOOR),
        (GameClock::Sims(100), 100),
        (GameClock::Sims(5_000), 976),
        (GameClock::TurnMs(f64::NAN), FLOOR),
        (GameClock::TurnMs(-5.0), FLOOR),
        (GameClock::TurnMs(1.0), FLOOR),
        (GameClock::TurnMs(10_000.0), 134),
        (GameClock::TurnMs(f64::INFINITY), 976),
        (GameClock::Unlimited, 976),
    ] {
        player.new_game(&[], clock).expect("an empty opening");
        assert_eq!(cap_of(&mut player), want, "{clock:?}");
    }
}

#[test]
fn a_clock_past_the_pool_plays_the_cap_and_flags_it() {
    let moves = &golden_moves(1)[0];
    let mut player = Player::new(block(false), "gnn_axis_r8", true).expect("a graph row");
    player.set_costs(DESK_CPU).expect("a device's costs");
    player
        .new_game(moves, GameClock::TurnMs(10_000.0))
        .expect("the golden replays");
    let result = turn(&mut player, Budget::Clock { turn_ms: 600_000.0 });
    assert!(
        result.budget_pool_capped && !result.floor_over_clock && result.sims_set == 134,
        "{result:?}"
    );
}

#[test]
fn a_turn_that_served_leaves_sets_the_costs_and_one_that_served_none_keeps_them() {
    use mantis_player::player::Costs;
    let mut player = Player::new(block(false), "gnn_axis_r8", true).expect("a graph row");
    player.set_costs(DESK_CPU).expect("a device's costs");
    player
        .new_game(&[], GameClock::Sims(SIMS))
        .expect("an empty opening");
    let opening = turn(&mut player, Budget::Sims(SIMS));
    assert_eq!(
        (opening.served_leaves, player.costs()),
        (0, Some(DESK_CPU)),
        "the origin serves nothing"
    );
    player
        .begin_turn(Budget::Sims(SIMS), 10.0)
        .expect("a turn begins");
    let result = loop {
        let outputs = match player.next_batch(410.0).expect("the turn advances") {
            Pull::Turn(result) => break result,
            Pull::Batch(batch) => stub(batch),
        };
        player
            .submit(&outputs.0, &outputs.1, 410.0)
            .expect("the outputs fit the batch");
    };
    assert!(result.served_leaves > 0 && result.wall_ms == 400.0);
    #[allow(clippy::cast_precision_loss)]
    let leaf_ms = 400.0 / result.served_leaves as f64;
    assert_eq!(
        player.costs(),
        Some(Costs {
            leaf_ms,
            root_ms: 0.0
        })
    );
    assert!(player
        .set_costs(Costs {
            leaf_ms: f64::NAN,
            root_ms: 0.0
        })
        .is_err());
    assert!(player
        .set_costs(Costs {
            leaf_ms: 0.0,
            root_ms: 0.0
        })
        .is_err());
}

#[test]
fn a_block_or_an_output_the_head_would_refuse_is_refused() {
    let mut zero = block(true);
    zero.leaf_batch_size = 0;
    assert!(
        Player::new(zero, "gnn_axis_r8", false).is_err(),
        "a leaf batch of 0"
    );
    let mut idle = block(true);
    if let Some(t) = idle.tactics.as_mut() {
        t.root_turns = 0;
    }
    assert!(
        Player::new(idle, "gnn_axis_r8", false).is_err(),
        "a root solver of 0 turns"
    );
    let moves = &golden_moves(1)[0];
    let mut player = Player::new(block(false), "gnn_axis_r8", false).expect("a graph row");
    player
        .new_game(moves, GameClock::Sims(SIMS))
        .expect("the golden replays");
    player
        .begin_turn(Budget::Sims(SIMS), 0.0)
        .expect("a turn begins");
    let Pull::Batch(batch) = player.next_batch(0.0).expect("the root step needs the net") else {
        panic!("a fresh game's root step is a miss");
    };
    let (probs, mut values) = stub(batch);
    values[0] = f32::NAN;
    assert!(matches!(
        player.submit(&probs, &values, 0.0),
        Err(mantis_player::player::PlayerError::NonFinite("value"))
    ));
}
