// >300 justify (R8): the binary's witnesses share one fixture set — the tiny net and its manifest, the stdio driver,
// the player's own turn and the goldens — that a split would duplicate.

//! The binary's witnesses on a tiny net of the shipped row's shape: the player's answers, the rules, the clock, the threads.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::json;
use sha2::{Digest, Sha256};

use mantis_player::batch::{build_batch, Position};
use mantis_player::budget::{pool_sims, sims_for_turn, CPU_FLOOR, FLOOR};
use mantis_player::forward::{RustForward, TensorRow};
use mantis_player::player::{Budget, DeployBlock, GameClock, Player, Pull};
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};

const ENC: &str = "gnn_axis_r8";
const SIMS: u32 = 32;
/// The sha256 of the tiny net's one-thread outputs on 16 goldens (logits, probs, value, LE bits), from a build of f93d6cc6.
const PINNED_FORWARD_SHA: &str = "752523b94f7f48cb5921f528074ea1d56b9f53d69f43a585b3a5baee8f04c33b";
const GOLDENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/graph_build/positions_r8.txt"
);
/// One binary at a time, so no test's clock reads another's load.
static SERIAL: Mutex<()> = Mutex::new(());

/// The deploy block of record, as the manifest below carries it.
fn block() -> DeployBlock {
    DeployBlock {
        c_puct: 1.5,
        fpu_reduction: 0.25,
        quiescence_enabled: true,
        quiescence_blend_2: 0.3,
        leaf_batch_size: 8,
        tactics: Some(TacticsConfig {
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

/// A two-layer net of width 8 over the row's 11 features and 91 edge codes, its floats a fixed pseudo-random draw.
fn tiny() -> (Vec<TensorRow>, Vec<f32>) {
    let (h, depth) = (32, 4);
    let mut shapes = vec![
        ("input_proj.wt".to_owned(), 11, h),
        ("input_proj.b".to_owned(), 1, h),
    ];
    for i in 0..depth {
        for (name, rows, cols) in [
            ("norm{}.w", 1, h),
            ("norm{}.b", 1, h),
            ("table{}", 91, h),
            ("eps{}", 1, 1),
            ("mlp{}.0.wt", h, 2 * h),
            ("mlp{}.0.b", 1, 2 * h),
            ("mlp{}.2.wt", 2 * h, h),
            ("mlp{}.2.b", 1, h),
        ] {
            shapes.push((name.replace("{}", &i.to_string()), rows, cols));
        }
    }
    for (name, rows, cols) in [
        ("final_norm.w", 1, h),
        ("final_norm.b", 1, h),
        ("policy.0.wt", depth * h, 4),
        ("policy.0.b", 1, 4),
        ("policy.2.wt", 4, 1),
        ("policy.2.b", 1, 1),
        ("fc1.wt", 2 * depth * h, 4),
        ("fc1.b", 1, 4),
        ("fc2.wt", 4, 5),
        ("fc2.b", 1, 5),
        ("support", 1, 5),
    ] {
        shapes.push((name.to_owned(), rows, cols));
    }
    let mut table = Vec::new();
    let mut offset = 0;
    for (name, rows, cols) in shapes {
        table.push(TensorRow {
            name,
            rows,
            cols,
            offset,
        });
        offset += rows * cols;
    }
    let mut state = 0x2545_f491_u32;
    let floats = (0..offset)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).expect("16 bits")) / 65_536.0 - 0.5
        })
        .collect();
    (table, floats)
}

fn forward() -> RustForward {
    let (table, floats) = tiny();
    RustForward::load(&table, &floats, 11).expect("the tiny table is one served net")
}

/// The tiny net's weights file and its manifest in a fresh directory; `weights_sha` overrides the bound sha256.
fn write_net(label: &str, weights_sha: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mantis-cli-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let (table, floats) = tiny();
    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(dir.join("tiny.weights.bin"), &bytes).expect("the weights file is written");
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let tensors: Vec<_> = table
        .iter()
        .map(|t| json!({"name": t.name, "rows": t.rows, "cols": t.cols, "offset": t.offset}))
        .collect();
    let manifest = json!({
        "format": "mantis-web-manifest-v2",
        "encoding": {"name": ENC, "registry_sha": mantis_encoding::registry_sha_hex(), "node_feat_dim": 11},
        "deploy": {
            "resolved": {"search_kind": "puct", "c_puct": 1.5, "fpu_reduction": 0.25, "quiescence_enabled": true,
                "quiescence_blend_2": 0.3, "leaf_batch_size": 8, "tactics": {"kind": "strict_turn", "leaf_turns": 3,
                "leaf_nodes": 256, "root_turns": 8, "root_nodes": 20_000, "audit": {"turns": 8, "nodes": 2_000,
                "k": 4, "m": 4, "total_nodes": 40_000, "mode": "hold"}}},
            "player": {"early_stop": true, "tree_reuse": true, "cache_entries": 8192, "virtual_loss": 1.0},
        },
        "files": [{"name": "tiny.weights.bin", "sha256": weights_sha.unwrap_or(&sha), "bytes": bytes.len(),
            "tensors": tensors, "passes_its_line": true}],
    });
    let path = dir.join("tiny.manifest.json");
    std::fs::write(&path, manifest.to_string()).expect("the manifest is written");
    path
}

/// The binary on stdio: protocol lines out on stdout, its counters on stderr.
struct Cli {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
}

fn field(line: &str, key: &str) -> f64 {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(&format!("{key}=")))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no number {key}= in {line:?}"))
}

impl Cli {
    /// The binary on `manifest` at `threads`, and its start-up line.
    fn start(manifest: &Path, threads: usize) -> (Cli, String) {
        let mut child = Command::new(env!("CARGO_BIN_EXE_mantis-cli"))
            .args([
                "--net",
                &manifest.to_string_lossy(),
                "--threads",
                &threads.to_string(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary starts");
        let mut cli = Cli {
            stdin: child.stdin.take().expect("piped"),
            stdout: BufReader::new(child.stdout.take().expect("piped")),
            stderr: BufReader::new(child.stderr.take().expect("piped")),
            child,
        };
        let ready = cli.err_line();
        if let Some(dir) = manifest.parent() {
            let _ = std::fs::remove_dir_all(dir); // the binary has read its net
        }
        (cli, ready)
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("the binary reads its stdin");
    }

    fn line(&mut self) -> String {
        let mut s = String::new();
        self.stdout.read_line(&mut s).expect("a stdout line");
        s.trim().to_owned()
    }

    fn err_line(&mut self) -> String {
        let mut s = String::new();
        self.stderr.read_line(&mut s).expect("a stderr line");
        s.trim().to_owned()
    }

    /// `position` then `go`: every stdout line through `bestmove`, the turn's counters line, and the wall ms.
    fn go(&mut self, moves: &[(i32, i32)], go: &str) -> (Vec<String>, Option<String>, f64) {
        let flat: String = moves.iter().map(|(q, r)| format!(" {q} {r}")).collect();
        let t0 = Instant::now();
        self.send(&format!("position radius 8 moves{flat}\n{go}"));
        let mut lines = vec![self.line()];
        while !lines[lines.len() - 1].starts_with("bestmove") {
            lines.push(self.line());
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let counters = (lines[lines.len() - 1] != "bestmove none").then(|| self.err_line());
        (lines, counters, ms)
    }

    fn cells(&mut self, moves: &[(i32, i32)], go: &str) -> Vec<(i32, i32)> {
        let (lines, _, _) = self.go(moves, go);
        let words: Vec<i32> = lines[lines.len() - 1]
            .split_whitespace()
            .skip(1)
            .map(|w| w.parse().expect("a cell coordinate"))
            .collect();
        words.chunks(2).map(|c| (c[0], c[1])).collect()
    }
}

impl Drop for Cli {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

/// The player's own turn, its batches through the same forward.
fn turn(player: &mut Player, net: &RustForward, budget: Budget) -> Vec<(i32, i32)> {
    player.begin_turn(budget, 0.0).expect("a turn begins");
    loop {
        let out = match player.next_batch(0.0).expect("the turn advances") {
            Pull::Turn(result) => return result.stones.iter().map(|s| s.cell).collect(),
            Pull::Batch(batch) => net.forward(batch).expect("a well-formed batch"),
        };
        player
            .submit(&out.probs, &out.value, 0.0)
            .expect("the outputs fit");
    }
}

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

/// A cell as the served path requests it, as the player's own `position_of` maps it.
fn side_of(c: mantis_core::Cell) -> i64 {
    match c {
        mantis_core::Cell::Empty => 0,
        mantis_core::Cell::P1 => 1,
        mantis_core::Cell::P2 => -1,
    }
}

/// The other side's two stones, outside the engine: the board's two smallest legal cells.
fn opponent(player: &Player) -> Vec<(i32, i32)> {
    let mut legal = player.board().legal_moves();
    legal.sort_unstable();
    legal.truncate(usize::from(player.board().moves_remaining));
    legal
}

#[test]
fn go_nodes_is_the_turns_budget_and_the_answers_are_the_players_stones_and_a_swapped_stone_order_answers_otherwise(
) {
    let _one = SERIAL.lock();
    let (mut cli, ready) = Cli::start(&write_net("parity", None), 2);
    assert!(ready.starts_with("mantis-cli ready net=tiny"), "{ready}");
    let net = forward();
    let mut player = Player::new(block(), ENC, true).expect("the block of record");
    let mut turns = 0;
    for moves in golden_moves(8) {
        cli.send("newgame");
        player
            .new_game(&moves, GameClock::Sims(SIMS))
            .expect("the golden replays");
        let mut line = moves.clone();
        for _ in 0..2 {
            if player.board().check_win() {
                break;
            }
            let due = u32::from(player.board().moves_remaining);
            let want = turn(&mut player, &net, Budget::Sims(SIMS));
            assert_eq!(
                cli.cells(&line, &format!("go nodes {}", SIMS * due)),
                want,
                "{line:?}"
            );
            line.extend(&want);
            turns += 1;
            if player.board().check_win() {
                break;
            }
            let theirs = opponent(&player);
            player
                .play(&theirs)
                .expect("the opponent's stones are legal");
            line.extend(theirs);
        }
    }
    assert!(turns >= 12, "the games ran {turns} turns");

    // PLANTED BREAK: a golden's last stone swapped with one of the turn before belong to the other sides: the answer parts.
    let moves = &golden_moves(2)[1];
    let mut swapped = moves.clone();
    swapped.swap(moves.len() - 1, moves.len() - 3);
    player
        .new_game(moves, GameClock::Sims(SIMS))
        .expect("the golden replays");
    let want = turn(&mut player, &net, Budget::Sims(SIMS));
    cli.send("newgame");
    assert_ne!(
        cli.cells(&swapped, &format!("go nodes {}", 2 * SIMS)),
        want,
        "the stone order went unread"
    );
}

#[test]
fn a_refused_cell_or_command_answers_error_and_leaves_the_game_as_it_was() {
    let _one = SERIAL.lock();
    let (mut cli, _) = Cli::start(&write_net("refused", None), 1);
    let refused = |cli: &mut Cli, moves: &[(i32, i32)], go: &str| {
        let (lines, counters, _) = cli.go(moves, go);
        assert!(
            lines[0].starts_with("error "),
            "{go} on {moves:?}: {lines:?}"
        );
        assert_eq!(lines[lines.len() - 1], "bestmove none");
        assert!(counters.is_none());
    };
    refused(&mut cli, &[(0, 0), (0, 0)], "go nodes 32");
    refused(&mut cli, &[(0, 0), (9, 0)], "go nodes 32");
    refused(&mut cli, &[(0, 0)], "go nodes 0");
    refused(&mut cli, &[(0, 0)], "go depth 3");
    refused(&mut cli, &[(0, 0)], "go nodes 1");
    refused(&mut cli, &[(0, 0)], "go nodes 1954");
    for line in [
        "position radius 7",
        "position radius 8 moves 1",
        "setoption cacheEntries 0",
        "uci",
    ] {
        cli.send(line);
        assert!(cli.line().starts_with("error "), "{line}");
    }
    cli.stdin
        .write_all(b"position \xff\n")
        .expect("the binary reads its stdin");
    assert_eq!(cli.line(), "error a line that is not UTF-8");
    cli.send("position moves 0 0\ngo nodes 64");
    assert!(
        cli.line().starts_with("bestmove "),
        "Six's default radius is 8"
    );
    let _ = cli.err_line();
    cli.send("position radius 7");
    assert!(cli.line().starts_with("error "));
    cli.send("go nodes 32");
    assert_eq!(
        cli.line(),
        "error go: no position",
        "a refused position leaves none to play"
    );
    assert_eq!(cli.line(), "bestmove none");

    // A live game: a refused extension changes nothing, so the legal one next is answered as the player answers it.
    let net = forward();
    let mut player = Player::new(block(), ENC, true).expect("the block of record");
    let moves = golden_moves(3)[2].clone();
    player
        .new_game(&moves, GameClock::Sims(SIMS))
        .expect("the golden replays");
    cli.send("newgame");
    let mut line = moves.clone();
    let ours = turn(&mut player, &net, Budget::Sims(SIMS));
    assert_eq!(cli.cells(&line, &format!("go nodes {}", 2 * SIMS)), ours);
    line.extend(&ours);
    let mut bad = line.clone();
    bad.extend([line[0], line[1]]);
    refused(&mut cli, &bad, &format!("go nodes {}", 2 * SIMS));
    let theirs = opponent(&player);
    player.play(&theirs).expect("legal");
    line.extend(theirs);
    let want = turn(&mut player, &net, Budget::Sims(SIMS));
    assert_eq!(
        cli.cells(&line, &format!("go nodes {}", 2 * SIMS)),
        want,
        "the refused stones leaked"
    );
}

#[test]
fn a_movetime_turn_spends_the_budget_rule_within_its_clock_and_six_clocks_read_as_an_overrun() {
    let _one = SERIAL.lock();
    let (mut cli, ready) = Cli::start(&write_net("movetime", None), 1);
    let (mut leaf, min_turn) = (field(&ready, "leaf_ms"), field(&ready, "min_turn_ms"));
    let t = (4.0 * min_turn).round();
    // An opening's first five stones: no turn of the game below is decided before it is searched.
    let mut line = golden_moves(50)[49][..5].to_vec();
    let start = line.clone();
    let cap = pool_sims(t, 0.0, leaf);
    cli.send("newgame");
    for _ in 0..4 {
        let (lines, counters, ms) = cli.go(&line, &format!("go movetime {t}"));
        let counters = counters.unwrap_or_else(|| panic!("{lines:?}"));
        let want = sims_for_turn(t, 0.0, leaf, 2, FLOOR, cap);
        assert_eq!(
            field(&counters, "sims_set"),
            f64::from(want),
            "the budget rule at {t} ms: {counters}"
        );
        assert_eq!(field(&counters, "floor_over_clock"), 0.0);
        assert!(
            ms <= t,
            "a {ms:.0} ms turn overran its {t} ms clock: {counters}"
        );
        leaf = field(&counters, "leaf_ms");
        let words: Vec<i32> = lines[lines.len() - 1]
            .split_whitespace()
            .skip(1)
            .map(|w| w.parse().expect("a cell"))
            .collect();
        if words.len() < 4 {
            break;
        }
        line.extend(words.chunks(2).map(|c| (c[0], c[1])));
    }

    // PLANTED BREAK: an engine given six times the clock overruns it by more than any turn-to-turn swing in its cost, and the line above reads that.
    cli.send("newgame");
    let (_, counters, ms) = cli.go(&start, &format!("go movetime {}", 6.0 * t));
    assert!(
        ms > t,
        "a turn given {} ms read no overrun of {t} ms: {counters:?}",
        6.0 * t
    );
    // A clock below the floor's plays the CPU floor anyway and flags it.
    cli.send("newgame");
    let (_, counters, _) = cli.go(&start, "go movetime 1");
    let counters = counters.expect("a turn at the floor");
    assert_eq!(field(&counters, "sims_set"), f64::from(CPU_FLOOR));
    assert_eq!(field(&counters, "floor_over_clock"), 1.0);
}

#[test]
fn a_weights_file_the_manifest_does_not_bind_is_refused_at_start() {
    let _one = SERIAL.lock();
    let manifest = write_net("unbound", Some(&"0".repeat(64)));
    let out = Command::new(env!("CARGO_BIN_EXE_mantis-cli"))
        .args(["--net", &manifest.to_string_lossy()])
        .stdin(Stdio::null())
        .output()
        .expect("the binary runs");
    if let Some(dir) = manifest.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("weights: tiny.weights.bin is not the file the manifest binds"),
        "{stderr}"
    );
}

#[test]
fn the_one_thread_forward_is_the_pinned_one_and_the_threaded_pass_is_it_bit_for_bit_and_reversed_edges_change_the_bits(
) {
    let _one = SERIAL.lock();
    let spec = mantis_encoding::lookup(ENC).expect("the row");
    let mut player = Player::new(block(), ENC, true).expect("the block of record");
    let positions: Vec<Position> = golden_moves(16)
        .iter()
        .map(|moves| {
            player
                .new_game(moves, GameClock::Sims(SIMS))
                .expect("the golden replays");
            let b = player.board();
            let stones = b
                .cells_iter()
                .map(|(&(q, r), &c)| (i64::from(q), i64::from(r), side_of(c)))
                .collect();
            let side = if b.current_player == mantis_core::Player::One {
                1
            } else {
                -1
            };
            (stones, side, i64::from(b.moves_remaining))
        })
        .collect();
    let batch = build_batch(&positions, spec).expect("the goldens batch");
    let net = forward();
    let one = net.forward(&batch).expect("a well-formed batch");
    let bits: Vec<u8> = [&one.logits, &one.probs, &one.value]
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let sha: String = Sha256::digest(&bits)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    // The one-thread pass's bits as the forward before `forward_on` (f93d6cc6) computed them on this batch.
    assert_eq!(sha, PINNED_FORWARD_SHA, "the one-thread forward moved");
    for threads in [2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        assert_eq!(
            net.forward_on(&batch, &pool).expect("the same batch"),
            one,
            "{threads} threads"
        );
    }
    // PLANTED BREAK: the same edges listed in reverse sum each node in another order, which bit equality reads.
    let mut reversed = batch.clone();
    for v in [&mut reversed.src, &mut reversed.dst, &mut reversed.code] {
        v.reverse();
    }
    assert_ne!(
        net.forward(&reversed).expect("the same graph"),
        one,
        "a reordered sum went unread"
    );
}

#[test]
fn a_winning_first_stone_answers_one_cell_and_a_won_game_refuses_another_stone_and_a_search() {
    let _one = SERIAL.lock();
    let (mut cli, _) = Cli::start(&write_net("won", None), 1);
    // x holds five in a row, open at both ends, with two stones to play.
    let five = vec![
        (0, 0),
        (0, 3),
        (5, -3),
        (1, 0),
        (2, 0),
        (-3, 3),
        (3, 3),
        (3, 0),
        (4, 0),
        (-4, -2),
        (6, -6),
    ];
    let reply = cli.cells(&five, &format!("go nodes {}", 2 * SIMS));
    assert_eq!(
        reply.len(),
        1,
        "a first stone that wins is the whole answer: {reply:?}"
    );
    assert!(reply[0] == (5, 0) || reply[0] == (-1, 0), "{reply:?}");
    let mut won = five.clone();
    won.push(reply[0]);
    let (lines, _, _) = cli.go(&won, &format!("go nodes {}", 2 * SIMS));
    assert_eq!(lines, ["error go: the game is over", "bestmove none"]);
    won.push((6, 1));
    let (lines, _, _) = cli.go(&won, &format!("go nodes {}", 2 * SIMS));
    assert!(
        lines[0].starts_with("error position: (6, 1) is not a legal stone"),
        "{lines:?}"
    );
    assert_eq!(lines[1], "bestmove none");
}
