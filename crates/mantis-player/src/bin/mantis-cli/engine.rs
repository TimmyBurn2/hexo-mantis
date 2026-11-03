//! The protocol loop: one game at a time over the step machine, its batches served by `RustForward` on the pool.

use std::io::{BufRead, Write};
use std::time::Instant;

use mantis_core::Board;
use mantis_player::cache::CacheRows;
use mantis_player::player::{
    Budget, GameClock, Player, PlayerError, Pull, Rows, StoneResult, TurnResult,
};

use crate::net::Net;
use crate::CliError;

pub fn parse<T: std::str::FromStr>(word: &str, what: &str) -> Result<T, CliError> {
    word.parse()
        .map_err(|_| CliError::Command(format!("{what}: {word:?} is not a number")))
}

pub fn cells(words: &[&str]) -> Result<Vec<(i32, i32)>, CliError> {
    if !words.len().is_multiple_of(2) {
        return Err(CliError::Command(
            "position: moves come as q r pairs".to_owned(),
        ));
    }
    words
        .chunks(2)
        .map(|c| Ok((parse(c[0], "position")?, parse(c[1], "position")?)))
        .collect()
}

/// One game at a time over `player`, whose board holds `history` until a `newgame` or a position it does not extend.
pub struct Engine {
    pub net: Net,
    pub player: Player,
    pub pool: rayon::ThreadPool,
    pub epoch: Instant,
    pub position: Option<Vec<(i32, i32)>>,
    pub history: Option<Vec<(i32, i32)>>,
    pub last: (Rows, CacheRows),
}

impl Engine {
    fn radius(&self) -> i32 {
        self.player.board().geometry().legal_move_radius
    }

    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() * 1e3
    }

    /// One turn under `budget`, the net served on the pool; a refusal mid-turn rolls the turn back.
    pub fn turn(&mut self, budget: Budget) -> Result<TurnResult, CliError> {
        self.player.begin_turn(budget, self.now())?;
        loop {
            let now = self.now();
            let out = match self.player.next_batch(now)? {
                Pull::Turn(result) => return Ok(result),
                Pull::Batch(batch) => self.net.forward.forward_on(batch, &self.pool),
            };
            match out {
                Ok(out) => self.player.submit(&out.probs, &out.value, self.now())?,
                Err(e) => {
                    self.player.abort();
                    return Err(CliError::Forward(e));
                }
            }
        }
    }

    fn go(&mut self, words: &[&str]) -> Result<TurnResult, CliError> {
        let (budget, clock) = match words {
            ["nodes", n] => match parse::<u32>(n, "go nodes")? {
                0 => return Err(CliError::Command("go nodes: 0 sims".to_owned())),
                n => (Budget::Sims(n), GameClock::Sims(n)),
            },
            ["movetime", t] => {
                let t = f64::from(parse::<u32>(t, "go movetime")?);
                (Budget::Clock { turn_ms: t }, GameClock::TurnMs(t))
            }
            _ => return Err(CliError::Command("go: nodes N or movetime T".to_owned())),
        };
        let moves = self
            .position
            .clone()
            .ok_or_else(|| CliError::Command("go: no position".to_owned()))?;
        let extends = self
            .history
            .as_ref()
            .filter(|h| moves.starts_with(h))
            .map(Vec::len);
        let from = extends.unwrap_or(0);
        let mut board = match extends {
            Some(_) => self.player.board().clone(),
            None => Board::with_geometry(self.player.board().geometry()),
        };
        // The rules the arena plays by, which the step machine leaves to its host: a legal cell, no stone after a win.
        for &(q, r) in &moves[from..] {
            if board.check_win() || !board.legal_moves_set().contains(&(q, r)) {
                return Err(CliError::Command(format!(
                    "position: ({q}, {r}) is not a legal stone after {} moves",
                    board.cells_iter().count()
                )));
            }
            board
                .apply_move(q, r)
                .map_err(|why| CliError::Command(why.to_owned()))?;
        }
        if extends.is_some() {
            self.player.play(&moves[from..])?;
        } else {
            self.history = None;
            self.player.new_game(&moves, clock)?;
        }
        self.history = Some(moves.clone());
        if self.player.board().check_win() {
            return Err(CliError::Command("go: the game is over".to_owned()));
        }
        let result = match self.turn(budget) {
            Err(CliError::Engine(PlayerError::BudgetPastPool { .. })) => {
                self.player.new_game(&moves, clock)?;
                self.turn(budget)?
            }
            other => other?,
        };
        if let Some(h) = self.history.as_mut() {
            h.extend(result.stones.iter().map(|s| s.cell));
        }
        Ok(result)
    }

    /// The turn's own counters to stderr, one line: its stones' sims, its served leaves, its wall and its rows.
    fn report(&mut self, t: &TurnResult) {
        let (r, c) = self.player.rows();
        let (r0, c0) = std::mem::replace(&mut self.last, (r, c));
        let join = |f: &dyn Fn(&StoneResult) -> String| {
            t.stones.iter().map(f).collect::<Vec<_>>().join(",")
        };
        eprintln!(
            "mantis-cli turn sims={} stopped={} decided={} inherited={} sims_set={} served={} wall_ms={:.1} \
             floor_over_clock={} pool_capped={} cache_hits={} cache_calls={} reroots={} stop_fired={} leaf_ms={}",
            join(&|s| s.sims.to_string()),
            join(&|s| u8::from(s.stopped).to_string()),
            join(&|s| u8::from(s.decided).to_string()),
            join(&|s| s.inherited.to_string()),
            t.sims_set,
            t.served_leaves,
            t.wall_ms,
            u8::from(t.floor_over_clock),
            u8::from(t.budget_pool_capped),
            c.hits - c0.hits,
            c.calls - c0.calls,
            r.reuse_reroots - r0.reuse_reroots,
            r.stop_fired - r0.stop_fired,
            self.player.costs().map_or(f64::NAN, |c| c.leaf_ms),
        );
    }

    pub fn serve(&mut self, input: impl BufRead, mut out: impl Write) -> Result<(), CliError> {
        for line in input.lines() {
            let line = line?;
            let words: Vec<&str> = line.split_whitespace().collect();
            let reply: Result<Vec<String>, CliError> = match words.as_slice() {
                [] => continue,
                ["quit"] => break,
                ["six"] => Ok(vec![
                    format!("id name mantis-cli {}", env!("CARGO_PKG_VERSION")),
                    format!(
                        "id net {} manifest {}",
                        self.net.stem, self.net.manifest_sha
                    ),
                    "sixok".to_owned(),
                ]),
                ["isready"] => Ok(vec!["readyok".to_owned()]),
                ["newgame"] => {
                    self.history = None;
                    Ok(Vec::new())
                }
                ["position", "radius", r, rest @ ..] => {
                    self.position = None;
                    let moves = match rest {
                        [] => Ok(Vec::new()),
                        ["moves", flat @ ..] => cells(flat),
                        _ => Err(CliError::Command(
                            "position: radius R [moves q r ...]".to_owned(),
                        )),
                    };
                    match (parse::<i32>(r, "position radius"), moves) {
                        (Ok(r), Ok(moves)) if r == self.radius() => {
                            self.position = Some(moves);
                            Ok(Vec::new())
                        }
                        (Ok(r), Ok(_)) => Err(CliError::Command(format!(
                            "position: radius {r} is not the net's ({})",
                            self.radius()
                        ))),
                        (Err(e), _) | (_, Err(e)) => Err(e),
                    }
                }
                ["go", rest @ ..] => match self.go(rest) {
                    Ok(t) => {
                        self.report(&t);
                        let flat: Vec<String> = t
                            .stones
                            .iter()
                            .map(|s| format!("{} {}", s.cell.0, s.cell.1))
                            .collect();
                        Ok(vec![format!("bestmove {}", flat.join(" "))])
                    }
                    // A driver reads through `bestmove`, so a refused search still answers one.
                    Err(e) => Ok(vec![format!("error {e}"), "bestmove none".to_owned()]),
                },
                [word, ..] => Err(CliError::Command(format!("unknown command {word:?}"))),
            };
            match reply {
                Ok(lines) => lines.iter().try_for_each(|l| writeln!(out, "{l}"))?,
                Err(e) => writeln!(out, "error {e}")?,
            }
            out.flush()?;
        }
        Ok(())
    }
}
