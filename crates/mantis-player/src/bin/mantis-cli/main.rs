//! `mantis-cli`: the deploy engine on stdio in Six's text protocol, over the net's weights file and manifest.

use std::fmt;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use mantis_player::budget::FLOOR;
use mantis_player::forward::ForwardError;
use mantis_player::player::{Budget, GameClock, Player, PlayerError};

use engine::{cells, Engine};
use net::{beside_binary, load};

mod engine;
mod net;

const USAGE: &str = "usage: mantis-cli [--net STEM.manifest.json] [--threads N]";
/// A 62-stone golden (`positions_r8.txt` #49) the start-up turn measures the device's cost a leaf on.
const WARMUP: &str = "0 -1 1 -7 -1 0 -7 4 -6 4 -8 4 -5 3 -5 5 -7 6 -6 5 -7 3 -9 6 -9 5 -9 7 -10 7 2 -2 2 -4 2 -1 0 -2 \
-8 3 4 -5 3 -4 -4 1 3 -6 -7 7 -4 0 -6 6 -2 0 -3 -1 -2 -1 2 -5 -4 3 -11 6 1 -8 -10 5 4 -6 5 -7 6 -8 6 -6 -3 4 -2 4 -4 4 \
1 -9 1 -6 -3 5 2 -8 1 -10 1 -11 -10 6 -12 6 4 -8 3 -9 3 -8 0 -6 -1 -5 -7 8 -3 -3 -7 9 -3 6 0 2 -1 3 -4 6";

/// Why the binary would not start, or refused one command (answered as `error <why>`).
#[derive(Debug)]
enum CliError {
    Usage(String),
    Manifest(String),
    Weights(String),
    Io(String),
    Command(String),
    Engine(PlayerError),
    Forward(ForwardError),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(why) => write!(f, "{why}\n{USAGE}"),
            Self::Manifest(why) => write!(f, "manifest: {why}"),
            Self::Weights(why) => write!(f, "weights: {why}"),
            Self::Io(why) | Self::Command(why) => write!(f, "{why}"),
            Self::Engine(e) => write!(f, "{}: {e}", e.name()),
            Self::Forward(e) => write!(f, "{e}"),
        }
    }
}

impl From<PlayerError> for CliError {
    fn from(e: PlayerError) -> Self {
        Self::Engine(e)
    }
}

impl From<io::Error> for CliError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

fn start() -> Result<Engine, CliError> {
    let mut args = std::env::args().skip(1);
    let (mut net, mut threads) = (None, None);
    let bad = |v: &str| CliError::Usage(format!("--threads {v:?} is not a count"));
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| CliError::Usage(format!("{arg} needs a value")))?;
        match arg.as_str() {
            "--net" => net = Some(PathBuf::from(value)),
            "--threads" => threads = Some(value.parse::<usize>().map_err(|_| bad(&value))?),
            _ => return Err(CliError::Usage(format!("unknown argument {arg}"))),
        }
    }
    let net = match net {
        Some(path) => path,
        None => beside_binary()?,
    };
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    if threads == Some(0) {
        return Err(bad("0"));
    }
    let threads = threads.unwrap_or(4).min(cores);
    let net = load(&net)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| CliError::Io(format!("thread pool: {e}")))?;
    let player = Player::new(net.block, &net.encoding, true)?;
    let mut engine = Engine {
        net,
        player,
        pool,
        epoch: Instant::now(),
        position: None,
        history: None,
        last: Default::default(),
    };
    let warmup = cells(&WARMUP.split_whitespace().collect::<Vec<_>>())?;
    engine.player.new_game(&warmup, GameClock::Sims(FLOOR))?;
    engine.turn(Budget::Sims(FLOOR))?;
    engine.last = engine.player.rows();
    let costs = engine
        .player
        .costs()
        .ok_or_else(|| CliError::Io("the warm-up served no leaf".to_owned()))?;
    eprintln!(
        "mantis-cli ready net={} manifest_sha256={} weights_sha256={} threads={threads} leaf_ms={} min_turn_ms={:.0}",
        engine.net.stem,
        engine.net.manifest_sha,
        engine.net.weights_sha,
        costs.leaf_ms,
        engine.player.min_turn_ms().unwrap_or(f64::NAN),
    );
    Ok(engine)
}

fn main() -> ExitCode {
    let started =
        start().and_then(|mut engine| engine.serve(io::stdin().lock(), io::stdout().lock()));
    match started {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mantis-cli: {e}");
            ExitCode::from(2)
        }
    }
}
