"""The Six rung adapter: the pinned `sixengine` release and a pinned network over the Six protocol, one process per concurrent game."""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mantis.bots.protocol import RungUnresolvable
from mantis.bots.strix import find_vendor_root
from mantis.util.hashing import sha256_file

PIN_NAME = "six"
ENGINE_ASSET = "engine"
#: Every pinned asset named with this prefix is the engine's ONNX Runtime, re-hashed with it at every start.
RUNTIME_PREFIX = "runtime"
#: Six's expansion cache survives `newgame`, so play could depend on earlier games; the ruler runs it off.
CACHE_ENTRIES = 0

VENDOR_ABSENT_MARKER = "six asset not fetched"
PIN_ABSENT_MARKER = "six pin absent"
SHA_MISMATCH_MARKER = "six asset sha256 mismatch"
PROVIDER_MARKER = "six execution provider refused"

#: Every forfeit is logged under this marker; `tools/strength_frontier.py` counts the lines into the cell record.
FINDING_LOG_MARKER = "six_forfeit_finding"
#: Every engine start logs its provider under this marker, which the frontier reads into the receipt.
PROVIDER_LOG_MARKER = "six_engine_provider"
#: Every bot logs its counters under this marker when it closes.
CLOSE_LOG_MARKER = "six_engine_closed"
#: The engine's `id name` when it loaded a network; without one it plays alpha-beta, which is not the rung.
NET_IDENT = "id name HexBot Net"
#: An empty board's forfeit cell: outside every legal ball, so the arena refuses it.
_OFF_BOARD = (1 << 20, 1 << 20)
_DEVICES = ("cuda", "cpu")
_COORD = re.compile(r"-?[0-9]+")
_LOG = logging.getLogger(__name__)


class SixEngineError(RuntimeError):
    """The engine process closed its output, or refused a setting it was sent."""


class SixSyncError(RuntimeError):
    """The observed move log is not the board's stones: the arena did not report a stone."""


@dataclass(frozen=True)
class SixReply:
    """One `go`'s answer: the turn's stones in order, or why it is not the engine's own search."""

    stones: tuple[tuple[int, int], ...]
    failure: str | None


def parse_reply(lines: Sequence[str]) -> SixReply:
    """The stones of the `bestmove` line; an `error` line, a failed search, `bestmove none` or a malformed answer is a failure."""
    failures = [ln for ln in lines if ln.startswith(("error", "info string search failed"))]
    words = lines[-1].split()[1:] if lines and lines[-1].startswith("bestmove") else ["none"]
    stones: tuple[tuple[int, int], ...] = ()
    if words == ["none"]:
        failures.append("bestmove none")
    elif len(words) % 2 or not all(_COORD.fullmatch(w) for w in words):
        failures.append(f"malformed answer {lines[-1]!r}")
    else:
        stones = tuple((int(words[i]), int(words[i + 1])) for i in range(0, len(words), 2))
    return SixReply(stones=stones if not failures else (), failure="; ".join(failures) or None)


def provider_of(stderr_lines: Sequence[str], device: str) -> str:
    """The provider the engine runs its network on: a CUDA request reports only its fallback, on stderr."""
    if device == "cpu":
        return "cpu"
    if not any(ln.startswith("CUDA is not available") for ln in stderr_lines):
        return "cuda"
    return "webgpu" if any(ln.startswith("using WebGPU") for ln in stderr_lines) else "cpu"


@dataclass(frozen=True)
class SixAssets:
    """The pinned engine and network on disk, each verified against the pin's sha256."""

    engine: Path
    engine_sha256: str
    net: Path
    net_sha256: str
    variant: str
    commit: str
    #: the pinned runtime members' sha256s in asset-name order, comma-joined; "" when the pin names none.
    runtime_sha256: str = ""


def _asset(root: Path, pin: dict[str, Any], name: str) -> tuple[Path, str]:
    spec = pin.get("assets", {})[name]
    path = root / "external" / spec["path"]
    if not path.is_file():
        raise RungUnresolvable(rung="six", reason=(
            f"{VENDOR_ABSENT_MARKER}: {spec['path']} is not under vendor/external; run `make vendor.six`"))
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise RungUnresolvable(rung="six", reason=(
            f"{SHA_MISMATCH_MARKER}: {path.name} hashes to {digest} but vendor/pins.toml pins {spec['sha256']}; "
            "the rung plays the pinned bytes or nothing"))
    return path, digest


def _pin(vendor_root: Path) -> dict[str, Any] | None:
    return tomllib.loads((vendor_root / "pins.toml").read_text(encoding="utf-8")).get("pins", {}).get(PIN_NAME)


def pin_record(variant: str) -> dict[str, str | None]:
    """What the pin declares for the engine and the `variant` network (a receipt's provenance), `None` where it declares nothing."""
    root = find_vendor_root()
    pin = (None if root is None else _pin(root)) or {}
    assets = pin.get("assets", {})
    engine, net = (assets.get(name, {}) for name in (ENGINE_ASSET, variant))
    return {"commit": pin.get("sha"), "engine_sha256": engine.get("sha256"),
            "net": Path(net["path"]).name if "path" in net else None, "net_sha256": net.get("sha256"),
            "runtime_sha256": ",".join(assets[n]["sha256"] for n in _runtime_names(assets))}


def _runtime_names(assets: dict[str, Any]) -> list[str]:
    return sorted(n for n in assets if n.startswith(RUNTIME_PREFIX))


def locate_six(vendor_root: Path | None, variant: str) -> SixAssets:
    """The engine, its pinned runtime and the `variant` network, re-hashed against the pin; Raises: RungUnresolvable on no root or pin, an unpinned network, an unfetched asset or a sha256 mismatch."""
    if vendor_root is None:
        raise RungUnresolvable(rung="six", reason=(
            f"{VENDOR_ABSENT_MARKER}: no ancestor of the installed package holds vendor/pins.toml"))
    pin = _pin(vendor_root)
    if pin is None:
        raise RungUnresolvable(rung="six", reason=f"{PIN_ABSENT_MARKER}: vendor/pins.toml declares no [pins.{PIN_NAME}]")
    networks = sorted(n for n, a in pin.get("assets", {}).items() if str(a.get("path", "")).endswith(".onnx"))
    if variant not in networks:
        raise RungUnresolvable(rung=f"six:{variant}", reason=f"the pin names the networks {networks}; not {variant!r}")
    engine, engine_sha = _asset(vendor_root, pin, ENGINE_ASSET)
    runtime = [_asset(vendor_root, pin, name)[1] for name in _runtime_names(pin.get("assets", {}))]
    net, net_sha = _asset(vendor_root, pin, variant)
    return SixAssets(engine=engine, engine_sha256=engine_sha, net=net, net_sha256=net_sha, variant=variant,
                     commit=str(pin["sha"]), runtime_sha256=",".join(runtime))


def six_availability(variant: str) -> tuple[bool, str]:
    """`(available, why-not)` for the vendored engine and the `variant` network."""
    try:
        locate_six(find_vendor_root(), variant)
    except RungUnresolvable as exc:
        return False, exc.reason
    return True, ""


def cuda_library_path() -> str | None:
    """The venv's NVIDIA wheel library directories (cuDNN, cuBLAS, the CUDA runtime), or None on a CPU venv."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return None
    dirs = sorted(str(p) for loc in spec.submodule_search_locations for p in Path(loc).glob("*/lib") if p.is_dir())
    return ":".join(dirs) or None


class Engine(Protocol):
    """One Six engine: a new game, one search per turn, its provider."""

    provider: str

    def new_game(self) -> None: ...

    def search(self, moves: Sequence[tuple[int, int]], *, radius: int, nodes: int) -> list[str]: ...

    def close(self) -> None: ...


class SixEngine:
    """One `sixengine` process, the net passed absolute (it runs from its own directory), the cache off; Raises: SixEngineError when it does not start, loads no network or refuses a setting."""

    def __init__(self, assets: SixAssets, *, device: str) -> None:
        cmd = [str(assets.engine), "--net", str(assets.net.resolve())] + (["--cpu"] if device == "cpu" else [])
        env = dict(os.environ)
        # The release's own ONNX Runtime first, so no library on the inherited path shadows it.
        libs = [str(assets.engine.parent)] + ([cuda_library_path() or ""] if device == "cuda" else [])
        env["LD_LIBRARY_PATH"] = ":".join(x for x in [*libs, env.get("LD_LIBRARY_PATH", "")] if x)
        self._stderr = tempfile.NamedTemporaryFile(prefix="sixengine-", suffix=".stderr")
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                          text=True, encoding="utf-8", bufsize=1, cwd=str(assets.engine.parent),
                                          env=env)
        except OSError as exc:
            self._stderr.close()
            raise SixEngineError(f"sixengine did not start: {exc}") from exc
        try:
            self.ident = self._ask("six", "sixok")
            if NET_IDENT not in self.ident:
                raise SixEngineError(f"sixengine loaded no network: it answers {self.ident[:2]}")
            errors = [ln for ln in self._ask(f"setoption cacheEntries {CACHE_ENTRIES}\nisready", "readyok")
                      if ln.startswith("error")]
            if errors:
                raise SixEngineError(f"sixengine refused a setting: {errors}")
        except SixEngineError:
            self.close()
            raise
        self.stderr_lines = self._read_stderr()
        # The engine loads its network before it reads a command, so every provider line precedes `sixok`.
        self.provider = provider_of(self.stderr_lines, device)

    def _read_stderr(self) -> list[str]:
        return Path(self._stderr.name).read_text(encoding="utf-8", errors="replace").splitlines()

    def _ask(self, command: str, until: str) -> list[str]:
        """Send `command`, read through the first line starting with `until`; Raises: SixEngineError when the engine closed its output."""
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(command + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SixEngineError(f"sixengine is gone (rc {self._proc.poll()}): {exc}; stderr tail "
                                 f"{self._read_stderr()[-3:]}") from exc
        lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise SixEngineError(f"sixengine closed its output (rc {self._proc.poll()}); stderr tail "
                                     f"{self._read_stderr()[-3:]}")
            lines.append(line.strip())
            if lines[-1].startswith(until):
                return lines

    def new_game(self) -> None:
        """Clear the engine's tree and solver for a fresh game; Raises: SixEngineError when the engine is gone."""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write("newgame\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SixEngineError(f"sixengine is gone (rc {self._proc.poll()}): {exc}") from exc

    def search(self, moves: Sequence[tuple[int, int]], *, radius: int, nodes: int) -> list[str]:
        """The game's stones in order, then `go nodes`: every line through `bestmove`; Raises: SixEngineError when the engine closed its output."""
        flat = "".join(f" {q} {r}" for q, r in moves)
        position = f"position radius {radius}" + (f" moves{flat}" if moves else "")
        return self._ask(f"{position}\ngo nodes {nodes}", "bestmove")

    def close(self) -> None:
        """Ask the engine to quit, killing it after 20 s; idempotent."""
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=20)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._proc.kill()
                self._proc.wait()
        for pipe in (self._proc.stdin, self._proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except (BrokenPipeError, OSError):
                pass  # a dead engine's stdin cannot flush the buffered `quit`
        self._stderr.close()


class SixBot:
    """`BotProtocol` over one engine: Six answers a whole turn and reads the stone ORDER, which the arena reports to `observe_move`."""

    def __init__(self, engine: Engine, *, name: str, nodes: int) -> None:
        self._engine = engine
        self._name = name
        self.nodes = nodes
        self._moves: list[tuple[int, int]] = []
        self._pending: tuple[tuple[int, int], int] | None = None
        self.searches = 0
        self.seconds = 0.0
        self.stale_pending = 0
        self.forfeits = {"failed": 0, "illegal": 0}
        self.findings: list[str] = []
        self._closed = False

    @property
    def provider(self) -> str:
        return self._engine.provider

    def name(self) -> str:
        return self._name

    def new_game(self) -> None:
        """Forget the game's stones and the pending stone; Raises: SixEngineError when the engine is gone."""
        self._moves = []
        self._pending = None
        self._engine.new_game()

    def observe_move(self, q: int, r: int) -> None:
        """One stone the arena applied, in order."""
        self._moves.append((int(q), int(r)))

    def select_move(self, board: Any) -> tuple[int, int]:
        """One stone of Six's turn, a failed or illegal one returned for the arena to forfeit; Raises: SixSyncError, SixEngineError."""
        stones = {(int(q), int(r)) for q, r, _p in board.get_stones()}
        if len(self._moves) != len(stones) or set(self._moves) != stones:
            raise SixSyncError(f"observed {len(self._moves)} stones, the board holds {len(stones)}")
        left = int(board.moves_remaining)
        if self._pending is not None:
            # The turn's second stone is played from the same search only on exactly the board it was chosen for.
            move, prefix = self._pending
            self._pending = None
            if left == 1 and len(self._moves) == prefix:
                return self._checked(move, board)
            self.stale_pending += 1
        t0 = time.perf_counter()
        lines = self._engine.search(self._moves, radius=int(board.legal_move_radius()), nodes=self.nodes)
        self.seconds += time.perf_counter() - t0
        self.searches += 1
        reply = parse_reply(lines)
        if reply.failure is not None or not reply.stones:
            return self._forfeit("failed", f"ply {len(self._moves)}: {reply.failure}")
        first, *rest = reply.stones
        if left == 2 and rest:
            self._pending = (rest[0], len(self._moves) + 1)
        return self._checked(first, board)

    def _checked(self, move: tuple[int, int], board: Any) -> tuple[int, int]:
        if not board.is_legal(*move):
            self._finding("illegal", f"ply {len(self._moves)}: {move} is not in our legal set; returned for the arena to forfeit")
        return move

    def _forfeit(self, kind: str, text: str) -> tuple[int, int]:
        self._finding(kind, text)
        return self._moves[-1] if self._moves else _OFF_BOARD

    def _finding(self, kind: str, text: str) -> None:
        self.forfeits[kind] += 1
        self.findings.append(f"{kind}: {text}")
        _LOG.warning("%s %s %s", FINDING_LOG_MARKER, kind, text)

    def close(self) -> None:
        """Log the counters and close the engine; idempotent."""
        if not self._closed:
            self._closed = True
            _LOG.info("%s searches=%d seconds=%.1f stale_pending=%d forfeits_failed=%d forfeits_illegal=%d",
                      CLOSE_LOG_MARKER, self.searches, self.seconds, self.stale_pending, self.forfeits["failed"],
                      self.forfeits["illegal"])
        self._engine.close()


def _device_kind(device: str | None) -> str:
    kind, _, index = (None, "", "") if device is None else str(device).partition(":")
    # The engine takes no device index: ONNX Runtime plays on device 0.
    if kind not in _DEVICES or index not in ("", "0"):
        raise RungUnresolvable(rung="six", reason=f"a six rung plays on one of {list(_DEVICES)} (device 0); got device {device!r}")
    return kind


def resolve_six(*, opponent_sims: int | None, variant: str, device: str | None,
                vendor_root: Path | None = None) -> Callable[[], SixBot]:
    """A factory over a fresh engine per call, re-hashed at every start (`vendor_root` None = the repo's own); Raises: RungUnresolvable, also from the factory on a failed start, a netless engine or a provider that is not the device's."""
    if opponent_sims is None:
        raise RungUnresolvable(rung="six", reason="six rung declares no nodes")
    kind = _device_kind(device)
    root = find_vendor_root() if vendor_root is None else vendor_root
    locate_six(root, variant)
    nodes = int(opponent_sims)
    name = f"six_{variant}_n{nodes}"

    def _factory() -> SixBot:
        assets = locate_six(root, variant)
        try:
            engine = SixEngine(assets, device=kind)
        except SixEngineError as exc:
            raise RungUnresolvable(rung="six", reason=f"six engine failed to start: {exc}") from exc
        if engine.provider != kind:
            engine.close()
            raise RungUnresolvable(rung="six", reason=(
                f"{PROVIDER_MARKER}: the rung plays on {kind}, the engine reports {engine.provider} "
                f"({[ln for ln in engine.stderr_lines if 'available' in ln or 'using' in ln][:2]}); "
                "a CUDA engine needs the venv's NVIDIA libraries: `make build.cuda`"))
        _LOG.info("%s provider=%s variant=%s nodes=%d engine_sha256=%s net_sha256=%s runtime_sha256=%s",
                  PROVIDER_LOG_MARKER, engine.provider, variant, nodes, assets.engine_sha256, assets.net_sha256,
                  assets.runtime_sha256)
        return SixBot(engine, name=name, nodes=nodes)

    return _factory


__all__ = [
    "CACHE_ENTRIES", "CLOSE_LOG_MARKER", "ENGINE_ASSET", "FINDING_LOG_MARKER", "NET_IDENT", "RUNTIME_PREFIX", "PIN_ABSENT_MARKER", "PIN_NAME", "PROVIDER_LOG_MARKER",
    "PROVIDER_MARKER", "SHA_MISMATCH_MARKER", "VENDOR_ABSENT_MARKER", "Engine", "SixAssets", "SixBot", "SixEngine",
    "SixEngineError", "SixReply", "SixSyncError", "cuda_library_path", "locate_six", "parse_reply", "pin_record",
    "provider_of", "resolve_six", "six_availability",
]
