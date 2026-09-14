"""The strix rung adapter (RUNG-2, R352(e)): the pinned `SootyOwl/hexo-strix` checkpoint as a
fixed external reference, played by `tools/strix_driver.py` in the vendored tree's own venv
(JSON lines; deterministic — strix's noise-off argmax-of-improved-policy acting). The fence is
read at contact: every reply's legal set is compared with the board's and counted (R257)."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, Protocol

from mantis.bots.protocol import RungUnresolvable
from mantis.bots.sealbot import find_vendor_root

PIN_NAME = "hexo-strix"
#: Path segments below the vendor root, kept as segments so no path-shaped literal lives here.
_VENDOR_TREE = ("external", "hexo-strix")
_MODELS_DIR = ("external", "strix_models")
_VENV_PYTHON = (".venv", "bin", "python")
_DRIVER = ("tools", "strix_driver.py")
BUILD_SCRIPT = "tools/vendor_build_strix.sh"

VENDOR_ABSENT_MARKER = "strix vendor tree not located"
VENV_ABSENT_MARKER = "strix venv not built"
CHECKPOINT_ABSENT_MARKER = "strix checkpoint not placed"
SHA_MISMATCH_MARKER = "strix checkpoint sha256 mismatch"
PIN_ABSENT_MARKER = "strix pin absent"

#: strix's `m_actions` at its own deploy (`scripts/play_vs_shrimp.py`); cell B keeps it too.
DEFAULT_M_ACTIONS = 16

#: Every fence finding is ALSO logged under this marker: the bot instance dies with the eval
#: child, and `tools/strength_frontier.py` counts the lines into the cell record.
FINDING_LOG_MARKER = "strix_fence_finding"
_LOG = logging.getLogger(__name__)


class Transport(Protocol):
    """One request in, one reply out; `close` ends the process."""

    def ask(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class DriverTransport:
    """The driver subprocess: JSON lines over stdin/stdout, stderr to the parent's."""

    def __init__(self, python: Path, driver: Path, cwd: Path) -> None:
        self._proc = subprocess.Popen(
            [str(python), str(driver)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", cwd=str(cwd), bufsize=1,
        )

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send one request, read one reply line; raises `RuntimeError` when the driver exited."""
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"strix driver closed its output (rc {self._proc.poll()})")
        return json.loads(line)

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class StrixBot:
    """`BotProtocol` over a strix transport; the position is rebuilt from the board on every call."""

    def __init__(self, *, transport: Transport, sims: int, name: str,
                 m_actions: int = DEFAULT_M_ACTIONS) -> None:
        self._transport = transport
        self._sims = int(sims)
        self._m_actions = int(m_actions)
        self._name = name
        self.moves = 0
        self.seconds = 0.0
        self.fence_disagreements = 0
        self.out_of_fence = 0
        self.findings: list[str] = []

    def name(self) -> str:
        return self._name

    def new_game(self) -> None:
        return None

    def select_move(self, board: Any) -> tuple[int, int]:
        """One stone: the board's position to the driver, its move back, the fence compared.

        Raises:
            RuntimeError: the driver reported an error for this position."""
        stones = [[int(q), int(r), int(p)] for q, r, p in board.get_stones()]
        if not stones:
            # The opening single: hexo_rs seats p1 at the origin by rule, and every opening
            # cell is the same position up to translation — not a decision to consult on.
            self.moves += 1
            return (0, 0) if board.is_legal(0, 0) else tuple(board.legal_moves()[0])
        request = {"op": "select", "stones": stones, "to_move": int(board.current_player),
                   "moves_remaining": int(board.moves_remaining)}
        t0 = time.perf_counter()
        reply = self._transport.ask(request)
        self.seconds += time.perf_counter() - t0
        if "error" in reply:
            raise RuntimeError(f"strix driver: {reply['error']}")
        self.moves += 1
        ours = {(int(q), int(r)) for q, r in board.legal_moves()}
        theirs = {(int(q), int(r)) for q, r in reply.get("legal", [])}
        if theirs != ours:
            self.fence_disagreements += 1
            self._finding(f"fence disagreement at ply {len(stones)}: only ours "
                          f"{sorted(ours - theirs)[:4]}, only strix's {sorted(theirs - ours)[:4]}")
        q, r = int(reply["move"][0]), int(reply["move"][1])
        if (q, r) not in ours:
            self.out_of_fence += 1
            self._finding(f"move ({q}, {r}) at ply {len(stones)} is out of our fence — "
                          "returned for the arena to forfeit")
        return q, r

    def _finding(self, text: str) -> None:
        self.findings.append(text)
        _LOG.warning("%s %s", FINDING_LOG_MARKER, text)

    def close(self) -> None:
        self._transport.close()


def _pin() -> dict[str, Any] | None:
    root = find_vendor_root()
    if root is None:
        return None
    pins = tomllib.loads((root / "pins.toml").read_text(encoding="utf-8")).get("pins", {})
    return pins.get(PIN_NAME)


def verify_checkpoint_sha(path: Path, expected: str) -> str:
    """The checkpoint's sha256, refused by name when it is not the pin's.

    Raises:
        RungUnresolvable: the bytes on disk are not the pinned checkpoint."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RungUnresolvable(
            rung="strix",
            reason=(f"{SHA_MISMATCH_MARKER}: {path.name} hashes to {digest} but vendor/pins.toml "
                    f"pins {expected}; the rung plays the pinned checkpoint or nothing"))
    return digest


def strix_availability() -> tuple[bool, str]:
    """`(available, why-not)`: the pin, the vendored tree, its venv and the pinned checkpoint."""
    try:
        locate_strix()
    except RungUnresolvable as exc:
        return False, exc.reason
    return True, ""


def locate_strix() -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """`(python, driver, cwd, checkpoint, pin)`, each refusal naming exactly its missing step.

    Raises:
        RungUnresolvable: the vendor root, pin, tree, venv or checkpoint is absent, or the sha differs."""
    root = find_vendor_root()
    if root is None:
        raise RungUnresolvable(rung="strix", reason=(
            f"{VENDOR_ABSENT_MARKER}: no ancestor of the installed package holds vendor/pins.toml; "
            "run `make vendor` from the repo root"))
    pin = _pin()
    if pin is None:
        raise RungUnresolvable(rung="strix", reason=(
            f"{PIN_ABSENT_MARKER}: vendor/pins.toml declares no [pins.{PIN_NAME}]"))
    tree = root.joinpath(*_VENDOR_TREE)
    if not tree.is_dir():
        raise RungUnresolvable(rung="strix", reason=(
            f"{VENDOR_ABSENT_MARKER}: {'/'.join(_VENDOR_TREE)} is not fetched under vendor/; "
            "run `make vendor` from the repo root"))
    python = tree.joinpath(*_VENV_PYTHON)
    if not python.is_file():
        raise RungUnresolvable(rung="strix", reason=(
            f"{VENV_ABSENT_MARKER}: no {'/'.join(_VENV_PYTHON)} under the vendored tree; run "
            f"`bash {BUILD_SCRIPT}` from the repo root (it verifies the pinned sha, then builds "
            "strix's own venv with CPU torch and its Rust engine)"))
    checkpoint = root.joinpath(*_MODELS_DIR) / str(pin["checkpoint"])
    if not checkpoint.is_file():
        raise RungUnresolvable(rung="strix", reason=(
            f"{CHECKPOINT_ABSENT_MARKER}: place the operator's {pin['checkpoint']} under "
            f"vendor/{'/'.join(_MODELS_DIR)}/ (sha256 {pin['checkpoint_sha256']})"))
    verify_checkpoint_sha(checkpoint, str(pin["checkpoint_sha256"]))
    driver = root.parent.joinpath(*_DRIVER)
    return python, driver, tree, checkpoint, pin


def resolve_strix(*, opponent_sims: int | None, variant: str) -> Any:
    """Probe the vendored tree EAGERLY and hand back a factory over a fresh driver process.

    Raises:
        RungUnresolvable: no sims, a variant the pin does not name, or a missing step."""
    if opponent_sims is None:
        raise RungUnresolvable(rung="strix", reason="strix rung declares no sims")
    python, driver, cwd, checkpoint, pin = locate_strix()
    stem = str(pin["checkpoint"]).rsplit(".", 1)[0]
    if variant not in (stem, PIN_NAME):
        raise RungUnresolvable(rung=f"strix:{variant}", reason=(
            f"the pin names {stem}; a strix rung plays the pinned checkpoint or nothing"))
    sims = int(opponent_sims)

    def _factory() -> StrixBot:
        transport = DriverTransport(python, driver, cwd)
        reply = transport.ask({"op": "load", "checkpoint": str(checkpoint), "sims": sims,
                               "m_actions": DEFAULT_M_ACTIONS})
        if "error" in reply:
            transport.close()
            raise RungUnresolvable(rung="strix", reason=f"strix driver failed to load: {reply['error']}")
        return StrixBot(transport=transport, sims=sims, name=f"strix_{stem}_s{sims}")

    return _factory


__all__ = [
    "BUILD_SCRIPT", "CHECKPOINT_ABSENT_MARKER", "DEFAULT_M_ACTIONS", "DriverTransport", "FINDING_LOG_MARKER",
    "PIN_ABSENT_MARKER", "PIN_NAME", "SHA_MISMATCH_MARKER", "StrixBot", "Transport",
    "VENDOR_ABSENT_MARKER", "VENV_ABSENT_MARKER", "locate_strix", "resolve_strix",
    "strix_availability", "verify_checkpoint_sha",
]
