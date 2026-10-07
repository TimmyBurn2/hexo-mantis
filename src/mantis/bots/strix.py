# >300 justify (R8): the vendored tree's locator, the variant grammar, the driver transport and the legality fence are one
# adapter whose parts are only checkable against each other.
"""The strix rung adapter: the pinned `SootyOwl/hexo-strix` checkpoint as a fixed external reference, played by
`tools/strix_driver.py` in the vendored tree's own venv (JSON lines; noise-off argmax acting, deterministic on the CPU; a
`@cuda` cell's GPU scatter may reorder, so its replay is measured, not promised). The fence counts every illegal reply."""
from __future__ import annotations

import json
import logging
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, Protocol

import mantis
from mantis.bots.protocol import RungUnresolvable
from mantis.util.hashing import sha256_file

PIN_NAME = "hexo-strix"
#: Path segments below the vendor root, kept as segments so no path-shaped literal lives here.
_VENDOR_TREE = ("external", "hexo-strix")
_MODELS_DIR = ("external", "strix_models")
#: Each device's own venv in the vendored tree: CPU torch, or strix's `cuda` group (`vendor_build_strix.sh --cuda`).
_DEVICE_VENVS = {"cpu": ".venv", "cuda": ".venv-cuda"}
_DRIVER = ("tools", "strix_driver.py")
BUILD_SCRIPT = "tools/vendor_build_strix.sh"

VENDOR_ABSENT_MARKER = "strix vendor tree not located"
VENV_ABSENT_MARKER = "strix venv not built"
CHECKPOINT_ABSENT_MARKER = "strix checkpoint not placed"
SHA_MISMATCH_MARKER = "strix checkpoint sha256 mismatch"
PIN_ABSENT_MARKER = "strix pin absent"

#: strix's `m_actions` at its own deploy (`scripts/play_vs_shrimp.py`); cell B keeps it too.
DEFAULT_M_ACTIONS = 16
#: `<stem>:net_only` plays the pinned checkpoint with its root VCF solver OFF — a distinct
#: instrument (the variant rides the regime key), never the rung on record, which loads solver ON.
NET_ONLY_SUFFIX = ":net_only"
#: `<stem>:r<N>` plays the pinned checkpoint, solver ON, at the driver's `placement_radius` N
#: (strix trained at 6; every reading on record rides the default 8); the load line carries the key only there.
RADIUS_SUFFIX = ":r"
#: `<variant>@cuda` plays that variant with strix's net on the GPU (its own CUDA venv); every other line rides the CPU.
DEVICE_SUFFIX = "@cuda"

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

    def __init__(self, *, transport: Transport, name: str) -> None:
        self._transport = transport
        self._name = name
        self.moves = 0
        self.seconds = 0.0
        self.fence_disagreements = 0
        self.out_of_fence = 0
        self.findings: list[str] = []
        #: The LAST reply's own `sims` (the driver's root-visit sum); None when no driver was consulted.
        self.last_sims: int | None = None

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
            self.last_sims = None
            return (0, 0) if board.is_legal(0, 0) else tuple(board.legal_moves()[0])
        request = {"op": "select", "stones": stones, "to_move": int(board.current_player),
                   "moves_remaining": int(board.moves_remaining)}
        t0 = time.perf_counter()
        reply = self._transport.ask(request)
        self.seconds += time.perf_counter() - t0
        if "error" in reply:
            raise RuntimeError(f"strix driver: {reply['error']}")
        self.moves += 1
        self.last_sims = None if reply.get("sims") is None else int(reply["sims"])
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


def find_vendor_root() -> Path | None:
    """The `vendor/` directory of the repo the package is installed from, or None. Returns None,
    never a default path and never an env-provided one: an endpoint that can point anywhere is a
    host-path channel wearing a disguise."""
    package_file = mantis.__file__
    if package_file is None:  # namespace package: nothing to walk up from
        return None
    for ancestor in Path(package_file).resolve().parents:
        if (ancestor / "vendor" / "pins.toml").is_file():
            return ancestor / "vendor"
    return None


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
    digest = sha256_file(path)
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


def venv_python(tree: Path, device: str) -> Path:
    """The python of `device`'s venv inside the vendored strix tree. Raises: ValueError on a device with no venv."""
    if device not in _DEVICE_VENVS:
        raise ValueError(f"strix has no {device!r} venv (known: {sorted(_DEVICE_VENVS)})")
    return tree / _DEVICE_VENVS[device] / "bin" / "python"


def variant_device(variant: str) -> str:
    """`cuda` for a `<variant>@cuda`, else `cpu`."""
    return "cuda" if variant.endswith(DEVICE_SUFFIX) else "cpu"


def _base_variant(variant: str) -> str:
    return variant[: -len(DEVICE_SUFFIX)] if variant.endswith(DEVICE_SUFFIX) else variant


def locate_strix(device: str = "cpu") -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """`(python, driver, cwd, checkpoint, pin)` with `device`'s venv, each refusal naming exactly its missing step.

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
    python = venv_python(tree, device)
    if not python.is_file():
        flag = "" if device == "cpu" else f" --{device}"
        raise RungUnresolvable(rung="strix", reason=(
            f"{VENV_ABSENT_MARKER}: no {'/'.join((_DEVICE_VENVS[device], 'bin', 'python'))} under the vendored tree; run "
            f"`bash {BUILD_SCRIPT}{flag}` from the repo root (it verifies the pinned sha, then builds "
            f"strix's own {device} venv and its Rust engine)"))
    checkpoint = root.joinpath(*_MODELS_DIR) / str(pin["checkpoint"])
    if not checkpoint.is_file():
        raise RungUnresolvable(rung="strix", reason=(
            f"{CHECKPOINT_ABSENT_MARKER}: place the operator's {pin['checkpoint']} under "
            f"vendor/{'/'.join(_MODELS_DIR)}/ (sha256 {pin['checkpoint_sha256']})"))
    verify_checkpoint_sha(checkpoint, str(pin["checkpoint_sha256"]))
    driver = root.parent.joinpath(*_DRIVER)
    return python, driver, tree, checkpoint, pin


def variant_radius(variant: str, *, stem: str) -> int | None:
    """The `placement_radius` a `<stem>:r<N>` variant loads strix at, None otherwise; Raises: RungUnresolvable on a malformed N."""
    variant = _base_variant(variant)
    if not variant.startswith(stem + RADIUS_SUFFIX):
        return None
    digits = variant[len(stem) + len(RADIUS_SUFFIX):]
    if not digits.isdigit() or int(digits) < 1:
        raise RungUnresolvable(rung=f"strix:{variant}", reason=(
            f"{stem}{RADIUS_SUFFIX}<N> names a positive placement radius; got {digits!r}"))
    return int(digits)


def variant_solver(variant: str, *, stem: str) -> bool:
    """Whether `variant` plays with the solver ON (the pinned stem, its name, or `<stem>:r<N>`) or OFF (`<stem>:net_only`).

    Raises:
        RungUnresolvable: a variant the pin does not name."""
    variant = _base_variant(variant)
    if variant in (stem, PIN_NAME) or variant_radius(variant, stem=stem) is not None:
        return True
    if variant == stem + NET_ONLY_SUFFIX:
        return False
    raise RungUnresolvable(rung=f"strix:{variant}", reason=(
        f"the pin names {stem}; a strix rung plays the pinned checkpoint (solver ON), "
        f"{stem}{NET_ONLY_SUFFIX} (solver OFF, R358(a)), {stem}{RADIUS_SUFFIX}<N> (the driver's "
        "placement_radius N, R365 E1) or nothing"))


def load_request(checkpoint: str, *, sims: int, variant: str, stem: str) -> dict[str, Any]:
    """The driver's `load` line; `placement_radius` only on `:r<N>`, `device` only on `@cuda`. Raises: RungUnresolvable."""
    request = {"op": "load", "checkpoint": checkpoint, "sims": int(sims), "m_actions": DEFAULT_M_ACTIONS,
               "disable_forcing_solver": not variant_solver(variant, stem=stem)}
    radius = variant_radius(variant, stem=stem)
    if radius is not None:
        request["placement_radius"] = radius
    if variant_device(variant) != "cpu":
        request["device"] = variant_device(variant)
    return request


def resolve_strix(*, opponent_sims: int | None, variant: str) -> Any:
    """Probe the vendored tree EAGERLY and hand back a factory over a fresh driver process.

    Raises:
        RungUnresolvable: no sims, a variant the pin does not name, or a missing step."""
    if opponent_sims is None:
        raise RungUnresolvable(rung="strix", reason="strix rung declares no sims")
    device = variant_device(variant)
    python, driver, cwd, checkpoint, pin = locate_strix(device)
    stem = str(pin["checkpoint"]).rsplit(".", 1)[0]
    solver_on = variant_solver(variant, stem=stem)
    radius = variant_radius(variant, stem=stem)
    sims = int(opponent_sims)
    request = load_request(str(checkpoint), sims=sims, variant=variant, stem=stem)
    name = (f"strix_{stem}_s{sims}" + ("" if solver_on else "_nosolver") + ("" if radius is None else f"_r{radius}")
            + ("" if device == "cpu" else f"_{device}"))

    def _factory() -> StrixBot:
        transport = DriverTransport(python, driver, cwd)
        reply = transport.ask(dict(request))
        if "error" in reply:
            transport.close()
            raise RungUnresolvable(rung="strix", reason=f"strix driver failed to load: {reply['error']}")
        return StrixBot(transport=transport, name=name)

    return _factory


__all__ = [
    "BUILD_SCRIPT", "CHECKPOINT_ABSENT_MARKER", "DEFAULT_M_ACTIONS", "DEVICE_SUFFIX", "DriverTransport", "FINDING_LOG_MARKER",
    "NET_ONLY_SUFFIX", "PIN_ABSENT_MARKER", "PIN_NAME", "RADIUS_SUFFIX", "SHA_MISMATCH_MARKER", "StrixBot",
    "Transport", "VENDOR_ABSENT_MARKER", "VENV_ABSENT_MARKER", "find_vendor_root", "load_request", "locate_strix", "resolve_strix",
    "strix_availability", "variant_device", "variant_radius", "variant_solver", "venv_python", "verify_checkpoint_sha",
]
