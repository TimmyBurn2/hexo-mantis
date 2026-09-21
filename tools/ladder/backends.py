"""The two backends behind one seam (LADDER-1 §1.2): `mantis`, the deploy head on a named checkpoint with every knob the run config's; `strix`, the pinned rung through its own driver — one COMPOUND turn per call, the leaves spent read from the head's own counter."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

Cell = tuple[int, int]


class BackendError(RuntimeError):
    """A backend that cannot be opened as asked; the reason names the missing or mismatched thing."""


#: R363 §0(5): the ONE preset beside the unit — the ladder tool's own row, labelled on every receipt, never a unit reading.
PRESET_SIMS: dict[str, int | None] = {"unit": None, "play": 64}


@dataclass(frozen=True)
class TurnResult:
    """Two placements, the leaves spent on the SEARCHED ones (the backend's own count), the wall the pair took, and how many of the two the book forced (R363(c))."""

    placements: tuple[Cell, Cell]
    sims: int
    ms: float
    book_stones: int = 0


class Backend(Protocol):
    """One registered bot's engine. `name` is `<backend>:<net_hash[:8]>`; `search` is what the receipt records."""

    backend: str
    name: str
    net_hash: str
    sims: int
    encoding: str
    search: dict[str, Any]
    seed: int | None

    def new_game(self, game_id: str) -> None: ...

    def select_turn(self, board: Any, forced: tuple[Cell, ...] = ()) -> TurnResult: ...

    def close(self) -> None: ...


def seed_from_game_id(game_id: str) -> int:
    """The per-game seed: the first 8 bytes of sha256(game_id), so a replayed match id draws the same noise."""
    return int.from_bytes(hashlib.sha256(game_id.encode("utf-8")).digest()[:8], "big")


def _resolve_sims(preset: str, unit_sims: int) -> int:
    """The sims a preset plays at: the unit's own, or the preset's row. Raises: BackendError on a preset not in PRESET_SIMS."""
    if preset not in PRESET_SIMS:
        raise BackendError(f"preset {preset!r} is not one of {sorted(PRESET_SIMS)}")
    override = PRESET_SIMS[preset]
    return int(unit_sims) if override is None else int(override)


def _two_stones(player: Any, board: Any, sims_of: Any, forced: tuple[Cell, ...] = ()) -> TurnResult:
    """One compound turn through a half-ply `BotProtocol` player: the book's `forced` stones first (unsearched, up to two), each remaining stone chosen on the board after the one before. Raises: BackendError when a forced stone is not legal on the board (the book does not replay here — a bug, never a server rejection)."""
    if len(forced) > 2:
        raise BackendError(f"a compound turn takes two stones; {len(forced)} were forced")
    work = board.clone()
    t0 = time.perf_counter()
    placements: list[Cell] = []
    spent = 0
    for i in range(2):
        if i < len(forced):
            q, r = int(forced[i][0]), int(forced[i][1])
            if not work.is_legal(q, r):
                raise BackendError(f"forced book stone ({q}, {r}) is not legal on this board after {len(work.get_stones())} stones")
        else:
            move = player.select_move(work)
            q, r = int(move[0]), int(move[1])
            spent += int(sims_of())
        work.apply_move(q, r)
        placements.append((q, r))
    ms = (time.perf_counter() - t0) * 1000.0
    return TurnResult(placements=(placements[0], placements[1]), sims=spent, ms=round(ms, 3), book_stones=len(forced))


class MantisBackend:
    """The deploy head on `checkpoint`: the config's `deploy.search.kind` at `eval.gate.deploy_sims`, its σ and batching — the follower's cell, one process, CPU."""

    backend = "mantis"

    def __init__(self, config: Any, checkpoint: Path, *, threads: int | None, preset: str = "unit") -> None:
        import torch

        from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
        from mantis.config.resolve.inference_batching import resolve_inference_batching
        from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
        from mantis.encoding import lookup, normalize_encoding_name
        from mantis.model import build_net
        from mantis.model.identity import net_param_hash
        from mantis.selfplay.inference_local import LocalInferenceEngine
        from mantis.train.checkpoints import deploy_state, load_checkpoint

        if threads is not None:
            torch.set_num_threads(int(threads))
        self.encoding = normalize_encoding_name(config.identity.encoding)
        ck = load_checkpoint(checkpoint)
        if normalize_encoding_name(ck.metadata.encoding_name) != self.encoding:
            raise BackendError(f"{Path(checkpoint).name} is stamped for encoding {ck.metadata.encoding_name!r}; the "
                               f"config declares {config.identity.encoding!r} — the head plays the config's geometry")
        if ck.metadata.arch is None:
            raise BackendError(f"{Path(checkpoint).name}: the stamp resolves no arch, so the net cannot be rebuilt")
        net = build_net(ck.metadata.arch)
        state, self.weights = deploy_state(ck)  # the EMA shadow when the stamp carries one (R366(b))
        net.load_state_dict(state)
        net.to("cpu").eval()
        self.net_hash = net_param_hash(net)
        self.name = f"{self.backend}:{self.net_hash[:8]}"
        self.checkpoint = Path(checkpoint)
        self.step = int(ck.metadata.step)
        dump = config.model_dump()
        graph = config.identity.representation == "graph"
        self._spec = lookup(self.encoding)
        self._engine = LocalInferenceEngine(
            net, torch.device("cpu"), encoding_spec=self._spec,
            fused_graph_caps=resolve_fused_graph_caps(dump) if graph else None,
            inference_batching=resolve_inference_batching(dump) if graph else None,
            max_in_flight=int(config.selfplay.leaf_batch_size),
            leaf_build_threads=resolve_leaf_build_threads(dump) if graph else 1,
        )
        self.sims = _resolve_sims(preset, int(config.eval.gate.deploy_sims))
        self._leaf_batch_size = int(config.selfplay.leaf_batch_size)
        self._c_visit, self._c_scale = float(config.selfplay.c_visit), float(config.selfplay.c_scale)
        self._q_rescale, self._gumbel_m = bool(config.selfplay.q_rescale), int(config.selfplay.gumbel_m)
        self._search_kind = str(config.deploy.search.kind)
        self.search: dict[str, Any] = {
            "kind": self._search_kind, "sims": self.sims, "preset": preset, "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "encoding": self.encoding, "checkpoint": self.checkpoint.name, "step": self.step,
            "weights": self.weights, "leaf_batch_size": self._leaf_batch_size, "c_visit": self._c_visit, "c_scale": self._c_scale,
            "q_rescale": self._q_rescale, "gumbel_m": self._gumbel_m}
        self.seed: int | None = None
        self._head: Any = None

    def new_game(self, game_id: str) -> None:
        """A fresh head seeded from the game id (the Gumbel draw; PUCT draws nothing and is deterministic already)."""
        from mantis.eval.worker import build_candidate_player

        self.seed = seed_from_game_id(game_id)
        self._head = build_candidate_player(
            self._engine, self.sims, spec=self._spec, leaf_batch_size=self._leaf_batch_size, c_visit=self._c_visit,
            c_scale=self._c_scale, q_rescale=self._q_rescale, search_kind=self._search_kind, gumbel_m=self._gumbel_m,
            gumbel_seed=self.seed)
        self._head.new_game()

    def select_turn(self, board: Any, forced: tuple[Cell, ...] = ()) -> TurnResult:
        """Two stones from the head (the book's `forced` ones first), `sims` the sum of its `last_sims` over the searched ones. Raises: BackendError before `new_game`, or on a forced stone the board refuses."""
        if self._head is None:
            raise BackendError("select_turn before new_game: the head is seeded per game")
        head = self._head
        return _two_stones(head, board, lambda: int(head.last_sims or 0), forced)

    def close(self) -> None:
        self._head = None


class StrixBackend:
    """The pinned strix checkpoint through `tools/strix_driver.py` in the vendored venv, solver ON, noise off (the unit on record); `sims` is the driver's own root-visit sum per stone."""

    backend = "strix"
    encoding = "gnn_axis_r8"  # the radius-8 fence: the server's placement radius, and strix's own

    def __init__(self, *, sims: int, threads: int | None, preset: str = "unit") -> None:
        from mantis.bots.strix import (
            DEFAULT_M_ACTIONS,
            DriverTransport,
            StrixBot,
            load_request,
            locate_strix,
        )

        sims = _resolve_sims(preset, int(sims))
        python, driver, cwd, checkpoint, pin = locate_strix()
        stem = str(pin["checkpoint"]).rsplit(".", 1)[0]
        request = load_request(str(checkpoint), sims=int(sims), variant=stem, stem=stem)
        if threads is not None:
            request["threads"] = int(threads)
        self._transport = DriverTransport(python, driver, cwd)
        reply = self._transport.ask(request)
        if "error" in reply:
            self._transport.close()
            raise BackendError(f"strix driver failed to load: {reply['error']}")
        self.sims = int(sims)
        self.net_hash = str(pin["checkpoint_sha256"])
        self.name = f"{self.backend}:{self.net_hash[:8]}"
        self._bot = StrixBot(transport=self._transport, sims=self.sims, name=self.name)
        self.search = {"kind": "gumbel", "sims": self.sims, "preset": preset, "m_actions": DEFAULT_M_ACTIONS,
                       "solver": str(reply.get("forcing_solver")), "acting": str(reply.get("acting")),
                       "torch": str(reply.get("torch")), "device": str(reply.get("device")),
                       "train_steps": reply.get("train_steps"), "commit": str(pin["sha"]),
                       "checkpoint": str(pin["checkpoint"])}
        self.seed: int | None = None

    def new_game(self, game_id: str) -> None:
        """The driver searches with a fixed seed (`seed=0`), so the game id enters nothing; recorded all the same."""
        self.seed = seed_from_game_id(game_id)
        self._bot.new_game()

    def select_turn(self, board: Any, forced: tuple[Cell, ...] = ()) -> TurnResult:
        """Two stones from the driver (the book's `forced` ones first); `sims` is the sum of the driver's `sims` over the searched ones. Raises: BackendError on a forced stone the board refuses."""
        bot = self._bot
        return _two_stones(bot, board, lambda: int(bot.last_sims or 0), forced)

    @property
    def findings(self) -> list[str]:
        """The fence findings so far (R257): every disagreement between the driver's legal set and ours."""
        return list(self._bot.findings)

    def close(self) -> None:
        self._bot.close()


def open_mantis(config: Any, checkpoint: Path, *, threads: int | None, preset: str = "unit") -> MantisBackend:
    """The mantis backend on `checkpoint` under `config`'s eval seam, at the unit's sims or a PRESET_SIMS row. Raises: BackendError when the stamp disagrees with the config's encoding, resolves no arch, or the preset is unknown."""
    return MantisBackend(config, Path(checkpoint), threads=threads, preset=preset)


def open_strix(*, sims: int, threads: int | None, preset: str = "unit") -> StrixBackend:
    """The strix backend at the pin, at `sims` or a PRESET_SIMS row. Raises: RungUnresolvable when the vendored tree, its venv or the pinned checkpoint is absent; BackendError when the driver refuses the load or the preset is unknown."""
    return StrixBackend(sims=sims, threads=threads, preset=preset)


__all__ = ["Backend", "BackendError", "Cell", "MantisBackend", "PRESET_SIMS", "StrixBackend", "TurnResult",
           "open_mantis", "open_strix", "seed_from_game_id"]
