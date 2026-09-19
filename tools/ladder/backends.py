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


@dataclass(frozen=True)
class TurnResult:
    """Two placements, the leaves spent on both (the backend's own count) and the wall the pair took."""

    placements: tuple[Cell, Cell]
    sims: int
    ms: float


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

    def select_turn(self, board: Any) -> TurnResult: ...

    def close(self) -> None: ...


def seed_from_game_id(game_id: str) -> int:
    """The per-game seed: the first 8 bytes of sha256(game_id), so a replayed match id draws the same noise."""
    return int.from_bytes(hashlib.sha256(game_id.encode("utf-8")).digest()[:8], "big")


def _two_stones(player: Any, board: Any, sims_of: Any) -> TurnResult:
    """One compound turn through a half-ply `BotProtocol` player: the second stone is chosen on the board after the first."""
    work = board.clone()
    t0 = time.perf_counter()
    first = player.select_move(work)
    spent = sims_of()
    work.apply_move(*first)
    second = player.select_move(work)
    spent += sims_of()
    ms = (time.perf_counter() - t0) * 1000.0
    return TurnResult(placements=((int(first[0]), int(first[1])), (int(second[0]), int(second[1]))),
                      sims=spent, ms=round(ms, 3))


class MantisBackend:
    """The deploy head on `checkpoint`: the config's `deploy.search.kind` at `eval.gate.deploy_sims`, its σ and batching — the follower's cell, one process, CPU."""

    backend = "mantis"

    def __init__(self, config: Any, checkpoint: Path, *, threads: int | None) -> None:
        import torch

        from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
        from mantis.config.resolve.inference_batching import resolve_inference_batching
        from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
        from mantis.encoding import lookup, normalize_encoding_name
        from mantis.model import build_net
        from mantis.model.identity import net_param_hash
        from mantis.selfplay.inference_local import LocalInferenceEngine
        from mantis.train.checkpoints import load_checkpoint

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
        net.load_state_dict(ck.model_state)
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
        self.sims = int(config.eval.gate.deploy_sims)
        self._leaf_batch_size = int(config.selfplay.leaf_batch_size)
        self._c_visit, self._c_scale = float(config.selfplay.c_visit), float(config.selfplay.c_scale)
        self._q_rescale, self._gumbel_m = bool(config.selfplay.q_rescale), int(config.selfplay.gumbel_m)
        self._search_kind = str(config.deploy.search.kind)
        self.search: dict[str, Any] = {
            "kind": self._search_kind, "sims": self.sims, "device": "cpu", "torch_threads": torch.get_num_threads(),
            "encoding": self.encoding, "checkpoint": self.checkpoint.name, "step": self.step,
            "leaf_batch_size": self._leaf_batch_size, "c_visit": self._c_visit, "c_scale": self._c_scale,
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

    def select_turn(self, board: Any) -> TurnResult:
        """Two stones from the head, `sims` the sum of its `last_sims` over both. Raises: BackendError before `new_game`."""
        if self._head is None:
            raise BackendError("select_turn before new_game: the head is seeded per game")
        head = self._head
        return _two_stones(head, board, lambda: int(head.last_sims or 0))

    def close(self) -> None:
        self._head = None


class StrixBackend:
    """The pinned strix checkpoint through `tools/strix_driver.py` in the vendored venv, solver ON, noise off (the unit on record); `sims` is the driver's own root-visit sum per stone."""

    backend = "strix"
    encoding = "gnn_axis_r8"  # the radius-8 fence: the server's placement radius, and strix's own

    def __init__(self, *, sims: int, threads: int | None) -> None:
        from mantis.bots.strix import (
            DEFAULT_M_ACTIONS,
            DriverTransport,
            StrixBot,
            load_request,
            locate_strix,
        )

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
        self.search = {"kind": "gumbel", "sims": self.sims, "m_actions": DEFAULT_M_ACTIONS,
                       "solver": str(reply.get("forcing_solver")), "acting": str(reply.get("acting")),
                       "torch": str(reply.get("torch")), "device": str(reply.get("device")),
                       "train_steps": reply.get("train_steps"), "commit": str(pin["sha"]),
                       "checkpoint": str(pin["checkpoint"])}
        self.seed: int | None = None

    def new_game(self, game_id: str) -> None:
        """The driver searches with a fixed seed (`seed=0`), so the game id enters nothing; recorded all the same."""
        self.seed = seed_from_game_id(game_id)
        self._bot.new_game()

    def select_turn(self, board: Any) -> TurnResult:
        """Two stones from the driver; `sims` is the sum of the driver's `sims` over both."""
        bot = self._bot
        return _two_stones(bot, board, lambda: int(bot.last_sims or 0))

    @property
    def findings(self) -> list[str]:
        """The fence findings so far (R257): every disagreement between the driver's legal set and ours."""
        return list(self._bot.findings)

    def close(self) -> None:
        self._bot.close()


def open_mantis(config: Any, checkpoint: Path, *, threads: int | None) -> MantisBackend:
    """The mantis backend on `checkpoint` under `config`'s eval seam. Raises: BackendError when the stamp disagrees with the config's encoding or resolves no arch."""
    return MantisBackend(config, Path(checkpoint), threads=threads)


def open_strix(*, sims: int, threads: int | None) -> StrixBackend:
    """The strix backend at the pin. Raises: RungUnresolvable when the vendored tree, its venv or the pinned checkpoint is absent; BackendError when the driver refuses the load."""
    return StrixBackend(sims=sims, threads=threads)


__all__ = ["Backend", "BackendError", "Cell", "MantisBackend", "StrixBackend", "TurnResult", "open_mantis",
           "open_strix", "seed_from_game_id"]
