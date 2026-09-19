"""The strix engine: the vendored pin through THIS tree's driver, its `analyze` op mapped onto RawRead / Search."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mantis._engine import Board
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.strix import DEFAULT_M_ACTIONS, DriverTransport, load_request, locate_strix
from mantis.encoding import lookup

from .engines import STRIX, ChildInfo, EngineInfo, EngineLoadError, RawRead, Search

DRIVER = Path(__file__).resolve().parents[1] / "strix_driver.py"
RAW_SIMS = 1
GAPS = ("no searched root value of its own: the head row shows the visit-weighted mean of per_child_q, labelled derived",
        "no quiescence counter (q-fires reads 0); strix's forcing solver may short-circuit a search (sims < requested)",
        "raw_value is the mover's, probed 2026-09-19: p2 to move holding an open four reads +1.0")
STRIX_BUILD = "make vendor.strix"


class StrixRefused(ValueError):
    """The driver refused this position; the message is the driver's own."""


def strix_info() -> EngineInfo:
    """The strix `/engines` row; unavailable → `note` carries the reason and names the build target."""
    try:
        _python, _driver, _cwd, checkpoint, _pin = locate_strix()
    except RungUnresolvable as exc:
        return EngineInfo(id="strix", kind=STRIX, path="", note=f"unavailable: {exc.reason}; run `{STRIX_BUILD}`")
    return EngineInfo(id="strix", kind=STRIX, path=str(checkpoint))


class StrixEngine:
    """Strix over its driver; `raw_read` is a 1-sim `analyze`, `search` an N-sim one; both carry q/prior/visits per legal cell."""

    raw_derivation = "strix's value head (the mover's) and its root priors, through tools/strix_driver.py analyze"
    head_derivation = "strix gumbel_mcts_with_diagnostics; root_value is the visit-weighted mean of per_child_q (derived)"

    def __init__(self, info: EngineInfo, *, encoding: str) -> None:
        if info.note:
            raise EngineLoadError(info.note)
        python, _driver, cwd, checkpoint, pin = locate_strix()
        stem = str(pin["checkpoint"]).rsplit(".", 1)[0]
        self._transport = DriverTransport(python, DRIVER, cwd)
        reply = self._transport.ask(load_request(str(checkpoint), sims=RAW_SIMS, variant=stem, stem=stem))
        if "error" in reply:
            self._transport.close()
            raise EngineLoadError(f"strix driver failed to load: {reply['error']}")
        self.encoding = encoding
        self.radius = int(lookup(encoding).legal_move_radius)
        self.card: dict[str, Any] = {
            "id": info.id, "run_id": "strix", "step": int(pin.get("checkpoint_train_steps", 0)),
            "sha8": str(pin["checkpoint_sha256"])[:8], "encoding": encoding, "radius": self.radius,
            "search_kind": "gumbel (strix)", "deploy_sims": 256, "params": reply.get("params"),
            "device": reply.get("device"), "threads": None, "seed": 0, "m_actions": DEFAULT_M_ACTIONS,
            "forcing_solver": reply.get("forcing_solver"), "driver": str(DRIVER),
            "encoding_note": "the Board is built with the mantis encoding named here (the legal fence)",
            "gaps": list(GAPS),
        }

    def _ask(self, board: Board, sims: int) -> dict[str, Any]:
        stones = [[int(q), int(r), int(p)] for q, r, p in board.get_stones()]
        reply = self._transport.ask({"op": "analyze", "stones": stones, "to_move": int(board.current_player),
                                     "moves_remaining": int(board.moves_remaining), "sims": int(sims)})
        if "error" in reply:
            raise StrixRefused(f"strix driver: {reply['error']}")
        return reply

    @staticmethod
    def _children(reply: dict[str, Any], *, visits: bool) -> list[ChildInfo]:
        return [((int(q), int(r)), i, float(reply["per_child_prior"][i]), int(reply["visits"][i]) if visits else 0,
                 float(reply["per_child_q"][i])) for i, (q, r) in enumerate(reply["legal"])]

    def raw_read(self, board: Board) -> RawRead:
        """The net's raw value (the mover's) and priors; raises StrixRefused on a board with no p1 stone."""
        t0 = time.perf_counter()
        reply = self._ask(board, RAW_SIMS)
        return RawRead(value=float(reply["raw_value"]), children=self._children(reply, visits=False),
                       ms=(time.perf_counter() - t0) * 1000.0)

    def search(self, board: Board, sims: int) -> Search:
        """Strix's Gumbel search at `sims`; the root value is the visit-weighted mean of per_child_q (stated on the card)."""
        t0 = time.perf_counter()
        reply = self._ask(board, sims)
        total = max(1, sum(int(v) for v in reply["visits"]))
        root = sum(int(v) * float(q) for v, q in zip(reply["visits"], reply["per_child_q"], strict=True)) / total
        move = (int(reply["move"][0]), int(reply["move"][1]))
        return Search(root_value=root, argmax=move, children=self._children(reply, visits=True),
                      root_visits=int(reply["sims"]), quiescence_fires=0, ms=(time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        self._transport.close()


__all__ = ["DRIVER", "GAPS", "STRIX_BUILD", "StrixEngine", "StrixRefused", "strix_info"]
