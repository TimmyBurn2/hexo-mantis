"""The dispatcher: an engine registry and `handle(request) -> {status, body}`; no threads, no HTTP, no defaults."""
from __future__ import annotations

from typing import Any

from .analysis import analyze
from .engines import MANTIS, SNAPSHOT_GAP, STRIX, EngineInfo, EngineLoadError, MantisEngine
from .instruments import trace
from .position import PositionRefused, parse_moves
from .strix import StrixEngine, StrixRefused, strix_info

OPS = ("engines", "analyze", "trace")
#: The Board strix's card is built with when no mantis engine is listed: the pinned rung's own fence.
STRIX_FALLBACK_ENCODING = "gnn_axis_r8"


def _refusal(seq: Any, status: int, reason: str) -> dict[str, Any]:
    return {"status": status, "body": {"seq": seq, "ok": False, "refused": reason}}


class Dispatcher:
    """Owns the engines (loaded on first use, cached) and answers one request at a time on the calling thread."""

    def __init__(self, infos: list[EngineInfo], *, device: str, threads: int | None, strix: bool = False) -> None:
        self.infos = list(infos) + ([strix_info()] if strix else [])
        self._by_id = {info.id: info for info in self.infos}
        self._device, self._threads = device, threads
        self._loaded: dict[str, Any] = {}

    def rows(self) -> list[dict[str, Any]]:
        """`/engines`: every row as a dict (a strix row's `note` is its availability)."""
        return [info.as_dict() for info in self.infos]

    def _strix_encoding(self) -> str:
        loaded = [e for e in self._loaded.values() if isinstance(e, MantisEngine)]
        if loaded:
            return loaded[0].encoding
        first = next((i for i in self.infos if i.kind == MANTIS), None)
        return self._engine(first).encoding if first is not None else STRIX_FALLBACK_ENCODING

    def _engine(self, info: EngineInfo) -> Any:
        if info.id in self._loaded:
            return self._loaded[info.id]
        if info.kind == SNAPSHOT_GAP:
            raise EngineLoadError(f"{info.id}: {info.note}")
        if info.kind == MANTIS:
            engine: Any = MantisEngine(info, device=self._device, threads=self._threads)
        elif info.kind == STRIX:
            engine = StrixEngine(info, encoding=self._strix_encoding())
        else:
            raise EngineLoadError(f"{info.id}: unknown engine kind {info.kind!r}")
        self._loaded[info.id] = engine
        return engine

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request → `{"status", "body"}`; every refusal is a named string, never an exception."""
        seq = request.get("seq")
        op = request.get("op")
        if op == "engines":
            return {"status": 200, "body": {"engines": self.rows()}}
        if op not in OPS:
            return _refusal(seq, 404, f"unknown op {op!r}; the ops are {', '.join(OPS)}")
        info = self._by_id.get(str(request.get("engine")))
        if info is None:
            return _refusal(seq, 404, f"unknown engine {request.get('engine')!r}; see /engines")
        try:
            engine = self._engine(info)
        except EngineLoadError as exc:
            return _refusal(seq, 503, str(exc))
        try:
            sims = int(request.get("sims", 0))
        except (TypeError, ValueError):
            return _refusal(seq, 400, f"sims must be an integer ≥ 0, got {request.get('sims')!r}")
        if sims < 0:
            return _refusal(seq, 400, f"sims must be an integer ≥ 0, got {sims}")
        try:
            moves = parse_moves(request.get("moves", ""))
            if op == "analyze":
                body: dict[str, Any] = {"record": analyze(engine, moves, sims, symmetry=bool(request.get("symmetry")))}
            else:
                body = {"trace": trace(engine, moves, sims)}
        except (PositionRefused, StrixRefused) as exc:
            return _refusal(seq, 400, str(exc))
        return {"status": 200, "body": {"seq": seq, "ok": True, **body}}

    def close(self) -> None:
        for engine in self._loaded.values():
            engine.close()
        self._loaded.clear()


__all__ = ["OPS", "STRIX_FALLBACK_ENCODING", "Dispatcher"]
