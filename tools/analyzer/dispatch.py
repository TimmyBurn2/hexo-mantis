"""The dispatcher: an engine registry and `handle(request) -> {status, body}`; no threads, no HTTP, no defaults."""
from __future__ import annotations

from typing import Any

from .analysis import analyze
from .engines import MANTIS, SNAPSHOT_GAP, EngineInfo, EngineLoadError, MantisEngine
from .instruments import trace
from .position import PositionRefused, parse_moves

OPS = ("engines", "analyze", "trace")


def _refusal(seq: Any, status: int, reason: str) -> dict[str, Any]:
    return {"status": status, "body": {"seq": seq, "ok": False, "refused": reason}}


class Dispatcher:
    """Owns the engines (loaded on first use, cached) and answers one request at a time on the calling thread."""

    def __init__(self, infos: list[EngineInfo], *, device: str, threads: int | None, strix: bool = False) -> None:
        self.infos = list(infos)
        self._by_id = {info.id: info for info in self.infos}
        self._device, self._threads, self._strix = device, threads, strix
        self._loaded: dict[str, Any] = {}

    def rows(self) -> list[dict[str, Any]]:
        """`/engines`: every row as a dict; a strix row carries its availability (phase 3)."""
        return [info.as_dict() for info in self.infos]

    def _engine(self, info: EngineInfo) -> Any:
        if info.id in self._loaded:
            return self._loaded[info.id]
        if info.kind == SNAPSHOT_GAP:
            raise EngineLoadError(f"{info.id}: {info.note}")
        if info.kind != MANTIS:
            raise EngineLoadError(f"{info.id}: unknown engine kind {info.kind!r}")
        engine: Any = MantisEngine(info, device=self._device, threads=self._threads)
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
        except PositionRefused as exc:
            return _refusal(seq, 400, str(exc))
        return {"status": 200, "body": {"seq": seq, "ok": True, **body}}

    def close(self) -> None:
        for engine in self._loaded.values():
            engine.close()
        self._loaded.clear()


__all__ = ["OPS", "Dispatcher"]
