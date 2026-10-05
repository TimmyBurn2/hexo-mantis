"""The dispatcher: an engine registry and `handle(request) -> {status, body}`; no threads, no HTTP, one stated fallback."""
from __future__ import annotations

from typing import Any

from .analysis import analyze
from .engines import MANTIS, SNAPSHOT_GAP, STRIX, EngineInfo, EngineLoadError, MantisEngine
from .instruments import trace
from .position import PositionRefused, parse_moves
from .strix import StrixEngine, StrixRefused, strix_info

OPS = ("engines", "analyze", "trace")
#: The Board strix's card is built with when no mantis engine is loaded: the pinned rung's own fence (radius 8).
STRIX_FALLBACK_ENCODING = "gnn_axis_r8"
#: What a stamp or a net can raise while loading; each becomes a 503 naming the type, never a 500.
_LOAD_FAILURES = (OSError, EOFError, RuntimeError, ValueError)


def refusal(seq: Any, status: int, reason: str) -> dict[str, Any]:
    """The one refusal envelope every route returns."""
    return {"status": status, "body": {"seq": seq, "ok": False, "refused": reason}}


class Dispatcher:
    """Owns the engines (loaded on first use, cached) and answers one request at a time on the calling thread."""

    def __init__(self, infos: list[EngineInfo], *, device: str, threads: int | None, strix: bool = False) -> None:
        self.infos = list(infos) + ([strix_info()] if strix else [])
        self._by_id = {info.id: info for info in self.infos}
        self._device, self._threads = device, threads
        self._loaded: dict[str, Any] = {}

    def rows(self) -> list[dict[str, Any]]:
        """`/engines`: every row as a dict (a strix row's `note` is its availability); touches no engine."""
        return [info.as_dict() for info in self.infos]

    def _engine(self, info: EngineInfo) -> Any:
        if info.id in self._loaded:
            return self._loaded[info.id]
        if info.kind == SNAPSHOT_GAP:
            raise EngineLoadError(f"{info.id}: {info.note}")
        try:
            if info.kind == MANTIS:
                engine: Any = MantisEngine(info, device=self._device, threads=self._threads)
            elif info.kind == STRIX:
                loaded = [e for e in self._loaded.values() if isinstance(e, MantisEngine)]
                engine = StrixEngine(info, encoding=loaded[0].encoding if loaded else STRIX_FALLBACK_ENCODING)
            else:
                raise EngineLoadError(f"{info.id}: unknown engine kind {info.kind!r}")
        except _LOAD_FAILURES as exc:
            raise EngineLoadError(f"{info.id}: {type(exc).__name__}: {exc}") from None
        self._loaded[info.id] = engine
        return engine

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request → `{"status", "body"}`; every refusal is a named string, never an exception."""
        seq = request.get("seq")
        op = request.get("op")
        if op == "engines":
            return {"status": 200, "body": {"engines": self.rows()}}
        if op not in OPS:
            return refusal(seq, 404, f"unknown op {op!r}; the ops are {', '.join(OPS)}")
        info = self._by_id.get(str(request.get("engine")))
        if info is None:
            return refusal(seq, 404, f"unknown engine {request.get('engine')!r}; see /engines")
        try:
            engine = self._engine(info)
        except EngineLoadError as exc:
            return refusal(seq, 503, str(exc))
        try:
            sims = int(request.get("sims", 0))
        except (TypeError, ValueError):
            return refusal(seq, 400, f"sims must be an integer ≥ 0, got {request.get('sims')!r}")
        if sims < 0:
            return refusal(seq, 400, f"sims must be an integer ≥ 0, got {sims}")
        try:
            moves = parse_moves(request.get("moves", ""))
            if op == "analyze":
                body: dict[str, Any] = {"record": analyze(engine, moves, sims, symmetry=bool(request.get("symmetry")))}
            else:
                body = {"trace": trace(engine, moves, sims)}
        except (PositionRefused, StrixRefused) as exc:
            return refusal(seq, 400, str(exc))
        except EngineLoadError as exc:
            self._loaded.pop(info.id, None)
            engine.close()
            return refusal(seq, 503, f"{exc}; the engine is evicted and respawns on the next request")
        return {"status": 200, "body": {"seq": seq, "ok": True, **body}}

    def close(self) -> None:
        for engine in self._loaded.values():
            engine.close()
        self._loaded.clear()


__all__ = ["OPS", "STRIX_FALLBACK_ENCODING", "Dispatcher", "refusal"]
