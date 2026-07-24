"""BotProtocol + RungUnresolvable — the ONE opponent-bot contract (design §a.2 protocol.py).

Every arena/eval opponent (in-repo or a future WP12-R adapter) satisfies this Protocol.
NO temperature, NO think-time parameter: deploy-matched play is argmax-only end to end
(dispatch item 7) — a bot that needed either would not be representable here, by design.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BotProtocol(Protocol):
    """ONE half-ply per `select_move` call.

    `board` is a `mantis._engine.Board` in production; tests may hand a minimal
    duck-typed stand-in exposing `legal_moves()` (and `apply_move()` where relevant).
    """

    def name(self) -> str: ...

    def new_game(self) -> None: ...

    def select_move(self, board: Any) -> tuple[int, int]: ...


class RungUnresolvable(RuntimeError):
    """A ladder rung's bot kind could not be resolved at HEAD.

    Carries `.rung` (the bot kind) and `.reason` — a human-readable string that
    DISTINGUISHES "no env key set" from "env key set but no adapter installed"; never a
    silent fallback to a default host/path (dispatch law: env keys / vendor pins only).
    """

    def __init__(self, *, rung: str, reason: str) -> None:
        super().__init__(f"rung {rung!r} unresolvable: {reason}")
        self.rung = rung
        self.reason = reason


__all__ = ["BotProtocol", "RungUnresolvable"]
