"""Minimal structured-logging shim over the stdlib ``logging`` module.

``structlog`` is not a mantis runtime dependency, so the ported corpus modules
cannot call ``structlog.get_logger()``. This shim preserves their
``log.info(event, **fields)`` call sites verbatim (zero churn at the call site)
while routing through stdlib ``logging``. Only the on-wire formatting differs
(fields are appended as ``key=value``); no computed output depends on logging.
"""
from __future__ import annotations

import logging
from typing import Any


class _BoundLogger:
    """stdlib-logging adapter accepting structlog-style keyword fields."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        if fields:
            kv = " ".join(f"{k}={v!r}" for k, v in fields.items())
            self._logger.log(level, "%s %s", event, kv)
        else:
            self._logger.log(level, "%s", event)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)


def get_logger(name: str | None = None) -> _BoundLogger:
    """Return a structlog-style bound logger backed by stdlib ``logging``."""
    return _BoundLogger(logging.getLogger(name or "mantis.data"))
