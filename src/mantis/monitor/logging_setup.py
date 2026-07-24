"""Human-diagnostic stderr logging (WP13-A §a.1) — stdlib only.

Replaces `hexo_rl/monitoring/configure.py`. What DIES and why: **structlog** (mantis takes
no such dependency — machine-readable event DATA now flows through the JSONL event sink,
which is the ONE channel, not through a log channel) and the **500 MB gzip-rotating file
handler** (the sink owns run-scoped persistence and rotates per run segment, §11 log
identity). What remains is exactly the human-diagnostic half: a level and a stream.
"""
from __future__ import annotations

import logging
import sys
from typing import TextIO

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_MANTIS_HANDLER = "_mantis_monitor_handler"


def configure_logging(
    level: int = logging.INFO, stream: TextIO | None = None
) -> logging.Handler:
    """Install (or replace) the ONE mantis stderr handler on the root logger.

    Idempotent: a previously installed mantis handler is removed first, so repeated calls
    (a resumed run, a supervisor relaunch in-process) never duplicate every line.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _MANTIS_HANDLER, False):
            root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    setattr(handler, _MANTIS_HANDLER, True)
    root.addHandler(handler)
    root.setLevel(level)
    return handler
