"""The named-counter home for every `data/**` skip/truncate loss (LAW-14 / LAW-18).

`data/**` used to lose training rows and corpus games through twelve blind
`except Exception:  # noqa: BLE001` arms plus three un-excepted off-window row DROPS.
Every one of them was invisible: a corpus where one file is corrupt and a corpus where
EVERY game truncates at ply 3 produced byte-identical logs. That is exactly the
"starved vs ineffective" indistinguishability LAW-18 exists to kill.

ONE registry now. :data:`REPLAY_COUNTERS` — the IN-RUN set, fed by the dense replayers
`replay.py` / `replay_v6w25.py` through `train/pretrain/dataset.py` — is DELETED with them
(R346(f)), and so is the `monitor_gates.data_loss_counters` key it published: a registry with
no producer publishing an always-empty mapping is the phantom-input class LAW-07 refuses, and
`{}` from it would have read as "nothing was lost" forever.

  - :data:`PIPELINE_COUNTERS` — the OFFLINE corpus-build set (`corpus_analysis`,
    `corpus_metrics`, `generate`, `human_seeding`, `sources/human`). No `EventSink`
    exists in that context; the consumer is :func:`log_pipeline_losses`, called at each
    offline entry point, which emits the snapshot on the module's own logger.

It is a `BestEffortCounters` — a TOTAL registry, so an untouched label reads 0 rather than
raising, and every label is thread-safe to bump. Labels are per-SITE and per-LOSS on purpose:
one shared bucket cannot tell "one corpus file is corrupt" from "every game is failing".
"""
from __future__ import annotations

from mantis.data._log import get_logger
from mantis.monitor.best_effort import BestEffortCounters

log = get_logger(__name__)

#: Offline corpus-build losses — published by :func:`log_pipeline_losses`.
PIPELINE_COUNTERS = BestEffortCounters()


def log_pipeline_losses(where: str) -> dict[str, int]:
    """Publish (and return) the offline loss snapshot from ``where``.

    The LAW-08 live consumer for :data:`PIPELINE_COUNTERS`. Emitted unconditionally —
    including when the snapshot is empty — because "this stage lost nothing" is the
    reading an operator needs to distinguish a clean corpus from an unmeasured one.
    """
    snapshot = PIPELINE_COUNTERS.snapshot()
    log.info("data_pipeline_losses", where=where,
             total=PIPELINE_COUNTERS.total(), **snapshot)
    return snapshot
