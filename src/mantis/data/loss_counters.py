"""The named-counter home for every `data/**` skip/truncate loss (LAW-14 / LAW-18).

`data/**` used to lose training rows and corpus games through twelve blind
`except Exception:  # noqa: BLE001` arms plus three un-excepted off-window row DROPS.
Every one of them was invisible: a corpus where one file is corrupt and a corpus where
EVERY game truncates at ply 3 produced byte-identical logs. That is exactly the
"starved vs ineffective" indistinguishability LAW-18 exists to kill.

Two registries, because the arms live in two execution contexts and must not share a
readout:

  - :data:`REPLAY_COUNTERS` — the IN-RUN set. `mantis.train.pretrain.dataset` imports
    `mantis.data.replay.replay_game_to_triples`, so `replay.py` / `replay_v6w25.py`
    execute inside a live training run. Their snapshot is published in the coordinator's
    `monitor_gates` event under `data_loss_counters` (the LAW-18 in-run channel), read
    LIVE as a module attribute — never from-imported (the buffer_persist counter-binding
    rule). The readout cadence is `monitor.gate_interval` — the ARMING stride, NOT the
    narration stride `train.log_interval` (R242): the whole point is that the loss is
    readable long before the first narration boundary.
  - :data:`PIPELINE_COUNTERS` — the OFFLINE corpus-build set (`corpus_analysis`,
    `corpus_metrics`, `generate`, `human_seeding`, `sources/human`). No `EventSink`
    exists in that context; the consumer is :func:`log_pipeline_losses`, called at each
    offline entry point, which emits the snapshot on the module's own logger.

Both are `BestEffortCounters` — a TOTAL registry, so an untouched label reads 0 rather
than raising, and every label is thread-safe to bump. Labels are per-SITE and per-LOSS
on purpose: one shared bucket cannot tell "one corpus file is corrupt" from "every game
is failing".
"""
from __future__ import annotations

from mantis.data._log import get_logger
from mantis.monitor.best_effort import BestEffortCounters

log = get_logger(__name__)

#: In-run (pretrain-reachable) losses — published to the event sink.
REPLAY_COUNTERS = BestEffortCounters()

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
