"""R349(b): `resume_state_persisted.unreceipted_bundles` is the run's reading of the puller's
lag (the dashboard warns at two); driving the coordinator's REAL publisher is its producer test."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import _microbatch_harness as H
from mantis.monitor.config import MonitorConfig
from mantis.train.bundle import complete_bundles
from mantis.train.bundle_receipts import bundle_member_paths
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.util.hashing import sha256_file
from mantis.util.mirror_receipts import write_receipt


def _coordinator(tmp_path: Path, sink: H.SpySink) -> StepCoordinator:
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    return StepCoordinator(
        monitor_cfg=MonitorConfig(),
        trainer=trainer, buffer=H.uniform_graph_buffer(8), pretrained_buffer=None,
        recent_buffer=None, pool=None, eval_pipeline=None, subsystems=None,
        anchor_state=None, shutdown=ShutdownState(), eval_model=None, bufs=None,
        config=SimpleNamespace(selfplay_stall_timeout_sec=1800.0),
        full_config={"run_id": "lag"}, sink=sink)


def _receipt_all(directory: Path) -> None:
    for manifest in complete_bundles(directory):
        for path in bundle_member_paths(manifest, directory):
            write_receipt(path, mirrored_sha256=sha256_file(path),
                          mirrored_bytes=path.stat().st_size, cycle=1, mirror_id="t")


def test_each_publication_reports_the_unreceipted_bundles(tmp_path: Path) -> None:
    """Killer: the field dropped, or computed before THIS bundle joined the set."""
    sink = H.SpySink()
    coord = _coordinator(tmp_path, sink)
    directory = Path(coord.trainer.checkpoint_dir)
    first = coord.trainer.save_checkpoint({"loss": 1.0})
    coord.persist_resume_state(first)
    coord.trainer.step += 1
    second = coord.trainer.save_checkpoint({"loss": 1.0})
    coord.persist_resume_state(second)
    events = sink.named("resume_state_persisted")
    assert [e["unreceipted_bundles"] for e in events] == [[0], [0, 1]], (
        "the lag reading must count the bundle just published and every retained one before "
        f"it; got {[e['unreceipted_bundles'] for e in events]}"
    )
    # The mirror catches up: the next publication counts only the new bundle.
    _receipt_all(directory)
    coord.trainer.step += 1
    coord.persist_resume_state(coord.trainer.save_checkpoint({"loss": 1.0}))
    assert sink.named("resume_state_persisted")[-1]["unreceipted_bundles"] == [2], (
        "receipted bundles still counted as lag — the reading does not read the receipts"
    )
