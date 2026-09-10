"""Every periodic checkpoint is a full resume bundle, or it says it is not.

A checkpoint carries weights, optimizer, scaler and scheduler state but not the replay ring or
the per-stop facts the envelope cannot hold, so a run killed between periodic saves resumed
with a fresh ring wearing a resume's name. Only the two signal-stop legs wrote a ring, so the
property held exactly when the process was shut down politely.

The ring belongs to the buffer and the sidecar's facts belong to the coordinator, so
`_maybe_periodic_checkpoint` stays the ONE reader of `train.checkpoint_interval` and calls an
injected publisher for everything it does not own. A trainer with no publisher still writes its
checkpoint, and the `periodic_checkpoint_save` event says `bundle: false`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _microbatch_harness as H
from mantis.train import bundle as B


def _drive(trainer: Any, buffer: Any, n: int) -> None:
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import _graph_step as production_graph_step

    for _ in range(n):
        wire, _targets = buffer.sample_graph_batch(4, augment=False, recent_frac=0.0)
        # Bound as default arguments, not captured: a closure over the loop variables reads
        # whatever the LAST iteration left them at (ruff B023).
        caps = MicrobatchCapsSpec(*H.non_binding_caps(wire))
        production_graph_step(
            trainer, buffer, H.GSPEC,
            batch_size=4, augment=False, recency_weight=0.0, recent_buffer=None,
            caps_provider=lambda caps=caps: caps,
            sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
        )


def test_a_periodic_save_with_a_publisher_writes_a_complete_bundle(tmp_path: Path) -> None:
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=2)
    buffer = H.uniform_graph_buffer()
    published: list[Path] = []

    def _publisher(checkpoint_path: Path, step: int) -> Path:
        return B.publish_bundle(
            checkpoint_path=checkpoint_path, run_id="t", step=step,
            write_ring=lambda p: Path(p).write_bytes(b"ring"),
            ring_path=B.ring_path_for(checkpoint_path),
            write_sidecar=lambda p: Path(p).write_text("{}", encoding="utf-8"),
            sidecar_path=Path(str(checkpoint_path) + ".resume.json"),
        )

    def _record(path: Path, step: int) -> Path:
        manifest = _publisher(path, step)
        published.append(manifest)
        return manifest

    trainer.bundle_publisher = _record

    _drive(trainer, buffer, 2)

    saves = sink.named("periodic_checkpoint_save")
    assert len(saves) == 1, f"the cadence did not fire exactly once at interval 2: {saves}"
    assert saves[0]["bundle"] is True, "the event does not record that a bundle was published"
    assert published, "the publisher was never called"
    newest = B.newest_complete_bundle(trainer.checkpoint_dir)
    assert newest is not None, "the periodic save produced no complete bundle"
    assert newest.ring is not None, "the bundle carries no ring — it is not a resume point"


def test_a_periodic_save_with_no_publisher_says_so(tmp_path: Path) -> None:
    """Without a publisher the checkpoint is still written — bench harnesses depend on that —
    but `bundle: false` is what stops a partial artefact reading as a complete one."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=2)
    assert trainer.bundle_publisher is None, "a publisher appeared from somewhere"

    _drive(trainer, H.uniform_graph_buffer(), 2)

    saves = sink.named("periodic_checkpoint_save")
    assert len(saves) == 1
    assert saves[0]["bundle"] is False
    assert saves[0]["path"] is not None, "the checkpoint itself was not written"
    assert B.newest_complete_bundle(trainer.checkpoint_dir) is None, (
        "a bundle appeared without a publisher, so `bundle: false` is a lie"
    )


def test_the_manifest_step_is_the_checkpoints_own_step(tmp_path: Path) -> None:
    """A bundle certifies a checkpoint, so its step must come FROM that checkpoint.

    The coordinator refreshes `_train_step` AFTER `_run_training_step` returns while the
    periodic seam fires INSIDE it, so a manifest built from that counter is one step behind the
    artefact it names — two authorities over one number, and a resume re-enters at the wrong
    step, with nothing failing.
    """
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=2)
    seen: list[tuple[str, int]] = []

    def _publish(checkpoint_path: Path, step: int) -> Path:
        seen.append((Path(checkpoint_path).name, step))
        return B.publish_bundle(
            checkpoint_path=checkpoint_path, run_id="t",
            # THE POINT: the step is read off the checkpoint, exactly as the coordinator does.
            step=B.step_of(checkpoint_path) or -1,
            write_ring=lambda p: Path(p).write_bytes(b"ring"),
            ring_path=B.ring_path_for(checkpoint_path),
            write_sidecar=lambda p: Path(p).write_text("{}", encoding="utf-8"),
            sidecar_path=Path(str(checkpoint_path) + ".resume.json"),
        )

    trainer.bundle_publisher = _publish
    _drive(trainer, H.uniform_graph_buffer(), 2)

    manifest = B.newest_complete_bundle(trainer.checkpoint_dir)
    assert manifest is not None
    assert manifest.step == 2, f"the manifest names step {manifest.step}, the checkpoint is 2"
    assert manifest.checkpoint.name == f"{seen[-1][0]}"
    assert B.step_of(manifest.checkpoint.name) == manifest.step, (
        "the manifest's step and its checkpoint's filename disagree — two authorities over "
        "one number, which is what deriving it from the filename makes impossible"
    )


def test_a_publisher_failure_is_run_fatal_and_not_swallowed(tmp_path: Path) -> None:
    """A bundle that failed to publish must not be reported as one that did."""
    import pytest

    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=2)

    def _explode(_path: Path, _step: int) -> Path:
        raise OSError("planted publisher failure")

    trainer.bundle_publisher = _explode

    with pytest.raises(OSError, match="planted publisher failure"):
        _drive(trainer, H.uniform_graph_buffer(), 2)

    assert not sink.named("periodic_checkpoint_save"), (
        "the save event was emitted for a bundle that did not publish — the stream now claims "
        "a resume point that does not exist"
    )
