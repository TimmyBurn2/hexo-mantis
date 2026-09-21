"""The held-out gap witness (R366(c)): a FROZEN slice of an older ring the run never trains on, read forward-only every `interval` steps against the train loss since the last read — the in-run reading of PROBE-1's C3-1 gap (run8: value 0.13 of 0.51 on 18 of 18 nets)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mantis._engine import HexgBuffer
from mantis.config.resolve.heldout_gap import HeldoutGapSpec
from mantis.train.coordinator.dispatch import run_declared_eval_step
from mantis.util.hashing import sha256_file


class HeldoutSliceError(RuntimeError):
    """The declared ring is not the file the config names, or holds no row."""


class HeldoutSlice:
    """The frozen slice: the ring loaded into its OWN engine buffer, re-seeded before every read so the `batches` production samples are the same rows each time; nothing here touches the training ring."""

    def __init__(self, spec: HeldoutGapSpec, buffer: Any, *, rows: int, ring_path: Path) -> None:
        self.spec = spec
        self.buffer = buffer
        self.rows = rows
        self.ring_path = ring_path
        self.reads = 0

    @classmethod
    def open(cls, spec: HeldoutGapSpec, *, encoding: str, visit_capacity: int, capacity: int,
             root: Path | None = None) -> HeldoutSlice:
        """Load the declared ring into a buffer of the RUN's geometry (the engine refuses another encoding or slot geometry by name); Raises: FileNotFoundError — no file at `spec.ring`; HeldoutSliceError — the sha is not the declared one or the ring holds no row; ValueError — the engine's own refusal."""
        path = Path(spec.ring) if root is None else root / spec.ring
        if not path.exists():
            raise FileNotFoundError(f"train.heldout_gap.ring not found: {path}")
        actual = sha256_file(path)
        if actual != spec.ring_sha256:
            raise HeldoutSliceError(
                f"train.heldout_gap.ring {path} hashes sha256 {actual}, but the config declares "
                f"{spec.ring_sha256}: the slice is NOT the one this run was pre-registered against"
            )
        buffer = HexgBuffer(int(capacity), encoding, int(visit_capacity))
        rows = int(buffer.load_from_path(str(path)))
        if rows < 1:
            raise HeldoutSliceError(f"train.heldout_gap.ring {path} loaded no row")
        return cls(spec, buffer, rows=rows, ring_path=path)

    def read(self, trainer: Any, step_spec: Any, *, batch_size: int, caps_provider: Callable[[], Any],
             sample_threads_provider: Callable[[], int],
             fast_policy_weight_provider: Callable[[], float]) -> dict[str, float]:
        """The mean forward-only `(policy_loss, value_loss)` over the frozen slice's `batches`, the sampler re-seeded first."""
        self.buffer.seed_sampler(self.spec.seed)
        policy = value = 0.0
        for _ in range(self.spec.batches):
            out = run_declared_eval_step(
                trainer, self.buffer, step_spec, batch_size=batch_size, caps_provider=caps_provider,
                sample_threads_provider=sample_threads_provider,
                fast_policy_weight_provider=fast_policy_weight_provider)
            policy += float(out["policy_loss"])
            value += float(out["value_loss"])
        self.reads += 1
        return {"policy_loss": policy / self.spec.batches, "value_loss": value / self.spec.batches}


__all__ = ["HeldoutSlice", "HeldoutSliceError"]
