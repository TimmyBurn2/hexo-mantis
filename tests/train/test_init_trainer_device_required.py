"""⊕ WPMAIN — `init_trainer`'s device is required, not defaulted (RED-TEAM RT-7b).

R126 promoted the train device to a CONFIG FACT (`train.device`, closed
`Literal["cpu","cuda"]`, required, no schema default) and deleted the `--device` flag from
both callers, on the measured ground that a cpu preflight against a cuda-minted run5
false-clears the 16 GiB GPU wall that killed the WPBOX burst (CARD-RUN5-GPU-OOM). In the same
WP, `DiskGuard.__init__`'s five parameter defaults were stripped on the explicit MF-2 ground
that *"a parameter default is a MIGRATED authority, not an absent one"*.

RED-TEAM found the doctrine had not been applied one layer down:
`train/orchestrator.py::init_trainer` carried `device: Any = None`, and
`train/trainer/core.py`'s `self.device = device or torch.device("cpu")` turns that `None`
into CPU — so a caller that simply OMITTED `device=` trained on CPU with no exception and no
event. CARD-RUN5-GPU-OOM's false-clear class, one layer down.

It was LATENT, not live: `build_run_collaborators` is the only production caller (O-A1 pins
that) and it passes `torch.device(config.train.device)`. It was also invisible to every
instrument in the tree — O-A5's `or`-fallback ban scans `run.py` only and only for roots
named `config`/`cfg`; CI gate 11 requires a registered-encoding literal; the consumer
bijection sees a key that IS consumed.

SCOPE, stated (SF-7): this file closes the `init_trainer` half — the boot path's own seam.
The `or torch.device("cpu")` behind it, in `Trainer.__init__` and `resume_trainer`, is
QUEUED (`Q-RT-TRAINER-DEVICE-FALLBACK`), not closed here: four in-tree `Trainer(...)`
constructions under `tests/train/` rely on that default, so removing it is a train-layer
edit rather than a boot-path one. What this file guarantees is that no BOOT can reach it.

Fakes: NONE — a signature census plus one call that never enters the body.
"""
from __future__ import annotations

import inspect

import pytest

from mantis.train.orchestrator import init_trainer


def test_init_trainer_takes_the_device_as_a_required_keyword_with_no_default() -> None:
    """The MF-2 census: the parameter exists, is keyword-only, and carries NO default.

    MUTATION THAT REDS IT: restore `device: Any = None`. Nothing else in the tree notices —
    the one production caller keeps passing the argument, so the whole corpus stays green
    while the omission becomes typeable again."""
    parameters = inspect.signature(init_trainer).parameters
    assert "device" in parameters, (
        "`init_trainer` must still TAKE the device: R126 keeps the collaborator-threading "
        "parameter and kills only the config-fact carriers (the `--device` flag)"
    )
    device = parameters["device"]
    assert device.default is inspect.Parameter.empty, (
        "`init_trainer(device=...)` may carry NO default. `Trainer.__init__` resolves a "
        "`None` to `torch.device(\"cpu\")`, so a default here is the device authority "
        "MIGRATED from the config into a signature (MF-2 Attack B): a caller that omits the "
        "argument trains on CPU silently — CARD-RUN5-GPU-OOM's false-clear class one layer "
        f"down. Got default={device.default!r}"
    )
    assert device.kind is inspect.Parameter.KEYWORD_ONLY, (
        "…and keyword-only, so it can never be supplied positionally by accident"
    )


def test_omitting_the_device_is_a_named_TypeError_and_not_a_cpu_run() -> None:
    """The behavioural half: the omission fails at the CALL, before any model is built.

    MUTATION THAT REDS IT: the same restore. With a default present this call returns a
    trainer pinned to CPU instead of raising — which is precisely the outcome that has no
    exception, no event and no instrument.

    The config payload is deliberately empty: a `TypeError` for a missing required keyword
    is raised by the binding, so the body never runs and this row cannot accidentally become
    a slow model-build test."""
    with pytest.raises(TypeError, match="device"):
        init_trainer(config={})  # type: ignore[call-arg]  — the omission IS the subject
