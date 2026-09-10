"""`init_trainer`'s device is required, not defaulted.

`Trainer.__init__` turns a `None` device into CPU, so a default on `init_trainer(device=)`
means a caller that OMITS the argument trains on CPU with no exception and no event — the
same false-clear class as a cpu preflight against a cuda-minted run.

SCOPE: this closes the boot path's seam only. The `or torch.device("cpu")` behind it, in
`Trainer.__init__` and `resume_trainer`, stays — in-tree `Trainer(...)` constructions rely on
it. What this file guarantees is that no BOOT can reach it.
"""
from __future__ import annotations

import inspect

import pytest

from mantis.train.orchestrator import init_trainer


def test_init_trainer_takes_the_device_as_a_required_keyword_with_no_default() -> None:
    """Census: the parameter exists, is keyword-only, and carries NO default.

    MUTATION THAT REDS IT: restore `device: Any = None` — nothing else in the tree notices.
    """
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

    MUTATION THAT REDS IT: the same restore — the call then returns a CPU-pinned trainer.
    The empty config payload is deliberate: the binding raises before the body runs.
    """
    with pytest.raises(TypeError, match="device"):
        init_trainer(config={})  # type: ignore[call-arg]
