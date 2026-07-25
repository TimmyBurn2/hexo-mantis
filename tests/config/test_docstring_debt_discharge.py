"""SC-A1/A2 oracle — R-TRAINCONFIG-SCHEMA / R-SELFPLAYCONFIG-SCHEMA debt discharge is
reflected in the docstrings that named it (DESIGN_P2.md §0 / PREREG_P2.md suite #16).

A grep-gate pinning a documentation fact this design treats as load-bearing (LAW-07-
flavored producer test, per PREREG's own framing): `core.py:23-26`/`core.py:96-103` must
no longer assert "training hyperparameters are NOT WP8 config keys" (SC-A1 makes that
claim false — training knobs are now first-class `TrainConfig` schema fields), and
`hparams.py:1-21`/`:9-15`'s R1-exception header text must be gone (SC-A2 discharges
R-SELFPLAYCONFIG-SCHEMA the same way). Prevents silent re-drift: a future edit accidentally
re-pasting either stale docstring is caught here, not rediscovered by archaeology.

GREEN-guard framing inverted: this suite is RED at HEAD (the debt-tagged docstrings are
CURRENTLY present, verbatim, at both cited sites) and turns GREEN only once SC-A1/A2 both
correct them.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PY = REPO_ROOT / "src" / "mantis" / "train" / "trainer" / "core.py"
HPARAMS_PY = REPO_ROOT / "src" / "mantis" / "selfplay" / "hparams.py"


def test_core_py_no_longer_asserts_training_knobs_are_not_config_keys():
    text = CORE_PY.read_text()
    assert "NOT WP8 config keys" not in text, (
        "core.py still asserts the pre-SC-A1 R-TRAINCONFIG-SCHEMA posture — SC-A1 makes "
        "training knobs first-class TrainConfig schema fields; this docstring is now false"
    )
    assert "R-TRAINCONFIG-SCHEMA" not in text or "discharged" in text.lower() or (
        "closure" in text.lower()
    ), (
        "if core.py still names R-TRAINCONFIG-SCHEMA, it must describe the debt as "
        "DISCHARGED/CLOSED, not as a standing R1-exception"
    )


def test_hparams_py_r1_exception_header_is_gone():
    text = HPARAMS_PY.read_text()
    assert "Tracked R1-exception" not in text, (
        "hparams.py still carries the R-SELFPLAYCONFIG-SCHEMA R1-exception header — SC-A2 "
        "discharges this debt (SelfPlayHParams/InferenceHParams now read a validated "
        "SelfplayConfig/InferenceConfig, never a raw flat dict with code-side defaults)"
    )
    assert "R-SELFPLAYCONFIG-SCHEMA" not in text or "discharged" in text.lower() or (
        "closure" in text.lower()
    )
