"""The teacher: a pinned Six network's raw policy and value through ONNX Runtime, one evaluation per position, no search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mantis.bots.six import locate_six
from mantis.bots.strix import find_vendor_root

DEFAULT_VARIANT = "gen0455"


class TeacherUnavailable(RuntimeError):
    """The teacher's runtime or its requested execution provider is not usable here."""


def lineage_tag(commit: str, variant: str) -> str:
    """The tag every labelled record carries: `six-<commit7>-gen<generation>` (gen0455 -> gen455)."""
    match = re.fullmatch(r"gen0*(\d+)", variant)
    gen = match.group(1) if match else variant
    return f"six-{commit[:7]}-gen{gen}"


@dataclass(frozen=True)
class Outputs:
    """One batch's raw heads: policy logits over the crop `[B, 625]`, value logits `[B, 2]` (mover wins, loses), score `[B]`."""

    policy: np.ndarray
    value_logits: np.ndarray
    score: np.ndarray

    @property
    def value(self) -> np.ndarray:
        """P(mover wins) − P(mover loses), the engine's own reduction of the two value logits."""
        return np.tanh(0.5 * (self.value_logits[:, 0] - self.value_logits[:, 1])).astype(np.float32)


class Teacher:
    """The pinned network (sha-verified) in an ONNX Runtime session, TF32 off on CUDA; Raises: TeacherUnavailable — no onnxruntime or no CUDA served; RungUnresolvable — the pin or network does not resolve."""

    def __init__(self, *, device: str, variant: str = DEFAULT_VARIANT, vendor_root: Path | None = None) -> None:
        assets = locate_six(vendor_root or find_vendor_root(), variant)
        try:
            import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # the `teacher` extra
        except ImportError as exc:
            raise TeacherUnavailable("onnxruntime is absent: sync with `--extra teacher`") from exc
        if device == "cuda":
            ort.preload_dlls(directory="")
            providers: list[Any] = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(assets.net), options, providers=providers)
        served = self.session.get_providers()[0]
        if device == "cuda" and served != "CUDAExecutionProvider":
            raise TeacherUnavailable(f"CUDA was asked for and the session serves {served}")
        self.provider = served
        self.ort_version = str(ort.__version__)
        self.net_path = assets.net
        self.net_sha256 = assets.net_sha256
        self.commit = assets.commit
        self.variant = variant
        self.lineage = lineage_tag(assets.commit, variant)

    def evaluate(self, planes: np.ndarray) -> Outputs:
        """Raw heads for a batch of planes `[B, 8, 25, 25]` float32."""
        policy, _opponent, value, score = self.session.run(None, {"planes": np.ascontiguousarray(planes, np.float32)})
        return Outputs(policy=np.asarray(policy, np.float32), value_logits=np.asarray(value, np.float32),
                       score=np.asarray(score, np.float32))

    def record(self) -> dict[str, str]:
        """The provenance a labelled set carries."""
        return {"lineage": self.lineage, "six_commit": self.commit, "variant": self.variant,
                "net": str(self.net_path.name), "net_sha256": self.net_sha256, "provider": self.provider,
                "onnxruntime": self.ort_version, "tf32": "off"}
