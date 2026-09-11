"""R349(a): fp32 on `train.device: cpu` is the ONE carve-out to LAW-06.

CARD-OC7-OVERRUN discriminated the 40-60 s CPU step as HOST: bf16 autocast on an AVX2 CPU takes
ATen's generic path and one GEMM at the trainer's edge shape measured 72x slower than fp32. The
carve-out disables the autocast CONTEXT on a CPU trainer; the dtype PIN is untouched, so the
CUDA trainer — every production path — still runs bf16 and an unknown representation still
raises. Both halves are pinned here so neither can drift into the other.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import _microbatch_harness as H
from mantis.model import build_net
from mantis.train.trainer.core import Trainer


def test_a_cpu_trainer_keeps_the_bf16_pin_but_runs_its_step_in_fp32(tmp_path: Path) -> None:
    """Killer: `self._autocast_enabled = self.amp_dtype == torch.bfloat16` (the pre-R349 line)."""
    trainer = H.tiny_graph_trainer(tmp_path)
    assert trainer.device.type == "cpu"
    assert trainer.amp_dtype is torch.bfloat16, (
        "the LAW-06 pin moved — the carve-out is on the autocast CONTEXT, never on the dtype"
    )
    assert trainer._autocast_enabled is False, (
        "a CPU trainer still autocasts to bf16: on an AVX2 host that is the 72x generic GEMM "
        "path CARD-OC7-OVERRUN measured, and the integration tier's bound yields nothing"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="R349(a): the CUDA half needs a card")
def test_a_cuda_trainer_still_autocasts_to_bf16(tmp_path: Path) -> None:
    """Killer: `enabled=False` everywhere — the carve-out widened into a repeal of LAW-06."""
    torch.manual_seed(H.SEED)
    arch = H.tiny_graph_arch()
    trainer = Trainer(build_net(arch), H.graph_config(), arch=arch,
                      checkpoint_dir=tmp_path / "ckpt", device=torch.device("cuda"),
                      train_hparams=H.graph_hparams())
    assert trainer.amp_dtype is torch.bfloat16
    assert trainer._autocast_enabled is True, (
        "the production trainer stopped autocasting: LAW-06's bf16 is what keeps the GINE "
        "sum-aggregation off fp16's 65504 ceiling (F-11), and fp32 doubles the step's bytes"
    )
