"""The fused aggregation costs the trainer no more peak memory than the bf16 `index_add_` path it replaced, at the minted caps."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import _microbatch_harness as H
from _gine_oracle import aggregating_with, exact_sum, index_add_aggregation
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.model import arch_from_spec_and_config, build_net
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.trainer.core import Trainer, TrainHParams

_MINTED = production_configs(Path(__file__).resolve().parents[2])[0]
#: One training step just under the minted caps peaks near 9 GiB, so a smaller card cannot run it.
_MIN_DEVICE_GIB = 12


def _big_enough() -> bool:
    return torch.cuda.is_available() and \
        torch.cuda.get_device_properties(0).total_memory >= _MIN_DEVICE_GIB * 1024 ** 3


@pytest.mark.integration
@pytest.mark.skipif(not _big_enough(),
                    reason=f"LOUD SKIP — the cap-regime step needs a CUDA card of >= {_MIN_DEVICE_GIB} GiB")
def test_iii_the_trainer_peak_at_the_caps_does_not_rise(tmp_path: Path) -> None:
    """Real steps just under both minted caps, after a discarded warm-up, the two paths alternated: max(fused) <= max(old)."""
    config = load_config(_MINTED).model_dump()
    caps = resolve_microbatch_caps(config)
    probe = H.uniform_graph_buffer(8)
    ec, nc = H.per_graph_counts(probe.sample_graph_batch(4, augment=False, recent_frac=0.0)[0])
    n_graphs = min(caps.max_edges // int(ec[0]), caps.max_nodes // int(nc[0]))
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(n_graphs + 8), n_graphs)
    arch = arch_from_spec_and_config(H.GSPEC, config)
    hparams = H.graph_hparams(aux_soft_policy=TrainHParams.from_config(config).aux_soft_policy)

    def peak(tag: str) -> int:
        torch.manual_seed(H.SEED)
        trainer = Trainer(build_net(arch), H.graph_config(), arch=arch, checkpoint_dir=tmp_path / tag,
                          device=torch.device("cuda"), train_hparams=hparams)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = int(torch.cuda.max_memory_allocated())
        run_declared_train_step(trainer, replay, H.GSPEC, batch_size=n_graphs, augment=False,
                                recency_weight=0.0, caps_provider=lambda: caps,
                                sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)
        torch.cuda.synchronize()
        got = int(torch.cuda.max_memory_allocated()) - before
        del trainer
        torch.cuda.empty_cache()
        return got

    peak("warm-up")  # one-time cuBLAS and allocator growth would otherwise land on whichever path runs first
    # Alternated, because the allocator's state alone moves one path's peak by ~18 MiB between runs.
    old, new = [], []
    for i in range(2):
        with index_add_aggregation():
            old.append(peak(f"index_add{i}"))
        new.append(peak(f"production{i}"))
    gib = [f"{x / 1024 ** 3:.3f}" for x in old + new]
    print(f"(iii): peak delta bf16 index_add_ {gib[:2]} GiB, fused {gib[2:]} GiB")
    assert max(new) <= max(old), f"the fused aggregation raised the trainer peak: {new} > {old} bytes"
    # PLANTED BREAK: an fp32 [E, H] message copy per layer must read as a rise, or the instrument is blind.
    try:
        with aggregating_with(exact_sum):
            planted = peak("planted")
    except torch.cuda.OutOfMemoryError:
        planted = torch.cuda.get_device_properties(0).total_memory
    print(f"(iii) control: fp32 [E, H] copy peak {planted / 1024 ** 3:.3f} GiB")
    assert planted > max(old), "the planted fp32 message copy did not raise the peak: (iii) cannot see a rise"
