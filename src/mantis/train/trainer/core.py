"""The Trainer — one gradient step + envelope-v2 checkpoint IO.

>300 justify (R8): the `Trainer` owns the full per-step training surface — optimizer / scaler /
scheduler / EMA lifecycle, the graph-GNN step, the periodic-checkpoint seam (THE one reader of
`train.checkpoint_interval`) and the checkpoint save/load delegates — ONE cohesive responsibility
kept in one file so the numeric contract is greppable. Autocast dtype comes from the DECLARED arch
and the graph path is bf16-pinned (LAW-06); hyperparameters come from the minted `train:` section
through a resolver, so `TrainHParams` is not a second default authority (R1).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

# Canonical stub-exported locations — `torch.amp` itself does not re-export for type checkers.
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.encoding import resolve_from_config
from mantis.model import (
    ModelArch,
    amp_dtype_for,
    arch_from_spec_and_config,
)
from mantis.model import (
    binned_value_loss as _binned_value_loss,
)
from mantis.selfplay.graph_wire_split import GraphEmptyBatchError
from mantis.train import checkpoints
from mantis.train.emit import emit_via
from mantis.train.events import tail_mass_block
from mantis.train.losses import (
    backward_accumulate,
    clip_and_step,
    ragged_policy_ce,
)

_LOG = logging.getLogger(__name__)

def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split params for AdamW weight decay: 2D+ weights decay, 1D params / biases do not. The
    no-decay group counters late-training plasticity loss. Yields exactly TWO param groups."""
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass(frozen=True)
class TrainHParams:
    """Training hyperparameters — the RUNTIME object a `Trainer` constructs and passes around.
    Every field is a REQUIRED constructor argument with NO Python default: `TrainConfig` is the
    sole default authority, and a default here would be a second, independently-editable one."""

    lr: float
    weight_decay: float
    grad_clip: float
    lr_schedule: str
    total_steps: int
    scheduler_t_max: int | None
    eta_min: float
    checkpoint_interval: int
    value_target: str
    policy_target: str
    draw_reward: float
    ply_cap_value: float

    @classmethod
    def from_config(cls, config: Any) -> TrainHParams:
        """Build hparams from a validated `RunConfig`-shaped mapping's `train` section. No flat-key
        fallback: `config['train']` is REQUIRED, with every field present."""
        cfg = config if isinstance(config, dict) else {}
        train = cfg.get("train")
        if not isinstance(train, dict):
            raise ValueError(
                "TrainHParams.from_config: config['train'] is required (R-TRAINCONFIG-SCHEMA "
                "closure) — no flat legacy training keys are read anymore."
            )
        if train["value_target"] != "pure_outcome_z":
            raise ValueError(f"train.value_target: unsupported {train['value_target']!r}")
        _assert_policy_target_consistency(train, cfg.get("search") or {})
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: train[k] for k in fields}
        return cls(**kwargs)


def _assert_policy_target_consistency(train: dict[str, Any], search: dict[str, Any]) -> None:
    """`train.policy_target` must name the target the SEARCH built.

    The `RunConfig`-level `model_validator` enforces this at schema-validate time; this is the
    defensive runtime assertion for a caller whose dict never went through it. A mapping carrying
    no `search` section is not checked — inventing a default kind to compare against would be the
    code-side default R1 forbids.

    Raises:
        ValueError: the declared target is not the one `search.kind` produces.
    """
    kind = search.get("kind")
    if kind is None:
        return
    expected = (
        "completed_improved_policy" if kind == "gumbel" else "raw_visit_distribution"
    )
    if train["policy_target"] != expected:
        raise ValueError(
            "train.policy_target disagrees with search.kind — "
            f"policy_target={train['policy_target']!r}, search.kind={kind!r} "
            f"produces {expected!r}."
        )


class Trainer:
    """Manage one training step and checkpoint IO.

    Args:
        model:  a net built by `mantis.model.build_net(arch)`.
        config: the RunConfig snapshot, or a legacy flat config on resume.
        arch:   the declared arch dataclass, the SOLE arch source at save; derived from `config`
                when omitted.
        checkpoint_dir / device: as named.
        train_hparams: explicit `TrainHParams`; derived from `config` when omitted.
        sink:   the injected `EventSink`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        *,
        arch: ModelArch | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        device: torch.device | None = None,
        train_hparams: TrainHParams | None = None,
        sink: Any = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.config = config
        #: THE LAUNCH PIN'S VERIFICATION SOURCE: `verify_launch_anchor_pin` reads it and FAILS
        #: CLOSED when a pin is set and this is `None`. Filled from `identity.warm_start` on a
        #: fresh launch, `None` on a resume and on any run with no warm-start row.
        self.checkpoint_source: str | Path | None = None
        self._sink = sink
        self.arch: ModelArch = arch if arch is not None else self._derive_arch(config)
        self.hp = train_hparams if train_hparams is not None else TrainHParams.from_config(config)

        # Off the DECLARED arch representation, no module sniff, no default (LAW-11).
        representation = self.arch.representation
        self.amp_dtype = amp_dtype_for(representation)

        # bf16 needs no GradScaler, but the scaler OBJECT stays: `save_checkpoint` writes its
        # state, and dropping it would move the resume bundle's shape.
        self.fp16 = False
        self._scaler_enabled = False
        # R349(a): fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 — bf16 autocast on
        # an AVX2 CPU takes ATen's generic path (72x on one GEMM, CARD-OC7-OVERRUN); the dtype pin
        # itself is untouched and no production path trains on the CPU.
        self._autocast_enabled = (
            self.amp_dtype == torch.bfloat16 and self.device.type == "cuda"
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            build_param_groups(self.model, float(self.hp.weight_decay)),
            lr=float(self.hp.lr),
        )
        self.scheduler = self._build_scheduler()

        from mantis.train.ema import build_ema_model, resolve_ema_config
        # The block is REQUIRED and read as given: a `{}` fallback made an absent block
        # indistinguishable from a declared OFF, which is how a lever stays silently disabled.
        _ema_enabled, _ema_decay, self.ema_update_every = resolve_ema_config(config)
        self.ema_model = (
            build_ema_model(getattr(self.model, "_orig_mod", self.model), decay=_ema_decay)
            if _ema_enabled else None
        )

        self.scaler = GradScaler(device=self.device.type, enabled=self._scaler_enabled)

        self.step = 0
        #: Non-finite guard counters. Both MUST read 0 in a healthy run; non-zero means a NaN/inf
        #: was produced and suppressed before it wrote NaN into every weight.
        self.nonfinite_loss_microbatches = 0
        self.nonfinite_grad_steps = 0
        #: Training steps on which NO optimizer step was taken. Distinct from
        #: `nonfinite_grad_steps`, which counted the same condition AFTER the weights were
        #: overwritten; `self.step` does not advance with this one.
        self.skipped_steps = 0
        #: The injected resume-bundle publisher, called with `(checkpoint_path, step)` AFTER a
        #: periodic checkpoint. `None` is a bench/fixture posture, RECORDED as `bundle: false`.
        self.bundle_publisher: Callable[[Path, int], Any] | None = None
        # Keys the resume F1 defer preserved (empty on a fresh run).
        self.f1_deferred_keys: frozenset[str] = frozenset()
        self.loaded_from_full_checkpoint = False
        self.ckpt_had_value_fc2_bins = False

    @staticmethod
    def _derive_arch(config: Any) -> ModelArch:
        cfg = dict(config) if isinstance(config, dict) else {}
        return arch_from_spec_and_config(_resolve_spec(cfg), cfg)

    def _build_scheduler(self):
        schedule = str(self.hp.lr_schedule or "none").lower()
        if schedule in {"none", "off", "disabled"}:
            return None
        if schedule == "cosine":
            t_max = self.hp.scheduler_t_max if self.hp.scheduler_t_max is not None else self.hp.total_steps
            if t_max is None:
                raise ValueError("lr_schedule: cosine requires total_steps / scheduler_t_max.")
            return CosineAnnealingLR(self.optimizer, T_max=max(1, int(t_max)),
                                     eta_min=float(self.hp.eta_min), last_epoch=-1)
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    def _base_model(self) -> nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    def eval_step_from_graph_batch(
        self,
        *,
        parts: Sequence[Callable[[], Any]],
        policy_denominator: float,
        value_denominator: float,
        total_edges: int,
        total_nodes: int,
        caps_max_edges: int,
        caps_max_nodes: int,
        batch_composition: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """FORWARD-ONLY loss over a partitioned graph batch — no gradient, no state.

        The sibling of `train_step_from_graph_batch`, sharing its `parts` contract and denominators
        so the two numbers are commensurable. What is ABSENT is absent rather than skipped: no
        `zero_grad`, `backward`, `clip_and_step`, `self.step` increment, scheduler step, EMA update
        or `_maybe_periodic_checkpoint`. THE MODE IS RESTORED IN A `finally`, because leaving the
        model in `eval()` would change every LATER step's dropout and normalisation and the loss
        curve would look BETTER for it. The return DELIBERATELY omits `grad_norm` and `lr`, so a
        later edit routing this dict into `grad_norm_hard_abort` raises rather than passing a zero.

        Args:
            parts: zero-arg callables, each materialising one micro-batch lazily.
            policy_denominator: the whole batch's policy denominator.
            value_denominator: the whole batch's value denominator.
            total_edges: edges across the un-split batch, for the caller's records.
            total_nodes: nodes across the un-split batch, for the caller's records.
            caps_max_edges: the resolved micro-batch edge cap.
            caps_max_nodes: the resolved micro-batch node cap.

        Returns:
            `{"loss", "policy_loss", "value_loss"}`, summed over the parts as the training step
            sums them.

        Raises:
            GraphEmptyBatchError: `parts` is empty, so there is nothing to evaluate.
        """
        del total_edges, total_nodes, caps_max_edges, caps_max_nodes  # recorded by the caller
        if len(parts) == 0:
            raise GraphEmptyBatchError(
                "eval_step_from_graph_batch: zero micro-batches — the sampled batch holds no "
                "graphs, so there is no loss to read. Raised rather than returning 0.0, which "
                "a patience stop would read as the best score ever achieved."
            )
        was_training = self.model.training
        loss_total = policy_total = value_total = 0.0
        try:
            self.model.eval()
            with torch.no_grad():
                for make in parts:
                    inputs = make()
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                  enabled=self._autocast_enabled):
                        policy_logits, _value, bin_logits = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                            inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index,
                            inputs.stone_mask, node_offsets=inputs.node_offsets)
                        policy_loss = ragged_policy_ce(
                            policy_logits, inputs.policy_target, inputs.legal_offsets,
                            full_search_mask=inputs.policy_row_weight,
                            explicit_mask=inputs.explicit_mask,
                            tail_mass=inputs.tail_mass,
                            denominator=policy_denominator)
                        value_loss = _binned_value_loss(
                            bin_logits, inputs.outcomes, value_mask=inputs.value_valid,
                            denominator=value_denominator)
                        loss = policy_loss + value_loss
                    if torch.isfinite(loss):
                        loss_total += loss.item()
                        policy_total += policy_loss.item()
                        value_total += value_loss.item()
                    del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss
        finally:
            if was_training:
                self.model.train()
        return {"loss": loss_total, "policy_loss": policy_total, "value_loss": value_total}

    def train_step_from_graph_batch(
        self,
        *,
        parts: Sequence[Callable[[], Any]],
        policy_denominator: float,
        value_denominator: float,
        total_edges: int,
        total_nodes: int,
        caps_max_edges: int,
        caps_max_nodes: int,
        batch_composition: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """One gradient update from a PARTITIONED graph batch. bf16 autocast (LAW-06).

        `parts` is a Sequence of ZERO-ARG CALLABLES and that is load-bearing: `Sequence` gives
        `len()` without consuming anything, and the callables keep materialisation LAZY — already
        collated batches would all be resident at once, defeating the cap while passing every
        count-based oracle. ONE OPTIMIZER STEP PER TRAINING STEP: clipping is nonlinear in the
        whole gradient and `grad_norm` is an armed gate's input. SINGLE TAIL, five keys — a path
        returning a dict without `grad_norm` would feed `grad_norm_hard_abort` a passing `0.0`.
        """
        if len(parts) == 0:
            raise GraphEmptyBatchError(
                "train_step_from_graph_batch: zero micro-batches — the sampled batch holds no "
                "graphs, so this step cannot produce a gradient. Raised BEFORE zero_grad, so "
                "no optimizer state moved: a silent no-op here would let the run report a "
                "step it never took (LAW-14)."
            )
        self.optimizer.zero_grad()
        loss_total = 0.0
        policy_total = 0.0
        value_total = 0.0
        contributing = 0
        # Every row's tail mass alpha, collected across the split so the reading is the STEP's
        # distribution and not one part's.
        tail_alphas: list[float] = []
        for make in parts:
            inputs = make()
            tail_alphas.extend(
                float(v) for v in inputs.tail_mass.detach().reshape(-1).tolist()
            )
            with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                          enabled=self._autocast_enabled):
                # `forward_batch` is GnnNet's real method; `nn.Module.__getattr__` types dynamic
                # attrs as Tensor | Module.
                policy_logits, _value, bin_logits = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                    inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index,
                    inputs.stone_mask, node_offsets=inputs.node_offsets)
                if int(bin_logits.shape[0]) != int(inputs.n_graphs):
                    raise ValueError(
                        f"train_step_from_graph_batch: bin_logits has "
                        f"{int(bin_logits.shape[0])} rows for {int(inputs.n_graphs)} graphs. "
                        "`binned_value_loss` reduces over bin_logits ROWS while the value "
                        "denominator is computed from the per-GRAPH mask, so a mismatch "
                        "would silently make the denominator a second authority over a count "
                        "it does not own.")
                policy_loss = ragged_policy_ce(policy_logits, inputs.policy_target,
                                               inputs.legal_offsets,
                                               full_search_mask=inputs.policy_row_weight,
                                               explicit_mask=inputs.explicit_mask,
                                               tail_mass=inputs.tail_mass,
                                               denominator=policy_denominator)
                value_loss = _binned_value_loss(bin_logits, inputs.outcomes,
                                                value_mask=inputs.value_valid,
                                                denominator=value_denominator)
                loss = policy_loss + value_loss
            # Without this guard one NaN/inf microbatch loss backwards into a NaN clip coefficient,
            # which writes NaN to EVERY weight while the run keeps reporting numbers. SKIPPED, not
            # zeroed — its gradient contribution is undefined — and counted, because a run dropping
            # half its microbatches looks exactly like a healthy one on loss alone (LAW-18).
            if not torch.isfinite(loss):
                self.nonfinite_loss_microbatches += 1
                if (self.nonfinite_loss_microbatches <= 5
                        or self.nonfinite_loss_microbatches % 50 == 0):
                    _LOG.warning(
                        "skipped_nonfinite_loss step=%s n_skipped=%s loss=%s",
                        self.step + 1, self.nonfinite_loss_microbatches,
                        float(loss.detach().item()),
                    )
                del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss
                continue
            backward_accumulate(loss, self.scaler, self._scaler_enabled)
            contributing += 1
            loss_total += loss.item()
            policy_total += policy_loss.item()
            value_total += value_loss.item()
            del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss

        # THE STEP IS TAKEN ONLY IF THERE IS A GRADIENT TO TAKE IT WITH. Both ways there is not
        # used to advance the clock anyway: every micro-batch skipped (`.grad` stays zeroed, so
        # `clip_and_step` returns a finite `0.0` and accumulated momentum genuinely moves the
        # weights), and a non-finite gradient from a FINITE loss, reached by a different route.
        if contributing == 0:
            grad_norm = float("nan")
            self.optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = clip_and_step(self.optimizer, self.scaler, self.model,
                                      self._scaler_enabled, float(self.hp.grad_clip))
        stepped = math.isfinite(grad_norm)
        if stepped:
            self.step += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema_model is not None and self.step % self.ema_update_every == 0:
                self.ema_model.update_parameters(self._base_model())
        else:
            # `nonfinite_grad_steps` is KEPT — the monitor rules and event manifest read it — but
            # now means "a step was refused". `skipped_steps` reconciles `self.step` against
            # wall-clock progress.
            self.nonfinite_grad_steps += 1
            self.skipped_steps += 1
            reason = "no_contributing_microbatch" if contributing == 0 else "nonfinite_gradient"
            _LOG.warning("skipped_step step=%s reason=%s n_skipped=%s grad_norm=%s",
                         self.step, reason, self.skipped_steps, grad_norm)
            emit_via(self._sink, {
                "event": "trainer_step_skipped", "step": self.step, "reason": reason,
                "representation": "graph", "skipped_steps": self.skipped_steps,
                "microbatches": len(parts), "contributing_microbatches": contributing,
                "nonfinite_loss_microbatches": self.nonfinite_loss_microbatches,
                "nonfinite_grad_steps": self.nonfinite_grad_steps,
            })
        lr = self.optimizer.param_groups[0]["lr"]
        # `result` stays the FIVE-key loss_info contract; the counters ride the EVENT instead,
        # because widening `loss_info` changes a contract the gates and checkpoint metadata pin.
        result = {"loss": loss_total, "policy_loss": policy_total,
                  "value_loss": value_total, "grad_norm": grad_norm, "lr": lr}
        # A REFUSED step emits `trainer_step_skipped` INSTEAD: emitting both would put a step in
        # the stream the step counter does not carry, and `periodic_checkpoint` must not fire
        # either — `self.step` did not move, so a crossed cadence boundary would be crossed twice.
        if stepped:
            emit_via(self._sink, {"event": "trainer_step", "step": self.step,
                                  "representation": "graph", **result,
                                  "microbatches": len(parts), "edges": int(total_edges),
                                  "nodes": int(total_nodes),
                                  "caps_max_edges": int(caps_max_edges),
                                  "caps_max_nodes": int(caps_max_nodes),
                                  "nonfinite_loss_microbatches": self.nonfinite_loss_microbatches,
                                  "nonfinite_grad_steps": self.nonfinite_grad_steps,
                                  "skipped_steps": self.skipped_steps,
                                  # What the sampled batch was made of. Rides the step event
                                  # rather than its own: a second event at the same cadence is a
                                  # second thing to keep in sync.
                                  **(batch_composition or {}),
                                  # The same reasoning for the tail mass.
                                  **tail_mass_block(tail_alphas)})
            self._maybe_periodic_checkpoint(result)
        return result

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """The state_dict self-play / eval / promotion consume (EMA weights when EMA is on)."""
        if self.ema_model is not None:
            return self.ema_model.state_dict()
        return self._base_model().state_dict()

    def _resolve_encoding_name(self) -> str | None:
        try:
            return _resolve_spec(dict(self.config)).name
        except Exception as exc:  # noqa: BLE001 — surfaced, but a resolvable config is required
            _LOG.error("checkpoint_encoding_resolve_failed error=%s", exc)
            return None

    def _maybe_periodic_checkpoint(self, loss_info: dict[str, float] | None) -> Path | None:
        """THE periodic-checkpoint seam — the ONE reader of `train.checkpoint_interval`.

        Both step tails share ONE authority for the cadence. `0` disables; a positive `N` fires at
        `N, 2N, 3N, …` against the POST-increment `self.step`, so the boundary is the step whose
        gradient update the artefact contains. The write is `self.save_checkpoint`, the same entry
        legs 2 and 3 call, so the artefact rides the one stamp path (LAW-12). A failure is NOT
        caught (LAW-14) — the counter it bumps is the persist-fatal watchdog's registered input —
        and the event lands AFTER the write, since a pre-emit falsifies the stream on a failed one.
        """
        interval = int(self.hp.checkpoint_interval)
        if interval <= 0 or self.step % interval != 0:
            return None
        path = self.save_checkpoint(loss_info)
        # The ring and the sidecar go NEXT, the manifest that commits all three goes last, and a
        # failure is NOT caught (LAW-14). The publisher returns `None` when it DECLINES — the
        # disk-guard posture declines, because persisting a large ring on the abort that fires
        # BECAUSE THE DISK IS FULL deepens the condition that fired. A decline is `bundle: false`.
        bundled = False
        if self.bundle_publisher is not None:
            bundled = self.bundle_publisher(path, self.step) is not None
        emit_via(self._sink, {
            "event": "periodic_checkpoint_save",
            "step": self.step,
            "interval": interval,
            "representation": self.arch.representation,
            "path": None if path is None else str(path),
            # A checkpoint that is not a continuation point and one that is must be
            # distinguishable in the stream. Before this field they were not.
            "bundle": bundled,
        })
        return path

    def save_checkpoint(self, loss_info: dict[str, float] | None = None) -> Path:
        """Write an envelope-v2 FULL checkpoint via the ONE writer: `{run_id}_{step:08d}_{sha8}.ckpt`,
        immutable stamp, config schema-validated on write. `encoding_name` from the registry resolver,
        `arch` from `self.arch`."""
        cfg = self.config if isinstance(self.config, dict) else {}
        metadata_kwargs = {
            "encoding_name": self._resolve_encoding_name(),
            "run_id": cfg.get("run_id"),
            "arch": self.arch,
            "corpus_sha256": cfg.get("corpus_sha256"),
        }
        return checkpoints.save_checkpoint(
            model=self.model, optimizer=self.optimizer, scaler=self.scaler,
            scheduler=self.scheduler, step=self.step, config=cfg,
            metadata_kwargs=metadata_kwargs, checkpoint_dir=self.checkpoint_dir, kind="full",
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        checkpoint_dir: str | Path | None = None,
        device: torch.device | None = None,
        fallback_config: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        declared_keys: frozenset | set | None = None,
        sink: Any = None,
    ) -> Trainer:
        """Restore a Trainer — thin delegate to `checkpoints.resume_trainer` (§c.7)."""
        return checkpoints.resume_trainer(
            cls, checkpoint_path, fallback_config=fallback_config,
            config_overrides=config_overrides, declared_keys=declared_keys,
            sink=sink, device=device,
        )


def _resolve_spec(config: Any):
    """Resolve the encoding spec from a config. Both the NESTED `identity.encoding` shape and the
    legacy FLAT `encoding` shape are read by `resolve_from_config` itself, the ONE authority for
    where an encoding may be declared; this is just the non-dict coercion its callers need."""
    return resolve_from_config(dict(config) if isinstance(config, dict) else {})
