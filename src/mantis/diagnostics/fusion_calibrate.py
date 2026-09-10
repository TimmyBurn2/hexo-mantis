# >300 justify (R8): this module is ONE CLAIM — "here is the measured pair
# `(max_fused_edges, max_fused_nodes)` that fits inside the budget you named, and here is every
# mechanism that produced it". A fit reported away from its sweep, or a pair reported away from
# the allocator posture it was measured under, is a number without its producing mechanism.
"""`python -m mantis.diagnostics.fusion_calibrate` — measure the fused graph inference
forward's peak allocation and RECOMMEND `inference.fused_graph_caps`.

It measures, fits, recommends and prints a copy-pasteable mint line; it NEVER mints and never
writes a config. It lives under `src/mantis` rather than `tools/` because it imports the
PRODUCTION forward: a calibration that skips the softmax or the D2H copies bounds a program
nobody runs.

- **CUDA ONLY.** The quantity fitted is a CUDA caching-allocator peak, so a host without one
  exits 2 with a named refusal and NO cap — never a degraded CPU estimate. `--shapes-only`
  runs the device-free half and reports `"calibrated": false`.
- **PEAK IS A DELTA**: `max_memory_allocated() - before`, around a `reset_peak_memory_stats`.
  An absolute reading charges this forward for whatever the co-resident trainer held.
- **THE BUDGET IS THE OPERATOR'S.** `--budget-bytes` has NO default; it is waived only by
  `--shapes-only` and `--no-recommend`.
- **THE REPORT GOES WHERE YOU SAY**: `--out`, else stdout. No invented filenames.

The sweep carries an explicit node-heavy / edge-light point because `peak ~ a + b*E + c*N` is
identifiable only if E/N VARIES: at one ratio `b` and `c` are collinear and `c` is a solver
artefact. `train.microbatch_caps` is this bound's partner in one budget, so the operator may
have to lower the training cap to buy the inference one; the budget block states every term it
subtracted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mantis.config.resolve.allocator_posture import (
    read_live_allocator_conf as _live_alloc_conf,
)
from mantis.model.amp import amp_dtype_for

_TOOL = "mantis.diagnostics.fusion_calibrate"
_KEY = "inference.fused_graph_caps"
#: The block's two member names, held ONCE as data. The resolver READS them off a config; this
#: tool only writes them, so constant subscripts here would read as a second authority.
_MEMBER_NAMES = ("max_fused_edges", "max_fused_nodes")


class CalibrationRefusal(Exception):
    """A named refusal that exits 2 and emits no cap, so every refusal takes ONE exit path with
    one code and one destination — a refusal on stdout would be parsed as a report."""


@dataclass(frozen=True)
class SweepPoint:
    """One `(regime, n_graphs)` cell of the sweep, with the shapes it actually built."""

    label: str
    n_graphs: int
    n_nodes: int
    n_edges: int
    max_graph_nodes: int
    max_graph_edges: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_graphs": self.n_graphs,
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "edges_per_node": round(self.n_edges / max(self.n_nodes, 1), 4),
            "largest_graph_nodes": self.max_graph_nodes,
            "largest_graph_edges": self.max_graph_edges,
        }


#: The three regimes the sweep spans: `stones` is the position's stone count (Ply counts
#: STONES, not turns), `spread` the lattice step between stones. The parameters were CHOSEN BY
#: MEASUREMENT — a scan over `stones x spread` put the reachable E/N band at roughly 21.2 to
#: 29.0, scattering stones does NOT starve the edge term (each gets its own dense legal-move
#: ball), and the rows take the measured minimum, maximum and a mid-game point. That band is
#: narrow enough that `_fit` DISCLOSES the span rather than reporting an unidentified `c`.
_REGIMES: tuple[tuple[str, int, int], ...] = (
    ("late_game_packed_low_ratio_node_heavy", 120, 1),
    ("mid_game_packed", 40, 1),
    ("late_game_spread_high_ratio_edge_heavy", 120, 5),
)
_GRAPH_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48, 64)


def _positions(stones: int, spread: int) -> list[tuple[int, int, int]]:
    """Build a legal stone list on a `spread`-spaced hex spiral — deterministic and
    geometry-only, so two runs build the same sweep and their fits are comparable."""
    out: list[tuple[int, int, int]] = []
    q = r = 0
    ring = 1
    while len(out) < stones:
        for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            for _ in range(ring):
                if len(out) >= stones:
                    break
                out.append((q, r, 1 if len(out) % 2 == 0 else -1))
                q += dq * spread
                r += dr * spread
            if len(out) >= stones:
                break
        ring += 1
    return out


def _empty_near(placed: list[tuple[int, int, int]], want: int = 2) -> list[tuple[int, int]]:
    """Return `want` EMPTY cells within hex distance 2 of the first stone, in a fixed order.

    Near a stone because the producer's legal-node set is the empty cells within the legal-move
    radius of SOME stone; distance 2 is inside the smallest radius the registry declares.
    """
    occupied = {(q, r) for q, r, _ in placed}
    q0, r0, _ = placed[0]
    found: list[tuple[int, int]] = []
    for dq in range(-2, 3):
        for dr in range(-2, 3):
            if abs(dq + dr) > 2:
                continue
            cell = (q0 + dq, r0 + dr)
            if cell not in occupied:
                found.append(cell)
                if len(found) == want:
                    return found
    raise CalibrationRefusal(
        f"{_TOOL}: could not find {want} empty legal cells beside the first stone of a "
        f"{len(placed)}-stone sweep position; the generator cannot build a valid visit row."
    )


#: The label the CORPUS source's single regime carries: a corpus has ONE distribution, and
#: three "regimes" over one ring would be three names for the same sample.
CORPUS_REGIME = "corpus"


def _load_corpus(encoding: str, max_moves: int, corpus_path: str) -> Any:
    """Load the run's OWN replay ring once and sample it per point — a cap fitted to a
    generator's geometry is fitted to a distribution the run does not have. Loading ZERO records
    is a REFUSAL: a tool that silently fitted nothing would emit a pair with no evidence."""
    from mantis._engine import HexgBuffer

    buf = HexgBuffer(1 << 20, encoding, max_moves)
    try:
        loaded = buf.load_from_path(corpus_path)
    except (OSError, ValueError) as exc:
        # Re-raised, never swallowed: the loader's refusals are correct and kept verbatim, but
        # they arrive with the path nowhere in them.
        raise CalibrationRefusal(
            f"{_TOOL}: --corpus-path {corpus_path} could not be read as a HEXG ring for "
            f"encoding {encoding!r}: {exc}"
        ) from exc
    if int(loaded) < 1:
        raise CalibrationRefusal(
            f"{_TOOL}: --corpus-path {corpus_path} loaded {loaded} records. A calibration "
            "fitted to an empty corpus is a pair with no evidence behind it; supply a ring "
            "with records, or use --source synthetic and say so in the sitting record."
        )
    return buf


def _build_wire(encoding: str, max_moves: int, n_graphs: int, stones: int, spread: int) -> Any:
    """Build one fused wire of `n_graphs` graphs through the REAL producer — the shapes fitted
    against must be shapes the production builder emits, or the cap bounds a geometry the run
    never sees."""
    from mantis._engine import HexgBuffer

    buf = HexgBuffer(max(n_graphs * 2, 8), encoding, max_moves)
    placed = _positions(stones, spread)
    # The visit row must be a DISTRIBUTION over EMPTY cells inside the rebuilt legal-node set;
    # the sampler refuses a row whose mass drops at align.
    visits = [(q, r, w) for (q, r), w in zip(_empty_near(placed), (0.6, 0.4), strict=True)]
    for i in range(max(n_graphs, 1)):
        buf.push_graph_position(
            placed, visits, 1, 30, 2 + i, True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i,
        )
    wire, _targets = buf.sample_graph_batch(n_graphs, augment=False, recent_frac=0.0)
    return wire


def _wire_for(
    encoding: str, max_moves: int, n_graphs: int, label: str, stones: int, spread: int,
    corpus: Any,
) -> Any:
    """Decide the sweep's data SOURCE — one dispatch and not two paths, because a
    `--source corpus` run that quietly fitted GENERATED graphs would emit a pair carrying the
    corpus's authority and the generator's geometry."""
    if corpus is not None:
        wire, _targets = corpus.sample_graph_batch(n_graphs, augment=False, recent_frac=0.0)
        return wire
    return _build_wire(encoding, max_moves, n_graphs, stones, spread)


def _regimes(corpus: Any) -> tuple[tuple[str, int, int], ...]:
    """The sweep's regimes for the chosen source — three for synthetic, ONE for a corpus."""
    return ((CORPUS_REGIME, 0, 0),) if corpus is not None else _REGIMES


def _sweep(encoding: str, max_moves: int, counts: tuple[int, ...],
           corpus: Any = None) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for label, stones, spread in _regimes(corpus):
        for n_graphs in counts:
            wire = _wire_for(encoding, max_moves, n_graphs, label, stones, spread, corpus)
            node_counts = np.diff(np.asarray(wire.node_offsets, dtype=np.int64))
            edge_counts = np.diff(np.asarray(wire.edge_offsets, dtype=np.int64))
            points.append(SweepPoint(
                label=label, n_graphs=n_graphs,
                n_nodes=int(node_counts.sum()), n_edges=int(edge_counts.sum()),
                max_graph_nodes=int(node_counts.max()), max_graph_edges=int(edge_counts.max()),
            ))
            del wire
    return points


def _measure_point(
    net: Any, spec: Any, device: Any, encoding: str, max_moves: int,
    point: SweepPoint, stones: int, spread: int, repeats: int, corpus: Any = None,
) -> dict[str, Any]:
    """Run the PRODUCTION forward `repeats` times and report the MEDIAN peak delta — one
    allocation retry or cache state makes a single reading unrepeatable."""
    import torch

    from mantis.selfplay.graph_collate import (
        collate_graph_batch,
        segment_softmax,
        stone_mask_from_batch,
    )

    peaks: list[int] = []
    allocated_abs: list[int] = []
    reserved: list[int] = []
    # Baseline for the per-point retry/OOM deltas: `memory_stats()`'s counters are
    # process-cumulative and `reset_peak_memory_stats` does NOT reset them.
    _base = torch.cuda.memory_stats()
    before_retries = int(_base.get("num_alloc_retries", 0))
    before_ooms = int(_base.get("num_ooms", 0))
    for _ in range(repeats):
        wire = _wire_for(encoding, max_moves, point.n_graphs, point.label, stones,
                         spread, corpus)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.max_memory_allocated()
        batch = collate_graph_batch(
            wire, expected_version=1, trunk_size=spec.trunk_size,
            win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
            edge_feat_dim=spec.edge_feat_dim, device=str(device), semantic="canary",
            canary_period=64,
        )
        stone_mask = stone_mask_from_batch(batch)
        # `amp_dtype_for` is the ONE dtype authority; a literal `torch.bfloat16` beside it
        # happens to be right for the graph path, which is exactly why it was invisible.
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype_for(str(spec.representation)),
            enabled=True,
        ):
            policy_logits, value, _bins = net.forward_batch(
                batch.x, batch.edge_index, batch.edge_attr, batch.legal_node_gather,
                stone_mask, batch.node_offsets,
            )
        probs = segment_softmax(policy_logits.float(), batch.legal_offsets)
        if not bool(torch.isfinite(probs).all()) or not bool(torch.isfinite(value).all()):
            raise CalibrationRefusal(
                f"{_TOOL}: the production forward produced NaN/Inf at "
                f"{point.label} n_graphs={point.n_graphs}. A peak measured through a broken "
                "forward is not a measurement of the forward this run uses."
            )
        # The three D2H copies are part of the program being measured.
        _ = probs.detach().cpu().numpy()
        _ = batch.legal_offsets.detach().cpu().numpy()
        _ = value.detach().float().cpu().numpy().reshape(-1)
        torch.cuda.synchronize()
        peaks.append(int(torch.cuda.max_memory_allocated()) - int(before))
        # The ABSOLUTE allocated peak beside the delta: `reserved_peak / allocated_peak` must
        # come from ONE reset, or the ratio has the right units and no meaning.
        allocated_abs.append(int(torch.cuda.max_memory_allocated()))
        stats = torch.cuda.memory_stats()
        reserved.append(int(stats["reserved_bytes.all.peak"]))
        del wire, batch, stone_mask, policy_logits, value, probs
    stats = torch.cuda.memory_stats()
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        **point.as_dict(),
        "peak_bytes_median": int(np.median(peaks)),
        "peak_bytes_all_repeats": peaks,
        "allocated_bytes_peak_median": int(np.median(allocated_abs)),
        "reserved_bytes_peak_median": int(np.median(reserved)),
        # These counters are PROCESS-CUMULATIVE, so the raw reading published under a
        # per-point key reported every OOM the whole sweep ever had. The DELTA sits beside it.
        "num_alloc_retries": int(stats["num_alloc_retries"]) - int(before_retries),
        "num_ooms": int(stats["num_ooms"]) - int(before_ooms),
        "num_alloc_retries_process_total": int(stats["num_alloc_retries"]),
        "num_ooms_process_total": int(stats["num_ooms"]),
        "mem_get_info_free_bytes": int(free_b),
        "mem_get_info_total_bytes": int(total_b),
    }


def _fit(measured: list[dict[str, Any]]) -> dict[str, Any]:
    """Least squares for `peak ~ a + b*E + c*N`, with R^2 and the max residual beside it — R^2
    alone hides a single badly-missed point, and for a memory bound that is the one that OOMs."""
    design = np.array([[1.0, float(m["edges"]), float(m["nodes"])] for m in measured])
    observed = np.array([float(m["peak_bytes_median"]) for m in measured])
    coeffs, *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ coeffs
    residuals = observed - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    ratios = [m["edges"] / max(m["nodes"], 1) for m in measured]
    span = max(ratios) / max(min(ratios), 1e-9)
    return {
        # DISCLOSED, not buried: over a narrow E/N band the design matrix's columns are nearly
        # proportional, so the TOTAL is measured and the SPLIT between `b` and `c` is not.
        "edges_per_node_span_ratio": span,
        "coefficients_are_separately_identified": bool(span >= 1.5),
        "identifiability_note": (
            "b and c are identifiable only through variation in E/N. This sweep spans "
            f"{min(ratios):.2f}..{max(ratios):.2f} (ratio {span:.2f}x). Below 1.5x, read the "
            "PREDICTED PEAK as measured and the b/c split as unresolved; the recommended pair "
            "is unaffected, because it is solved at the measured operating ratio where the "
            "two columns are not separated in the first place."
        ),
        "a_bytes": float(coeffs[0]),
        "b_bytes_per_edge": float(coeffs[1]),
        "c_bytes_per_node": float(coeffs[2]),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "max_residual_bytes": float(np.max(np.abs(residuals))),
        "measured_edges_per_node_min": float(min(ratios)),
        "measured_edges_per_node_max": float(max(ratios)),
        "operating_edges_per_node": float(np.median(ratios)),
        # DERIVED FROM CODE, NOT MEASURED — a prediction to be CHECKED against the fit, and an
        # UPPER BOUND, since CPython frees rebound intermediates earlier.
        "derived_not_measured_expectation": {
            "b_bytes_per_edge_upper_bound": 1316,
            "c_bytes_per_node_upper_bound": 3686,
            "source": "F816_10_DESIGN sections 1.2/1.3, hand-derived from tensor shapes",
        },
    }


def _recommend(
    fit: dict[str, Any], budget_bytes: int, margin: float, points: list[dict[str, Any]],
    unmeasured: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Solve `a + b*E + c*N <= budget * margin` with `E/N` pinned to the measured ratio,
    publishing BOTH the requested `margin` and the `margin_achieved` the pair actually occupies."""
    ratio = fit["operating_edges_per_node"]
    usable = budget_bytes * margin - fit["a_bytes"]
    per_node = fit["b_bytes_per_edge"] * ratio + fit["c_bytes_per_node"]
    if usable <= 0 or per_node <= 0:
        raise CalibrationRefusal(
            f"{_TOOL}: the fit admits no positive pair under a budget of {budget_bytes} bytes "
            f"at margin {margin} (fixed term {fit['a_bytes']:.0f} B, per-node term "
            f"{per_node:.1f} B). Either the budget is below the forward's fixed cost or the "
            "fit is not usable; no pair is emitted, because a non-positive cap is not a cap."
        )
    nodes = int(usable // per_node)
    edges = int(nodes * ratio)
    # The OUTPUT condition, separate from the input one: `usable > 0` admits a `usable` so small
    # that the floor division lands on ZERO, letting the refusal's own sentence be violated by
    # the branch that skipped it.
    if nodes < 1 or edges < 1:
        raise CalibrationRefusal(
            f"{_TOOL}: the fit's usable budget ({usable:.0f} B after the {margin} margin and "
            f"the {fit['a_bytes']:.0f} B fixed term) buys {nodes} nodes / {edges} edges — "
            "not a cap. A pair with a non-positive member would be refused by the schema's "
            "`ge=1` anyway; it is refused HERE so the message names the budget that produced "
            "it. Raise --budget-bytes, or re-fit: no pair and no mint line are emitted."
        )
    predicted = (fit["a_bytes"] + fit["b_bytes_per_edge"] * edges
                 + fit["c_bytes_per_node"] * nodes)
    # "Seen in this sweep" over MEASURED points only left an OOM'd point — the biggest shapes —
    # unseen, so the flag read False for a pair that cannot hold a graph the sweep built.
    seen = list(points) + list(unmeasured or [])
    largest_nodes = max(p["largest_graph_nodes"] for p in seen)
    largest_edges = max(p["largest_graph_edges"] for p in seen)
    return {
        "max_fused_edges": edges,
        "max_fused_nodes": nodes,
        "budget_bytes": budget_bytes,
        "margin": margin,
        "predicted_peak_bytes": predicted,
        # Strictly below `margin` because `nodes` is a FLOOR division: the fraction landed on
        # is the measurement, `margin` only the ceiling solved against.
        "margin_achieved": predicted / budget_bytes,
        "pinned_edges_per_node": ratio,
        # The floor BELOW which the cap starts refusing single graphs.
        "largest_single_graph_nodes_seen": largest_nodes,
        "largest_single_graph_edges_seen": largest_edges,
        # The denominator, published: how many points the maxima were taken over.
        "seen_points": len(seen),
        "seen_points_unmeasured": len(unmeasured or []),
        "refuses_a_graph_seen_in_this_sweep": bool(
            edges < largest_edges or nodes < largest_nodes
        ),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[3]), check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _provenance(config_path: Path, spec: Any, arch: Any) -> dict[str, Any]:
    """Return every mechanism behind every number above."""
    import torch

    cuda = torch.cuda.is_available()
    _alloc_conf = _live_alloc_conf()
    return {
        "tool": _TOOL,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
        "device_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda else None
        ),
        # The allocator posture is STAMPED because a cap does not transfer across a change of
        # it. Read through the ONE authority: c10 honours `PYTORCH_ALLOC_CONF` as a fallback, so
        # reading `PYTORCH_CUDA_ALLOC_CONF` alone ASSERTED the default posture for a run that was
        # not in it. The variable the value came FROM is stamped beside it.
        "pytorch_cuda_alloc_conf": _alloc_conf.raw,
        "pytorch_alloc_conf_source_var": _alloc_conf.source_var,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "encoding_spec": spec.name,
        "encoding_registry_sha": getattr(spec, "registry_sha", None),
        "arch": repr(arch),
        "git_head": _git_head(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {_TOOL}",
        description=(
            "Measure the fused graph inference forward's peak CUDA allocation over a sweep "
            f"of fused batches, fit peak ~ a + b*E + c*N, and recommend a {_KEY} pair for a "
            "budget you supply. Never mints, never writes a config."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="The run config to calibrate FOR, read through the real loader. Its encoding, "
             "geometry and batch sizes are the sweep's shapes.",
    )
    parser.add_argument(
        "--budget-bytes", type=int, metavar="BYTES",
        help="Total inference-side byte budget the recommended pair must fit under. Carries "
             "no fallback value and none is invented: every cap this tool emits is a "
             "function of it, so a budget nobody minted would be a cap nobody measured. "
             "Required for a recommending run; waived when the run is not recommending.",
    )
    parser.add_argument(
        "--source", choices=("corpus", "synthetic"), default="synthetic",
        help="Where the sweep's graphs come from: 'corpus' samples a real replay ring at the "
             "run's own operating distribution (preferred at the box); 'synthetic' builds "
             "positions through the real producer.",
    )
    parser.add_argument(
        "--corpus-path", metavar="PATH",
        help="The replay ring to sample when --source corpus. Required for that source.",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, metavar="N",
        help="Measurements per sweep point; the MEDIAN is reported, never a single shot. "
             "(5 when unset.)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.85, metavar="M",
        help="Headroom multiplier applied to the budget. 0.85 is INHERITED with its "
             "provenance from the training-side sizing pass, where it covered a >1 GiB "
             "fragmentation swing and the eval child; the eval child OOM'd anyway, so treat "
             "it as a floor. Whatever is given is stamped into the report. (0.85 when unset.) "
             "R327(c) PINS this value AS PROCEDURE: the R326 partition closes at exactly "
             "k = predicted_peak/budget = 0.849998, so k IS this knob rather than a search "
             "outcome, and the largest margin that partition affords is 0.8568 -- 0.86 would "
             "REFUSE the same partition on the same card. Moving the pin to the measured edge "
             "after observing it would be criterion movement, so it does not move; the "
             "0.86-refuses fact is recorded as the partition being tight and honest.",
    )
    parser.add_argument(
        "--shapes-only", action="store_true",
        help="Run only the device-free half: build the sweep, report each batch's (N, E) and "
             "the operating ratio, and NULL every measured field. Succeeds on a host with no "
             "CUDA. Emits no cap and no mint line.",
    )
    parser.add_argument(
        "--no-recommend", action="store_true",
        help="Measure and fit, but emit no pair and no mint line. This is the mode for a "
             "sitting that needs the fit BEFORE the budget is known (an allocator-posture "
             "A/B), which is otherwise circular.",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Where to write the JSON report. Written to stdout when unset; the tool never "
             "picks a filename of its own.",
    )
    return parser


def _emit(report: dict[str, Any], out: str | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    if out is None:
        print(text)
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"{_TOOL}: report written to {path}")


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recommending = not (args.shapes_only or args.no_recommend)
    # Budget check before device check: both refuse with exit 2, so a run missing its budget on
    # a CPU host is still told which it is missing.
    if recommending and args.budget_bytes is None:
        raise CalibrationRefusal(
            f"{_TOOL}: --budget-bytes is required for a recommending run and has no fallback "
            "value (R1 applied to a tool: every cap emitted is a function of it, so a budget "
            "nobody minted is a cap nobody measured). Supply it, or pass --no-recommend for "
            "a fit without a pair, or --shapes-only for the device-free half."
        )
    if args.repeats < 1:
        raise CalibrationRefusal(f"{_TOOL}: --repeats must be at least 1; got {args.repeats}")
    if args.source == "corpus" and args.corpus_path is None:
        raise CalibrationRefusal(
            f"{_TOOL}: --source corpus needs --corpus-path; there is no default ring, and "
            "silently falling back to synthetic would fit the cap to a distribution the run "
            "does not have."
        )

    from mantis.config.loader import load_config
    from mantis.encoding import lookup

    config_path = Path(args.config)
    config = load_config(config_path)
    if config.identity.representation != "graph":
        raise CalibrationRefusal(
            f"{_TOOL}: {config_path} declares representation "
            f"{config.identity.representation!r}. There is no fused graph forward on the "
            f"dense route and therefore nothing to bound — the grid batch is a fixed-shape "
            "tensor already bounded by inference.inference_batch_size."
        )
    spec = lookup(config.identity.encoding)
    counts = tuple(n for n in _GRAPH_COUNTS if n <= int(config.inference.inference_batch_size))
    if not counts:
        counts = (int(config.inference.inference_batch_size),)
    max_moves = int(config.selfplay.max_game_moves)
    corpus = (
        _load_corpus(config.identity.encoding, max_moves, args.corpus_path)
        if args.source == "corpus" else None
    )

    if args.shapes_only:
        points = _sweep(config.identity.encoding, max_moves, counts, corpus)
        report = {
            "calibrated": False,
            "source": args.source,
            "corpus_path": args.corpus_path,
            "why_uncalibrated": (
                "--shapes-only: the device-free half ran and nothing was measured. "
                "`peak_bytes` and `fit` are null because there is NO PRODUCER on this path — "
                "the unproduced-field convention, not a zero and not an extrapolation."
            ),
            "sweep": [p.as_dict() for p in points],
            "peak_bytes": None,
            "fit": None,
            # DERIVABLE DEVICE-FREE, so reported rather than nulled: these are SHAPES.
            "operating_ratio_e_over_n": float(np.median(
                [p.n_edges / max(p.n_nodes, 1) for p in points]
            )),
            "largest_single_graph": {
                "nodes": max(p.max_graph_nodes for p in points),
                "edges": max(p.max_graph_edges for p in points),
            },
            "fragmentation_ratio": None,
            "margin_requested": None,
            "margin_achieved": None,
            "recommended": None,
            "mint_line": None,
            "provenance": _provenance(config_path, spec, None),
        }
        _emit(report, args.out)
        return 0

    import torch

    if not torch.cuda.is_available():
        raise CalibrationRefusal(
            f"{_TOOL}: torch.cuda.is_available() is False. This calibration measures a CUDA "
            "caching-allocator peak; on a host without one there is nothing to measure, so "
            "NO cap is emitted and none is estimated. A number produced here would carry "
            "this tool's authority without its mechanism (R69). Re-run on the box, or pass "
            "--shapes-only for the device-free half."
        )

    from mantis.model import arch_from_spec_and_config, build_net

    arch = arch_from_spec_and_config(spec, config.model_dump())
    device = torch.device("cuda")
    net = build_net(arch).to(device)
    net.eval()

    points = _sweep(config.identity.encoding, max_moves, counts, corpus)
    by_label = {label: (stones, spread) for label, stones, spread in _regimes(corpus)}
    measured: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    for point in points:
        try:
            measured.append(_measure_point(
                net, spec, device, config.identity.encoding, max_moves, point,
                *by_label[point.label], args.repeats, corpus,
                # THREADED from the config, never named here, so this call site is not a second
                # dtype authority the day a grid calibration exists.
            ))
        except torch.cuda.OutOfMemoryError as exc:
            # RECORDED, NEVER RETRIED, and never estimated: dying here would cost the sitting,
            # and a smaller substitute would report a shape this point never ran. The point is
            # excluded from the fit BY NAME.
            torch.cuda.empty_cache()
            unmeasured.append({
                **point.as_dict(), "peak_bytes_median": None,
                "not_measured": "cuda_out_of_memory", "error": str(exc)[:300],
            })
            print(
                f"{_TOOL}: OOM at {point.label} n_graphs={point.n_graphs} "
                f"(N={point.n_nodes}, E={point.n_edges}) — recorded as unmeasured, excluded "
                "from the fit, NOT retried.",
                file=sys.stderr,
            )
    if len(measured) < 3:
        raise CalibrationRefusal(
            f"{_TOOL}: only {len(measured)} sweep point(s) could be measured; a three-term "
            "fit needs at least three. No cap is emitted — a pair solved from an "
            "underdetermined fit is a guess wearing a measurement's name."
        )
    fit = _fit(measured)
    recommendation = (
        _recommend(fit, int(args.budget_bytes), float(args.margin), measured,
                   unmeasured=unmeasured)
        if recommending else None
    )
    mint_line = None
    if recommendation is not None:
        # Read back through the block's own member NAMES, so this module is not a second
        # constant-string reader of the cap block.
        edges, nodes = (recommendation[member] for member in _MEMBER_NAMES)
        mint_line = (
            "uv run python tools/mint_config.py --template <template> --out "
            f"{config_path} --force <existing --set deltas...> --set "
            f"{_KEY}.{_MEMBER_NAMES[0]}={edges} --set {_KEY}.{_MEMBER_NAMES[1]}={nodes}"
        )
    # The MEASURED allocator posture as one number: `reserved_peak / allocated_peak` over the
    # sweep, which the budget arithmetic divides by. Reported, never assumed.
    ratios = [m["reserved_bytes_peak_median"] / max(m["allocated_bytes_peak_median"], 1)
              for m in measured]
    report = {
        "calibrated": True,
        "source": args.source,
        "corpus_path": args.corpus_path,
        "sweep": measured,
        "unmeasured_sweep_points": unmeasured,
        "peak_bytes": {m["label"] + f"@{m['n_graphs']}": m["peak_bytes_median"]
                       for m in measured},
        "fit": fit,
        "operating_ratio_e_over_n": fit["operating_edges_per_node"],
        "largest_single_graph": {
            # Over every point the sweep BUILT: an OOM'd point is a graph this sweep saw.
            "nodes": max(m["largest_graph_nodes"] for m in measured + unmeasured),
            "edges": max(m["largest_graph_edges"] for m in measured + unmeasured),
        },
        "fragmentation_ratio": float(np.median(ratios)),
        # The INPUT under its own name and the MEASUREMENT under its own: this field was
        # `float(args.margin)`, the argument echoed back as though measured.
        "margin_requested": float(args.margin) if recommending else None,
        "margin_achieved": (recommendation["margin_achieved"]
                            if recommendation is not None else None),
        "recommended": recommendation,
        "mint_line": mint_line,
        "provenance": _provenance(config_path, spec, arch),
    }
    _emit(report, args.out)
    if mint_line is not None:
        print(mint_line)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except CalibrationRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
