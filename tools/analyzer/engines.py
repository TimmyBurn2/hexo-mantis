"""Engines: stamped-checkpoint discovery, the deploy-matched composition, the cached raw tree and the search."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mantis._engine import Board, MCTSTree
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
from mantis.config.resolve.search import MissingSearchKindError, resolve_deploy_search_kind
from mantis.encoding import lookup
from mantis.eval.worker import _graph_expand_fn, build_candidate_player
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.train.bundle_receipts import CHECKPOINT_NAME_RE, stamped_checkpoints
from mantis.train.checkpoints import load_checkpoint

#: The Gumbel draw's seed for every analyzer search; a FRESH player per search is what makes an answer repeat.
ANALYZER_GUMBEL_SEED = 0
MANTIS, SNAPSHOT_GAP, STRIX = "mantis", "snapshot_gap", "strix"
_SNAPSHOT = "best_model.pt"
ChildInfo = tuple[tuple[int, int], int, float, int, float]


class EngineLoadError(RuntimeError):
    """An engine that cannot be built; the message names the stamp key or the failure."""


@dataclass(frozen=True)
class EngineInfo:
    """One `/engines` row: `kind` mantis (loadable) | snapshot_gap (stated, never loaded) | strix."""

    id: str
    kind: str
    path: str
    run_id: str | None = None
    step: int | None = None
    sha8: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawRead:
    """The net's own read of a root: its value and the decoded priors over the legal children (visits all 0)."""

    value: float
    children: list[ChildInfo]
    ms: float


@dataclass(frozen=True)
class Search:
    """The deploy head's answer: its root value (quiescence included), its move, the children, the counters."""

    root_value: float
    argmax: tuple[int, int]
    children: list[ChildInfo]
    root_visits: int
    quiescence_fires: int
    ms: float


def _snapshot_note(path: Path) -> str:
    sidecar = path.with_name(path.name + ".provenance.json")
    if sidecar.is_file():
        try:
            prov = json.loads(sidecar.read_text(encoding="utf-8"))
            return (f"snapshot format, no stamp; its provenance names step {prov['step']} of "
                    f"{prov['run_id']} — pick that stamped step")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return "snapshot format, no stamp — pick a stamped step"


def discover(dirs: list[Path]) -> list[EngineInfo]:
    """Stamped `.ckpt` rows (oldest step first) from each dir and its `*/checkpoints` subdirs, plus snapshots as stated gaps."""
    rows: list[EngineInfo] = []
    for top in dirs:
        top = Path(top)
        for d in [top, *sorted(p for p in top.glob("*/checkpoints") if p.is_dir())]:
            for path in stamped_checkpoints(d):
                m = CHECKPOINT_NAME_RE.match(path.name)
                assert m is not None
                rows.append(EngineInfo(id=path.stem, kind=MANTIS, path=str(path), run_id=m["run_id"],
                                       step=int(m["step"]), sha8=m["sha8"]))
            snap = d / _SNAPSHOT
            if snap.is_file():
                rows.append(EngineInfo(id=f"{d.name if d is top else d.parent.name}/{_SNAPSHOT}",
                                       kind=SNAPSHOT_GAP, path=str(snap), note=_snapshot_note(snap)))
    return rows


class MantisEngine:
    """A stamped checkpoint as an engine: the run's decode, a cached quiescence-off tree for the net, a fresh head per search."""

    def __init__(self, info: EngineInfo, *, device: str, threads: int | None) -> None:
        ck = load_checkpoint(Path(info.path))
        cfg: dict[str, Any] = ck.config
        try:
            sp = cfg["selfplay"]
            self._hparams = dict(leaf_batch_size=int(sp["leaf_batch_size"]), c_visit=float(sp["c_visit"]),
                                 c_scale=float(sp["c_scale"]), q_rescale=bool(sp["q_rescale"]),
                                 gumbel_m=int(sp["gumbel_m"]))
            self.deploy_sims = int(cfg["eval"]["gate"]["deploy_sims"])
        except (KeyError, TypeError) as exc:
            raise EngineLoadError(f"{info.id}: the stamp's config lacks {exc.args[0]!r}; nothing is defaulted here") from None
        try:
            self.search_kind = resolve_deploy_search_kind(cfg)
        except MissingSearchKindError as exc:
            raise EngineLoadError(f"{info.id}: {exc}") from None
        if ck.metadata.arch is None:
            raise EngineLoadError(f"{info.id}: the stamp resolves no arch, so the net cannot be rebuilt")
        self.info = info
        self.encoding = str(ck.metadata.encoding_name)
        self.spec = lookup(self.encoding)
        self.radius = int(self.spec.legal_move_radius)
        net = build_net(ck.metadata.arch)
        net.load_state_dict(ck.model_state)
        net.eval()
        if threads:
            torch.set_num_threads(int(threads))
        graph = self.spec.representation == "graph"
        self.engine = LocalInferenceEngine(
            net, torch.device(device), encoding_spec=self.spec,
            fused_graph_caps=resolve_fused_graph_caps(cfg) if graph else None,
            inference_batching=resolve_inference_batching(cfg) if graph else None,
            max_in_flight=self._hparams["leaf_batch_size"],
            leaf_build_threads=resolve_leaf_build_threads(cfg) if graph else 1)
        self._expand: Callable[[MCTSTree, list[Board]], None] = (
            _graph_expand_fn(self.engine, self.spec) if graph else self._grid_expand)
        self._raw_tree = MCTSTree(quiescence_enabled=False)
        self._raw_tree.configure_search(self.search_kind, self._hparams["c_visit"], self._hparams["c_scale"],
                                        self._hparams["q_rescale"])
        self.card: dict[str, Any] = {
            "id": info.id, "run_id": ck.metadata.run_id, "step": int(ck.metadata.step), "sha8": info.sha8,
            "encoding": self.encoding, "radius": self.radius, "search_kind": self.search_kind,
            "deploy_sims": self.deploy_sims, "params": sum(p.numel() for p in net.parameters()),
            "net_hash": net_param_hash(net), "device": device, "threads": torch.get_num_threads(),
            "seed": ANALYZER_GUMBEL_SEED,
            "quiescence": "off on the net row; the head row is the deploy head's own, its override counted",
        }

    def _grid_expand(self, tree: MCTSTree, leaves: list[Board]) -> None:
        policies, values = self.engine.infer_batch(leaves)
        tree.expand_and_backup(policies, values)

    def raw_read(self, board: Board) -> RawRead:
        """The net on this root through the cached quiescence-off tree; raises RuntimeError when the root yields no leaf."""
        t0 = time.perf_counter()
        self._raw_tree.new_game(board)
        leaves = self._raw_tree.select_leaves(1)
        if not leaves:
            raise RuntimeError("raw read: the root yielded no leaf (a terminal position is refused before this)")
        self._expand(self._raw_tree, leaves)
        return RawRead(value=float(self._raw_tree.root_value()), children=self._raw_tree.get_root_children_info(),
                       ms=(time.perf_counter() - t0) * 1000.0)

    def search(self, board: Board, sims: int) -> Search:
        """A fresh deploy head at `sims`; raises ValueError (the head's own) when the root has no children."""
        player = build_candidate_player(self.engine, int(sims), spec=self.spec, search_kind=self.search_kind,
                                        gumbel_seed=ANALYZER_GUMBEL_SEED, **self._hparams)
        player.new_game()
        t0 = time.perf_counter()
        move = player.select_move(board)
        ms = (time.perf_counter() - t0) * 1000.0
        assert player.last_root is not None
        root_value, children = player.last_root
        # The head exposes no tree accessor; `_tree` is read for its two counters only (pinned by test).
        tree = player._tree  # noqa: SLF001
        assert tree is not None
        return Search(root_value=float(root_value), argmax=(int(move[0]), int(move[1])), children=children,
                      root_visits=int(tree.root_visits()), quiescence_fires=int(tree.quiescence_fire_count), ms=ms)

    def close(self) -> None:
        self.engine.close()


__all__ = ["ANALYZER_GUMBEL_SEED", "MANTIS", "SNAPSHOT_GAP", "STRIX", "ChildInfo", "EngineInfo", "EngineLoadError",
           "MantisEngine", "RawRead", "Search", "discover"]
