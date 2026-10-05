"""One served run: its directory's events, game shards, heartbeat and config, its monitor records and the cell sidecars, read into a snapshot."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mantis.util.yaml_io import DuplicateKeyError, parse_config_yaml

from . import liveness, saves, sidecars
from .events import EmptyRunRecord, EventTail, record_run_ids
from .horizon import Horizon, HorizonReducer
from .series import Snapshot
from .shards import ShardIndex


@dataclass(frozen=True)
class RunSnapshot:
    """Everything a Run view needs about one run as of one poll; nothing in it changes after the poll that built it."""

    label: str
    run_id: str
    events: Snapshot
    horizon: Horizon | None
    records: saves.Records | None
    six: sidecars.Strength
    strix: sidecars.Strength
    cells_skipped: tuple[str, ...]
    parent_stem: str | None
    beat: liveness.Liveness
    games_indexed: int


def parent_stem(run_dir: Path) -> tuple[str | None, str]:
    """The stem of `identity.warm_start.checkpoint` in `resolved_config.yaml`, and a note naming why there is none."""
    path = run_dir / "resolved_config.yaml"
    try:
        raw: Any = parse_config_yaml(path)
    except (OSError, UnicodeDecodeError, DuplicateKeyError, yaml.YAMLError) as exc:
        return None, f"{path.name} unreadable ({type(exc).__name__})"
    identity = raw.get("identity") if isinstance(raw, dict) else None
    warm = identity.get("warm_start") if isinstance(identity, dict) else None
    ckpt = warm.get("checkpoint") if isinstance(warm, dict) else None
    if not ckpt:
        return None, "no warm start: the run has no parent"
    return Path(str(ckpt)).name.split(".ckpt")[0], "warm_start.checkpoint"


class RunRecord:
    """Binds one run's inputs; `poll()` reads what is new and returns a fresh `RunSnapshot`; the shard index is shared under `lock`."""

    def __init__(self, label: str, run_dir: Path, records_dir: Path | None = None,
                 cells: tuple[Path, ...] = ()) -> None:
        self.label, self.run_dir, self.records_dir, self.cells = label, Path(run_dir), records_dir, cells
        ids = record_run_ids(self.run_dir / "logs")
        if not ids:
            raise EmptyRunRecord(f"{self.run_dir / 'logs'} holds no events_<run>_seg*.jsonl")
        self.run_id = ids[-1] if len(ids) == 1 else self._config_run_id(ids)
        self.tail = EventTail(self.run_dir / "logs", self.run_id)
        self.horizon = HorizonReducer()
        self.games = ShardIndex(self.run_dir / "logs" / "games", self.run_id, observers=(self.horizon,))
        self.lock = threading.Lock()
        self.parent_stem, self.parent_note = parent_stem(self.run_dir)

    def _config_run_id(self, ids: list[str]) -> str:
        try:
            raw = parse_config_yaml(self.run_dir / "resolved_config.yaml")
        except (OSError, UnicodeDecodeError, DuplicateKeyError, yaml.YAMLError):
            raw = None
        named = raw.get("run_id") if isinstance(raw, dict) else None
        if named not in ids:
            raise EmptyRunRecord(f"{self.run_dir / 'logs'} holds several runs {ids} and the config names none of them")
        return str(named)

    def poll(self) -> RunSnapshot:
        """Read every new row and file and return the run's snapshot. Raises: EmptyRunRecord, OSError."""
        events = self.tail.poll()
        with self.lock:
            self.games.poll()
            horizon, indexed = self.horizon.read(), self.games.total
        records = saves.load(self.records_dir) if self.records_dir is not None else None
        cells, skipped = sidecars.load(self.cells)
        return RunSnapshot(
            label=self.label, run_id=self.run_id, events=events, horizon=horizon, records=records,
            six=sidecars.strength(cells, "six", self.run_id, self.parent_stem),
            strix=sidecars.strength(cells, "strix", self.run_id, self.parent_stem),
            cells_skipped=tuple(skipped), parent_stem=self.parent_stem,
            beat=liveness.read(self.run_dir / "logs", self.run_id), games_indexed=indexed)
