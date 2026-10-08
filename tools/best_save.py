"""The best-save rule over one run's saves read on rung 16 and S. usage: best_save.py DIR... --run-id R --arm A"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_TOOLS = Path(__file__).resolve().parent


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / rel)
    if spec is None or spec.loader is None:
        raise ImportError(f"no module at {_TOOLS / rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_follower = _load("strix_follower_for_best_save", "strix_follower.py")
# The dash's reader is the one sidecar reading: Six's forfeits come out of our wins there.
_sidecars = _load("dash_sidecars_for_best_save", "dash/readers/sidecars.py")
RUNG = next(unit for unit, nodes in _follower.LADDER_UNITS.items() if nodes == 16)
S = _follower.EQUAL_WORK_ARENA


def _series(cells: list[Any], run_id: str, unit: str, arm: str | None, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    """The run's whole-book cells on `unit` by checkpoint stem, and why each other cell of it does not count."""
    name = unit if arm is None else f"{unit}.{arm}"
    own = [c for c in cells if c.run_id == run_id and c.name == name]
    instruments = {c.unit for c in own}
    if len(instruments) > 1:
        raise ValueError(f"{name} spans {len(instruments)} instruments for {run_id} (a pin, book or device changed)")
    whole, full, notes = _follower.whole_book(unit), {}, {}
    for c in own:
        if whole is not None and c.n + c.forfeits != whole:
            notes[c.stem] = f"{label} read by a {c.n + c.forfeits}-game screen, not the whole book"
        elif c.lo is None or c.hi is None:
            notes[c.stem] = f"{label} read with no interval"
        else:
            full[c.stem] = c
    return full, notes


def _reading(c: Any) -> dict[str, Any]:
    return {"wr": c.wr, "ci": [c.lo, c.hi], "games": c.n}


def best_save(dirs: list[Path], *, run_id: str, arm: str | None) -> dict[str, Any]:
    """First on rung 16 and S, else the first on rung 16 whose S interval holds the best S. Raises: ValueError, OSError."""
    cells, skipped = _sidecars.load(dirs)
    (rung, rung_notes), (s, s_notes) = _series(cells, run_id, RUNG, arm, "rung 16"), _series(cells, run_id, S, arm, "S")
    both = sorted(set(rung) & set(s), key=lambda k: (-rung[k].wr, -s[k].wr, k))
    left_out = sorted([f"{k}: {s_notes.get(k, 'no S reading')}" for k in set(rung) - set(s)]
                      + [f"{k}: {rung_notes.get(k, 'no rung-16 reading')}" for k in set(s) - set(rung)])
    if not both:
        raise ValueError(f"no save of {run_id} carries both a rung-16 and an S reading")
    s_best = max(s[k].wr for k in both)
    if s[both[0]].wr == s_best:
        pick, rule = both[0], "first on both"
    else:
        held = [k for k in both if s[k].lo <= s_best <= s[k].hi]
        if not held:
            raise ValueError(f"no save's S interval holds the best S reading {s_best:.4f}")
        pick, rule = held[0], "first on rung 16 with S within its interval"
    rows = [{"checkpoint": k, "step": rung[k].step, "rung16": _reading(rung[k]), "s": _reading(s[k])}
            for k in sorted(both, key=lambda k: rung[k].step)]
    return {"pick": pick, "rule": rule, "s_best": s_best, "rows": rows, "left_out": left_out, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    """Print the pick as JSON. Raises: ValueError (no save read on both; a unit over two instruments), OSError."""
    ap = argparse.ArgumentParser(description="Pick a run's best save by rung 16 and S.")
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--run-id", required=True, help="the run whose saves are ranked (its sidecars' run_id)")
    ap.add_argument("--arm", required=True, help="the A/B arm the sidecars name ('' for none)")
    args = ap.parse_args(argv)
    print(json.dumps(best_save(args.dirs, run_id=args.run_id, arm=args.arm or None), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
