"""`python tools/distill.py <command>`: corpus, validate, label, rings, train, kl, read — each writes its record beside its output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mantis.encoding import lookup
from mantis.util.hashing import sha256_file

from .corpus import SRC_RING, build_corpus
from .heldout import read_baseline, read_kl
from .label import label, open_labels
from .read import verdict
from .rings import (
    TEACHER_MIN_PLY,
    choose_slots,
    compute_targets,
    empty_board_mass,
    legal_check,
    teacher_value,
    write_control_ring,
    write_provenance,
    write_teacher_ring,
)
from .teacher import Teacher
from .train import ArmSpec, Recipe, train_arm
from .validate import validate

ENCODING = "gnn_axis_r8"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")


def load_corpus(directory: Path) -> dict[str, np.ndarray]:
    """A saved corpus's arrays; Raises: OSError — the corpus file is absent."""
    with np.load(directory / "corpus.npz") as data:
        return {k: data[k] for k in data.files}


def cmd_corpus(a: argparse.Namespace) -> int:
    a.out.mkdir(parents=True, exist_ok=False)
    games = sorted(a.games.glob(f"games_{a.run_id}_seg*.jsonl"))
    out, meta = build_corpus(a.rings, games, run_id=a.run_id, heldout_rows=a.heldout_rows, split_seed=a.split_seed,
                             openings_n=a.openings, openings_seed=a.openings_seed, openings_plies=a.openings_plies,
                             step_margin=a.step_margin)
    arrays: dict[str, Any] = dict(out)
    np.savez(a.out / "corpus.npz", **arrays)
    meta["summary"]["rings_sha256"] = {p.name: sha256_file(p) for p in a.rings}
    meta["summary"]["args"] = {k: str(v) for k, v in vars(a).items() if k != "func"}
    _write_json(a.out / "corpus.json", meta)
    print(json.dumps(meta["summary"], indent=1, default=str))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    corpus = load_corpus(a.corpus)
    res = validate(corpus, Teacher(device=a.device), Teacher(device="cpu"), n=a.n, seed=a.seed,
                   radius=lookup(ENCODING).legal_move_radius)
    _write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1, default=str))
    return 0 if res["passed"] else 2


def cmd_label(a: argparse.Namespace) -> int:
    corpus = load_corpus(a.corpus)
    rec = label(corpus, Teacher(device=a.device), a.corpus / "labels", radius=lookup(ENCODING).legal_move_radius,
                batch=a.batch, workers=a.workers)
    _write_json(a.corpus / "labels" / "labels.json", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


def cmd_rings(a: argparse.Namespace) -> int:
    corpus, labels = load_corpus(a.corpus), open_labels(a.corpus / "labels")
    teacher = json.loads((a.corpus / "labels" / "labels.json").read_text(encoding="utf-8"))["teacher"]
    radius = lookup(ENCODING).legal_move_radius
    out = a.corpus / a.out_name
    out.mkdir(exist_ok=False)
    held = corpus["heldout"]
    labelled = corpus["k"] >= TEACHER_MIN_PLY
    train_rows, held_rows = np.nonzero(~held & labelled)[0], np.nonzero(held & labelled)[0]
    tag = [teacher["lineage"]]
    rec: dict[str, Any] = {"encoding": ENCODING, "radius": radius, "teacher": teacher, "teacher_min_ply": TEACHER_MIN_PLY,
                           "empty_board_rows_left_out": int((~labelled).sum())}
    t_train = compute_targets(corpus, labels, train_rows, radius=radius)
    slots, quantiles = choose_slots(t_train["mass"])
    rec.update(slots=slots, slot_quantiles=quantiles, support_mean=float(t_train["n_support"].mean()),
               support_max=int(t_train["n_support"].max()))
    if "teacher" in a.only:
        t_held = compute_targets(corpus, labels, held_rows, radius=radius)
        for name, rows, t in (("teacher_train", train_rows, t_train), ("teacher_heldout", held_rows, t_held)):
            rec[name] = write_teacher_ring(out / f"{name}.ring.bin", corpus, labels, rows, t, teacher_value(labels, rows),
                                           slots, encoding=ENCODING)
            write_provenance(out / f"{name}.ring.bin", kind=name, lineage=tag, record=rec[name])
        ring_held = corpus["source"][held_rows] == SRC_RING
        held_ring = held_rows[ring_held]
        rec["teacher_heldout_z"] = write_teacher_ring(
            out / "teacher_heldout_z.ring.bin", corpus, labels, held_ring, {k: v[ring_held] for k, v in t_held.items()},
            corpus["outcome"][held_ring], slots, encoding=ENCODING, value_valid=corpus["value_valid"][held_ring].astype(bool))
        write_provenance(out / "teacher_heldout_z.ring.bin", kind="teacher_heldout_z", lineage=tag,
                         record=rec["teacher_heldout_z"])
        v_t, z = teacher_value(labels, held_ring), corpus["outcome"][held_ring]
        decided = corpus["value_valid"][held_ring].astype(bool) & (z != 0)
        rec["teacher_value_v_z"] = {"rows": int(decided.sum()), "pearson": float(np.corrcoef(v_t[decided], z[decided])[0, 1]),
                                    "sign_agreement": float((np.sign(v_t[decided]) == np.sign(z[decided])).mean())}
    del t_train
    if "knownbad" in a.only:
        source = train_rows[np.random.default_rng(a.knownbad_seed).permutation(len(train_rows))]
        t_kb = compute_targets(corpus, labels, train_rows, radius=radius, source_rows=source)
        rec["knownbad_train"] = write_teacher_ring(out / "knownbad_train.ring.bin", corpus, labels, train_rows, t_kb,
                                                   teacher_value(labels, source), slots, encoding=ENCODING)
        rec["knownbad_seed"] = a.knownbad_seed
        write_provenance(out / "knownbad_train.ring.bin", kind="knownbad_train", lineage=tag, record=rec["knownbad_train"])
    if "control" in a.only:
        control_rows = np.nonzero(~held & (corpus["source"] == SRC_RING))[0]
        rec["control_train"] = write_control_ring(out / "control_train.ring.bin", corpus, control_rows, encoding=ENCODING)
        write_provenance(out / "control_train.ring.bin", kind="control_train", lineage=[], record=rec["control_train"])
    sample = np.sort(np.random.default_rng(a.check_seed).choice(train_rows, size=min(a.check_rows, len(train_rows)),
                                                                 replace=False))
    rec["legal_check"] = legal_check(corpus, labels, sample, radius=radius, encoding=ENCODING)
    rec["empty_board"] = empty_board_mass(labels, np.nonzero(corpus["k"] == 0)[0], radius)
    rec["sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("*.ring.bin"))}
    _write_json(out / "rings.json", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    arm = ArmSpec(run_id=a.run_id, hidden=a.hidden, layers=a.layers, ring=a.ring, warm_start=a.warm_start,
                  value_mask_p=a.value_mask)
    recipe = Recipe(base_config=a.base_config, steps=a.steps, save_at=tuple(a.save_at), lr=a.lr, eta_min=a.eta_min,
                    seed=a.seed)
    rec = train_arm(arm, recipe, a.out, device=a.device, heldout_ring=a.heldout_ring, heldout_every=a.heldout_every,
                    heldout_batches=a.heldout_batches, sample_threads=a.sample_threads,
                    caps_override=None if a.caps is None else (a.caps[0], a.caps[1]))
    print(json.dumps(rec, indent=1, default=str))
    return 0


def cmd_kl(a: argparse.Namespace) -> int:
    rec = read_kl(a.ckpts, a.heldout_ring, batches=a.batches, batch_size=a.batch_size, threads=a.threads, seed=a.seed)
    _write_json(a.out, rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


def cmd_read(a: argparse.Namespace) -> int:
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    rec = verdict(spec, a.cells)
    _write_json(a.out, rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


def cmd_baseline(a: argparse.Namespace) -> int:
    rec = read_baseline(a.ckpt, a.ring, batches=a.batches, batch_size=a.batch_size, threads=a.threads, seed=a.seed)
    _write_json(a.out, rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="distill", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("corpus", help="match ring rows to their games, split held-out by game, add arena openings")
    c.add_argument("--rings", type=Path, nargs="+", required=True)
    c.add_argument("--games", type=Path, required=True, help="the run's game-shard directory")
    c.add_argument("--run-id", required=True)
    c.add_argument("--heldout-rows", type=int, default=100_000)
    c.add_argument("--split-seed", type=int, default=20261008)
    c.add_argument("--openings", type=int, default=50_000)
    c.add_argument("--openings-seed", type=int, default=20261009)
    c.add_argument("--openings-plies", type=int, default=5)
    c.add_argument("--step-margin", type=int, default=25_000, help="game steps read below the oldest ring's step")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_corpus)
    v = sub.add_parser("validate", help="V1: our argmax v the engine's `go nodes 1` on screened held-out rows")
    v.add_argument("--corpus", type=Path, required=True)
    v.add_argument("--n", type=int, default=1000)
    v.add_argument("--seed", type=int, default=20261011)
    v.add_argument("--device", default="cuda")
    v.add_argument("--out", type=Path, required=True)
    v.set_defaults(func=cmd_validate)
    lb = sub.add_parser("label", help="the teacher's raw heads for every corpus row")
    lb.add_argument("--corpus", type=Path, required=True)
    lb.add_argument("--device", default="cuda")
    lb.add_argument("--batch", type=int, default=2048)
    lb.add_argument("--workers", type=int, default=8)
    lb.set_defaults(func=cmd_label)
    rg = sub.add_parser("rings", help="the teacher, held-out, known-bad and control rings")
    rg.add_argument("--corpus", type=Path, required=True)
    rg.add_argument("--knownbad-seed", type=int, default=20261010)
    rg.add_argument("--check-seed", type=int, default=20261012)
    rg.add_argument("--check-rows", type=int, default=20_000)
    rg.add_argument("--only", nargs="+", choices=("teacher", "knownbad", "control"),
                    default=["teacher", "knownbad", "control"])
    rg.add_argument("--out-name", default="rings", help="the rings directory under the corpus")
    rg.set_defaults(func=cmd_rings)
    t = sub.add_parser("train", help="one arm through the production trainer on a frozen ring")
    t.add_argument("--ring", type=Path, required=True)
    t.add_argument("--run-id", required=True)
    t.add_argument("--hidden", type=int, required=True)
    t.add_argument("--layers", type=int, required=True)
    t.add_argument("--warm-start", type=Path, default=None, help="a stamped checkpoint whose deploy net seeds the arm")
    t.add_argument("--value-mask", type=float, required=True, help="train.value_mask_redraw_p (0 is off)")
    t.add_argument("--base-config", type=Path, required=True)
    t.add_argument("--steps", type=int, default=15_000)
    t.add_argument("--save-at", type=int, nargs="+", default=[10_000, 15_000])
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--eta-min", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=20261008)
    t.add_argument("--heldout-ring", type=Path, default=None)
    t.add_argument("--heldout-every", type=int, default=1000)
    t.add_argument("--heldout-batches", type=int, default=8)
    t.add_argument("--sample-threads", type=int, default=None)
    t.add_argument("--caps", type=int, nargs=2, default=None, metavar=("MAX_EDGES", "MAX_NODES"),
                   help="micro-batch caps (the split is exact; memory only)")
    t.add_argument("--device", default="cuda")
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(func=cmd_train)
    k = sub.add_parser("kl", help="held-out KL(teacher || net) and value reads, every net on the same samples")
    k.add_argument("--ckpts", type=Path, nargs="+", required=True)
    k.add_argument("--heldout-ring", type=Path, required=True)
    k.add_argument("--batches", type=int, default=80)
    k.add_argument("--batch-size", type=int, default=256)
    k.add_argument("--threads", type=int, default=8)
    k.add_argument("--seed", type=int, default=20261013)
    k.add_argument("--out", type=Path, required=True)
    k.set_defaults(func=cmd_kl)
    rd = sub.add_parser("read", help="two-save panels against the reference, the controls and the outcome")
    rd.add_argument("--spec", type=Path, required=True, help="JSON: reference, control, knownbad, arms {name: class, saves}")
    rd.add_argument("--cells", type=Path, nargs="+", required=True, help="directories holding the cell sidecars")
    rd.add_argument("--out", type=Path, required=True)
    rd.set_defaults(func=cmd_read)
    b = sub.add_parser("baseline", help="a net's top-1 agreement with the teacher and its value's correlation with z")
    b.add_argument("--ckpt", type=Path, required=True)
    b.add_argument("--ring", type=Path, required=True, help="the teacher-policy ring whose z is the game's")
    b.add_argument("--batches", type=int, default=80)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--threads", type=int, default=8)
    b.add_argument("--seed", type=int, default=20261013)
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=cmd_baseline)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Dispatch one subcommand; its exit status is the command's."""
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
