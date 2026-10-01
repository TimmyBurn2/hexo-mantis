"""The retired-path authority names only paths the schema no longer has, and splitting them off never mutates the record."""
from __future__ import annotations

from mantis.config.retired import FOLDED_PATHS, RETIRED_NULL_POSTURES, RETIRED_PATHS, split_retired
from mantis.config.schema import RunConfig, leaf_paths


def test_no_retired_path_is_still_a_schema_path() -> None:
    """A retired path the schema still carries (as a leaf or a block prefix) would tolerate a LIVE key's absence as provenance."""
    live = leaf_paths(RunConfig)
    still_live = sorted(p for p in RETIRED_PATHS if any(l == p or l.startswith(f"{p}.") for l in live))
    assert not still_live, f"retired paths the schema still carries: {still_live}"


def test_split_retired_copies_and_leaves_the_record_untouched() -> None:
    """The split returns the removed values and a copy; the stamp it was handed is never repaired."""
    record = {"train": {"value_target": "pure_outcome_z", "lr": 1e-3}, "search": {"kind": "gumbel"}}
    kept, removed = split_retired(record)
    assert kept == {"train": {"lr": 1e-3}}
    assert removed == {"train.value_target": "pure_outcome_z", "search": {"kind": "gumbel"}}
    assert record["train"]["value_target"] == "pure_outcome_z" and "search" in record


def test_a_record_predating_a_fold_reads_its_folded_value_and_stays_untouched() -> None:
    """PLANTED BREAK: drop `FOLDED_PATHS` and a pre-fold stamp's null horizon and zero sims fail the schema."""
    record = {"train": {"total_steps": 1_000_000, "scheduler_t_max": None},
              "selfplay": {"mcts": {"n_simulations": 2}, "playout_cap": {"n_sims_full": 0}}}
    kept, removed = split_retired(record)
    assert kept == {"train": {"scheduler_t_max": 1_000_000}, "selfplay": {"mcts": {}, "playout_cap": {"n_sims_full": 2}}}
    assert removed == {"train.total_steps": 1_000_000, "selfplay.mcts.n_simulations": 2}
    assert record["train"]["scheduler_t_max"] is None and record["selfplay"]["playout_cap"]["n_sims_full"] == 0


def test_a_fold_never_overrides_a_value_the_record_set() -> None:
    """An armed config keeps its own full arm and horizon; the folded source only fills an unset target."""
    record = {"train": {"total_steps": 1_000_000, "scheduler_t_max": 108_000},
              "selfplay": {"mcts": {"n_simulations": 320}, "playout_cap": {"n_sims_full": 320}}}
    kept, _ = split_retired(record)
    assert kept["train"]["scheduler_t_max"] == 108_000 and kept["selfplay"]["playout_cap"]["n_sims_full"] == 320


def test_every_fold_runs_from_a_retired_path_to_a_live_one() -> None:
    live = set(leaf_paths(RunConfig))
    assert all(src in RETIRED_PATHS and dst in live for src, dst in FOLDED_PATHS.items()), FOLDED_PATHS


def test_a_record_whose_null_named_a_retired_posture_reads_as_predating_the_leaf() -> None:
    """PLANTED BREAK: drop the null-posture rule and a screen/confirm-era stamp's `sequential: null` fails the schema."""
    record = {"eval": {"gate": {"stride": 1, "sequential": None}}}
    kept, removed = split_retired(record)
    assert kept == {"eval": {"gate": {"stride": 1}}} and removed == {"eval.gate.sequential": None}
    armed = {"eval": {"gate": {"sequential": {"mu0": 0.42}}}}
    assert split_retired(armed)[0] == armed, "an armed block is the live rule and stays"
    live = set(leaf_paths(RunConfig))
    assert all(any(leaf.startswith(f"{p}.") for leaf in live) for p in RETIRED_NULL_POSTURES), RETIRED_NULL_POSTURES
