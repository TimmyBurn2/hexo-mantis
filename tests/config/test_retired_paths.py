"""The retired-path authority names only paths the schema no longer has, and splitting them off never mutates the record."""
from __future__ import annotations

from mantis.config.retired import RETIRED_PATHS, split_retired
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
