"""The config census (R367(a)): production is every config on disk minus the exempt rows, taken at point of use."""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.config import census

_REPO = Path(__file__).resolve().parents[2]


def _tree(tmp_path: Path, names: tuple[str, ...]) -> Path:
    root = tmp_path / "tree"
    for name in names:
        target = root / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("run_id: x\n", encoding="utf-8")
    return root


def test_production_is_discovery_minus_the_exempt_rows(tmp_path: Path) -> None:
    exempt = tuple(Path(rel).relative_to("configs").as_posix() for rel in census.exempt_config_paths())
    root = _tree(tmp_path, (*exempt, "b.yaml", "a.yaml", "nested/c.yml"))
    assert [p.relative_to(root / "configs").as_posix() for p in census.production_configs(root)] == ["a.yaml", "b.yaml", "nested/c.yml"]


def test_a_config_that_lands_on_disk_joins_the_census_with_no_edit(tmp_path: Path) -> None:
    exempt = tuple(Path(rel).relative_to("configs").as_posix() for rel in census.exempt_config_paths())
    root = _tree(tmp_path, (*exempt, "a.yaml"))
    before = census.production_configs(root)
    (root / "configs" / "minted_later.yaml").write_text("run_id: y\n", encoding="utf-8")
    assert census.production_configs(root) == (*before, root / "configs" / "minted_later.yaml")


def test_a_stale_exempt_row_refuses_by_name(tmp_path: Path) -> None:
    root = _tree(tmp_path, ("a.yaml",))
    with pytest.raises(census.ConfigCensusError, match="STALE"):
        census.production_configs(root)


def test_an_empty_census_refuses(tmp_path: Path) -> None:
    exempt = tuple(Path(rel).relative_to("configs").as_posix() for rel in census.exempt_config_paths())
    root = _tree(tmp_path, exempt)
    with pytest.raises(census.ConfigCensusError, match="EMPTY"):
        census.production_configs(root)


def test_the_real_tree_has_a_census_and_every_exempt_row_carries_grounds() -> None:
    production = census.production_configs(_REPO)
    assert production and all(p.is_file() for p in production)
    assert census.exempt_config_paths().isdisjoint(p.relative_to(_REPO).as_posix() for p in production)
    assert all(reason.strip() for _rel, reason in census.EXEMPT_CONFIGS)
