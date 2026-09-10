"""Hold the shared-authority invariant: whatever `load_config` accepts, `discover_configs` sees.

The loader's accept-set is defined by CONTENT, so any name-based discovery filter leaves a config
a run can be launched from and no gate can enumerate. The contrapositive is driven too.
"""
from pathlib import Path

import pytest

from mantis.config.loader import discover_configs, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN5 = REPO_ROOT / "configs" / "run6.yaml"

#: The complement of an enumeration plus the enumeration, each name planted as a byte-for-byte
#: copy of a real config so every one of them really is loadable.
_NAMES = (
    "run6.yaml", "run6.yml",                     # the original two
    "prod/run6.yaml", "prod/nested/run6.yml",    # at depth
    "run6.txt", "run6.conf", "run6.json",        # a plain unknown suffix
    "run6.YAML", "run6.YML", "run6.Yaml",        # a CASE variant of a known one
    "run6", "run6.",                             # no suffix at all
    "run6.yaml.bak", "run6.yml.orig",            # a known suffix that is not final
    ".yaml", ".yml",                             # a dotfile NAMED like a suffix
    ".hidden/run6.yaml",                         # a config under a HIDDEN directory
    "run6.yamlx", "run6.xyaml",                  # a known suffix as a substring
)


def _loadable(path: Path) -> bool:
    """Report whether the loader reads this path. The catch is deliberately total: narrowing it
    to one exception type would let a new refusal mode read as an acceptance."""
    try:
        load_config(path)
    except Exception:  # noqa: BLE001 — total by design; see the docstring above
        return False
    return True


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """Plant every name in `_NAMES` under one `configs/` directory as a real config."""
    configs = tmp_path / "configs"
    body = RUN5.read_text()
    for name in _NAMES:
        target = configs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return configs


def test_the_shared_authority_INVARIANT_holds_over_the_whole_corpus(planted) -> None:
    """Prove every loadable path under the root is discovered, and every skipped path refused.

    Quantified over the tree as it is on disk rather than over `_NAMES`, so incidentally created
    paths are inside the claim; an empty discovery would satisfy neither direction.
    """
    discovered = {path.resolve() for path in discover_configs(planted)}
    on_disk = sorted(planted.rglob("*"))
    assert len(on_disk) > len(_NAMES), "the corpus must include the directories rglob walks"

    loadable = [path for path in on_disk if _loadable(path)]
    assert len(loadable) == len(_NAMES), (
        "every planted name is a byte-for-byte copy of a real config, so all of them must be "
        f"loadable or this row is measuring the fixture rather than the invariant; got "
        f"{len(loadable)} of {len(_NAMES)}"
    )
    for path in loadable:
        assert path.resolve() in discovered, (
            f"{path.relative_to(planted).as_posix()!r}: the loader READS it and discovery does "
            "not enumerate it. That gap IS ADJ-13 F-1 — a file a run can be launched from and "
            "no gate can see. Discovery must not filter by name; the loader decides by content"
        )

    skipped = [path for path in on_disk if path.resolve() not in discovered]
    for path in skipped:
        assert not _loadable(path), (
            f"{path.relative_to(planted).as_posix()!r} was skipped by discovery and the loader "
            "reads it — a file discovery skips must be a file the loader refuses (R75)"
        )
    assert skipped, "the corpus must exercise the skip arm, or the contrapositive is vacuous"


def test_the_ONLY_thing_discovery_skips_is_a_REAL_directory(planted) -> None:
    """Name the one exclusion exactly, so widening it is a visible edit: a real directory is the
    only path type the loader refuses BY TYPE, and `rglob` recurses through it anyway."""
    discovered = {path.resolve() for path in discover_configs(planted)}
    skipped = [path for path in planted.rglob("*") if path.resolve() not in discovered]
    assert {path.relative_to(planted).as_posix() for path in skipped} == {
        "prod", "prod/nested", ".hidden",
    }, f"only real directories may be skipped; got {[str(p) for p in skipped]}"
    for path in skipped:
        assert path.is_dir() and not path.is_symlink()
        with pytest.raises(IsADirectoryError):
            load_config(path)


def test_a_symlinked_DIRECTORY_is_enumerated_because_rglob_will_not_walk_it(tmp_path) -> None:
    """Prove a symlinked directory stays enumerated, because rglob will not walk through it and
    skipping it would hide its whole subtree from both gates."""
    configs = tmp_path / "configs"
    configs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden_cfg.yaml").write_text(RUN5.read_text())
    (configs / "link").symlink_to(outside)

    assert "link/hidden_cfg.yaml" not in {
        path.relative_to(configs).as_posix() for path in configs.rglob("*")
    }, "premise of this row: rglob does not walk a symlinked directory"
    assert (configs / "link").resolve() in {path.resolve() for path in discover_configs(configs)}, (
        "a symlinked directory hides a subtree from discovery, so it must be enumerated itself "
        "and fail loudly — dropping it is a silent hole the size of that subtree"
    )
    with pytest.raises(IsADirectoryError):
        load_config(configs / "link")


def test_the_loader_accepts_a_config_at_ANY_shape(tmp_path) -> None:
    """Prove the loader reads by content, accepting a config at any path shape; the suffix symbols
    are asserted GONE, since a dead exception class invites re-arming the refusal it named."""
    import mantis.config as package
    import mantis.config.loader as loader

    for name in ("run6.txt", "run6.YAML", "run6", "run6.yaml.bak", "run6.yamlx", ".yaml"):
        path = tmp_path / name
        path.write_text(RUN5.read_text())
        assert load_config(path).run_id == "run6", f"{name} must load — R75 declined the refusal"

    for dead in ("CONFIG_SUFFIXES", "ConfigSuffixError", "is_config_path"):
        assert not hasattr(loader, dead), (
            f"{dead} lost its last live consumer when R75 removed the refusal and the mint "
            "guard; a constant nothing reads is exactly what R1 / LAW-08 forbid"
        )
        assert dead not in package.__all__, f"{dead} is still on mantis.config.__all__"


def test_discovery_is_RECURSIVE_and_SORTED_and_does_not_skip_dotfiles(planted) -> None:
    """Prove discovery recurses, sorts, and does not skip dotfiles — `glob.glob` skips them while
    `pathlib.rglob` does not, and order is pinned so two consumers cannot disagree."""
    found = [path.relative_to(planted).as_posix() for path in discover_configs(planted)]
    assert "prod/run6.yaml" in found and "prod/nested/run6.yml" in found, (
        f"a config in a subdirectory must be discovered at any depth; got {found}"
    )
    assert ".yaml" in found and ".hidden/run6.yaml" in found, (
        f"a dotfile config and a config under a hidden directory are both loadable; got {found}"
    )
    assert found == sorted(found), f"discovery must be ordered; got {found}"


def test_a_config_SHAPED_but_BROKEN_path_stays_INSIDE_the_answer_set(tmp_path) -> None:
    """Prove a config-shaped but broken path stays enumerated: the loader's refusal is an accident
    of the target's absence, not a property of the type."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "broken.yaml").symlink_to(tmp_path / "nowhere.yaml")
    (configs / "broken.txt").symlink_to(tmp_path / "nowhere.txt")
    (configs / "real.yaml").write_text(RUN5.read_text())

    found = {path.name for path in discover_configs(configs)}
    assert found == {"broken.yaml", "broken.txt", "real.yaml"}, (
        f"a dangling symlink is a LOUD failure, not something to filter into silence; got {found}"
    )
    for name in ("broken.yaml", "broken.txt"):
        with pytest.raises(FileNotFoundError):
            load_config(configs / name)


def test_the_invariant_holds_on_the_REAL_configs_tree() -> None:
    """Prove the invariant on the tree that ships, not only on planted ones."""
    configs = REPO_ROOT / "configs"
    discovered = {path.resolve() for path in discover_configs(configs)}
    assert discovered, "gate 7 must never be vacuous"
    for path in configs.rglob("*"):
        if _loadable(path):
            assert path.resolve() in discovered, (
                f"{path} is loadable and undiscovered on the REAL tree — a launchable config no "
                "gate audits"
            )


def test_the_authority_is_EXPORTED_from_the_package_that_claims_it() -> None:
    """Prove the package exports the discovery authority it claims; a surface without it invites
    the next consumer to write another glob."""
    import mantis.config as package

    for name in ("discover_configs", "load_config"):
        assert name in package.__all__, f"{name} is not on mantis.config.__all__"
        assert getattr(package, name) is not None
