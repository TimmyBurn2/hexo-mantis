"""Which configs are PRODUCTION: every file `discover_configs` finds under `configs/` that no `EXEMPT_CONFIGS` row names — a census taken at point of use, never a list edited per mint."""
from __future__ import annotations

from pathlib import Path

from mantis.config.loader import discover_configs

#: The directory the census walks, repo-relative; the exempt rows are spelled from the repo root.
CONFIG_DIR_REL = "configs"

#: The ONE by-name list `(repo-relative path, why exempt)`: unlisted means production, so
#: "forgotten" reads as "audited", and the gate prints the reason, so no exemption is bare.
EXEMPT_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "configs/dev_example.yaml",
        "developer template, never minted for a run; DISARMED by design (R59). R346(f) pruned "
        "configs/ to run6 plus one smoke and this file was cut with the rest — it is BACK, and "
        "the ground is LAW-07: ADJ-13 N-3 makes it the mutation corpus's M1 row, the one real "
        "committed config that demonstrates gate 12 going RED on the real `configs/` tree. "
        "With run5, the shakedown and the plain smoke gone it is the only disarmed config "
        "left, so deleting it would leave the gate with no red-capability demonstration on "
        "the tree it audits. Pinned by "
        "`test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored`.",
    ),
    (
        "configs/smoke_preflight_armed.yaml",
        "armed preflight-rehearsal smoke config (WPTS Phase F, R103): NOT a production run, "
        "but unlike the R59 smokes it ARMS both required rows at burst-scale guard values so "
        "mode PREFLIGHT can run a completed bounded burst off-run5. Exempt from the "
        "every-production-config gate-12 sweep for the same reason the other smokes are; "
        "`--config` still unions it into the audit set, and its live consumer is the burst "
        "oracle in tests/tools/test_preflight_armed_smoke.py (LAW-08).",
    ),
)


class ConfigCensusError(ValueError):
    """The census cannot be taken: an exempt row names no file on disk, or nothing is left to bind."""


def exempt_config_paths() -> frozenset[str]:
    """The exempt rows' repo-relative paths."""
    return frozenset(rel for rel, _reason in EXEMPT_CONFIGS)


def discovered_config_paths(repo_root: str | Path) -> list[str]:
    """Every config on disk under `<repo_root>/configs`, repo-relative and sorted — gate 7's own enumeration."""
    root = Path(repo_root)
    return sorted(path.relative_to(root).as_posix() for path in discover_configs(root / CONFIG_DIR_REL))


def production_configs(repo_root: str | Path) -> tuple[Path, ...]:
    """Every config under `<repo_root>/configs` that is not exempt, sorted absolute paths; Raises: ConfigCensusError — an exempt row names no file on disk, or no config is left (an empty census binds no law)."""
    root = Path(repo_root)
    present = discovered_config_paths(root)
    stale = sorted(exempt_config_paths() - set(present))
    if stale:
        raise ConfigCensusError(
            f"EXEMPT_CONFIGS names {stale}, absent from disk under {root / CONFIG_DIR_REL}: a STALE "
            "exemption is auditing a file nobody will run — re-point or drop the row"
        )
    production = tuple(root / rel for rel in present if rel not in exempt_config_paths())
    if not production:
        raise ConfigCensusError(
            f"the config census under {root / CONFIG_DIR_REL} is EMPTY once the exempt rows are "
            "removed — an empty census binds no config to any law"
        )
    return production
