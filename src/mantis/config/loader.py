"""Config loader: yaml.safe_load (duplicate-key-rejecting) -> schema validation. No merge,
no defaults, no env expansion. Missing key and unknown key both raise pydantic.ValidationError
(loud, listing every error). A duplicate YAML key (frozen loader silently last-won) is a
HARD error here (judgment #8).

This module also carries `discover_configs` — the ONE enumeration both gate 7 and gate 12
consume (R71 / ADJ-13 F-1), governed by the **shared-authority invariant** (R75).

**The class, in one sentence.** A file under the audit root that the loader will READ but
discovery will not ENUMERATE is a production config nobody audits — and every name-based
discovery filter creates exactly that gap, because the loader's accept-set is defined by
CONTENT, not by name.

Four escapes, each one the complement of the previous fix's enumeration:

    configs/run6.yaml   (MF-7)      -> fixed, and run6.yml walked through
    configs/run6.yml    (RED-TEAM)  -> fixed, and configs/prod/ walked through
    configs/prod/*.yaml (RED-TEAM)  -> fixed, and run6.txt walked through
    configs/run6.txt, run6.YAML (RECHECK R-2)

**The invariant, stated as the rule this module holds:**

    load_config(p) succeeds  =>  p in discover_configs(root)      for every p under root

Equivalently, and this is the form to test: *a file discovery skips must be a file the loader
refuses.* It closes the class from the discovery side, WITHOUT constraining what a run may be
launched from — a run may be launched from a path of any shape, which is what the loader did
before ADJ-13's corrective pass and does again (R75 DECLINED that narrowing).

**Why discovery is name-agnostic.** `load_config` decides by content: it reads the bytes and
hands them to the schema. No suffix test can bound that set from above, so any suffix test in
discovery leaves the complement launchable-and-invisible. Discovery therefore enumerates
EVERYTHING under the root and drops exactly one kind of path — a **real directory** — because
`read_text()` on a directory raises `IsADirectoryError` unconditionally, for every directory,
by type rather than by name. That is a proof, not a heuristic, and it is the only such proof
available. A symlink TO a directory is deliberately NOT dropped: `rglob` does not recurse
through it, so dropping it would hide a whole subtree of loadable configs (the input just
outside this boundary — R71). Kept, it is a loud gate-7 failure instead of a silent hole.

**The measured cost, and it is the point rather than a side effect.** Gate 7 schema-validates
whatever discovery returns and gate 12 requires it to be declared, so `configs/` may now contain
ONLY complete configs: a stray `README.md`, a `.gitkeep` or an editor's `run5.yaml.bak` is a red
gate. That is CORRECT, not a defect — `run5.yaml.bak` is a near-copy of a production config in
the audit root and it IS loadable, i.e. escape #5; and the only way to spare the `README.md`
without sparing the `.bak` is a name filter, which is the class. Notes belong in `docs/`.

**What this does NOT close:** where a config may LIVE. A loadable file outside `configs/` —
`<repo>/scratch/run6.yaml`, `/tmp/run6.yaml` — is launchable and `discover_configs(configs/)`
will never see it. Deliberate (preflighting a candidate from a scratch directory is the normal
case, and every `--config` route depends on it), carded as `CARD-CONFIG-DISCOVERY-ROOT`. The
mint path is covered shape-agnostically instead: `preflight_mint.py --config <path>` unions any
named path into the audit set regardless of its shape (`_audit_paths`).

SF-4 is preserved: `discover_configs` takes the directory as an ARGUMENT and resolves no repo
root. `REPO_ROOT` stays in the tool (`preflight_mint.py:104`), where `parents[2]` is structurally
sound rather than dependent on an editable install.
"""
from pathlib import Path

import yaml

from mantis.config.schema import RunConfig


class DuplicateKeyError(ValueError):
    """A YAML mapping declared the same key twice (silent last-wins is banned)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that HARD-ERRORS on a duplicate key at any nesting depth.

    The frozen utils/config.py merely LOGGED a warning and kept the last value; a duplicate
    key is a copy-paste hazard, so it becomes a load-time error. Safe construction is preserved
    (SafeLoader subclass — no arbitrary object construction).
    """

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at {key_node.start_mark}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def discover_configs(configs_dir: str | Path) -> list[Path]:
    """EVERY path under `configs_dir` that is not a real directory, RECURSIVELY, sorted.

    The one enumeration both gate 7 and gate 12 consume (R71), and the discovery side of the
    shared-authority invariant (R75): *a file discovery skips must be a file the loader
    refuses.* Name-agnostic on purpose — `load_config` decides by content, so a suffix filter
    here leaves its complement launchable-and-invisible, which is ADJ-13 F-1's whole class.

    The one exclusion, and why it cannot become a hole. `path.is_dir() and not
    path.is_symlink()` is a REAL directory: `read_text()` on it raises `IsADirectoryError` for
    every directory unconditionally, so the loader provably refuses it, and `rglob` recurses
    THROUGH it, so everything loadable beneath it is enumerated anyway. Both halves are needed:

    * drop `is_dir()` and `configs/prod/` — a supported `mint_config.py --out` target — becomes
      a gate-7 failure for existing;
    * drop `not is_symlink()` and a symlink to a directory is dropped too, while `rglob` refuses
      to recurse through it (measured: `link/hidden_cfg.yaml` absent from `rglob("*")`), so an
      entire subtree of loadable, disarmed configs goes invisible to both gates. That is the
      input just outside this boundary (R71), and keeping the symlink in the enumeration is
      what closes it: gate 7 hits `IsADirectoryError` and goes loud.

    Sorted so two consumers of one tree cannot disagree about order. Dotfiles and hidden
    subdirectories are included — `pathlib.rglob` does not skip them (measured), and
    `configs/.yaml` was escape #6.
    """
    return sorted(path for path in Path(configs_dir).rglob("*")
                  if not (path.is_dir() and not path.is_symlink()))


def load_config(path: str | Path) -> RunConfig:
    """Load and schema-validate one complete config file (duplicate keys rejected).

    Shape-agnostic: a run may be launched from a path of any name (R75 DECLINED the accept-set
    narrowing that briefly stood here). What keeps that safe is not a suffix test but the
    invariant `discover_configs` holds — whatever this function accepts under the audit root,
    the audit sees.
    """
    raw = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    return RunConfig.model_validate(raw)
