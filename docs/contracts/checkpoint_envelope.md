# Contract: checkpoint envelope

- version: v2
- owner: mantis.train.checkpoints
- status: LIVE — the envelope writer/reader and THE ONE loader landed with WP10; this text is
  the contract half, filled by WPMINT Phase W. `CHECKPOINT_SCHEMA_VERSION = 2` is the ENVELOPE
  axis and is deliberately distinct from a config's own `schema_version: 1`.

## Summary
One format, one loader (docs/design/repo_design.md §6). The filename carries run-id + content
hash; stamps are written once and are immutable; an artifact that cannot be stamped cannot be
written; every read surface is `torch.load(weights_only=True)`; a legacy payload is never
auto-upgraded to a v2 stamp on read.

## Shape

**Filename** — `{run_id}_{step:08d}_{sha8}.ckpt` (`checkpoint_filename`). `sha8` is the first
8 hex of a sha256 over a key-ordered serialization of the WHOLE payload (`content_sha8`), so a
one-byte `model_state` mutation changes the name. Both halves are re-verified at load, which
makes a cross-lineage same-step collision structurally impossible.

**Payload** — `schema_version: 2`, `kind: "full" | "weights"`, `model_state`, `metadata`,
`config` (a complete snapshot, schema-validated); `kind == "full"` additionally carries
`optimizer_state`, `scaler_state`, `scheduler_state`.

**`metadata`** — `encoding_name` (REQUIRED, LAW-11, no fallback), `run_id`, `step`,
`commit_sha` (`"unknown"` outside a git checkout — never blocks a write), `created_utc`
(ISO-8601 Z, stamped once), `arch` (the declared `CnnArch`/`GnnArch` dataclass — the SOLE arch
source at load), optional `corpus_sha256`.

## Who asserts what where

| assertion | where |
|---|---|
| stamps are written exactly once; a `metadata_kwargs` carrying `created_utc`/`commit_sha` is a REFUSED re-stamp (LAW-12) | mantis.train.checkpoints (`_build_stamped_metadata` -> `CheckpointStampError`) |
| an unstampable artifact is not written: an absent `encoding_name`, `run_id` or `arch` each raise | mantis.train.checkpoints (`_build_stamped_metadata`) |
| a failed write is persist-FATAL, never swallowed; the survive-the-run path writes a `.quarantine` file, NEVER a canonical `.ckpt`, and increments the persist counter (LAW-14) | mantis.train.checkpoints (`_write_quarantine`, `persist_errors_total`) |
| filename run-id + content hash are re-verified at load (provenance) | mantis.train.checkpoints (`load_checkpoint` -> `_verify_provenance`) |
| every read surface is `torch.load(weights_only=True)`; there is no pickle-exec fallback | mantis.train.checkpoints (`load_checkpoint`, `load_legacy_weights`, `strip_and_restamp`, `resume_trainer`) |
| a non-v2 payload is REFUSED by `load_checkpoint` and must go through the distinct legacy surface — a v2 stamp is never auto-minted on read | mantis.train.checkpoints (`load_checkpoint`, `load_legacy_weights`) |
| arch travels on `metadata.arch`; there is no shape-inference and no sniff-reconstruct on either surface | mantis.train.checkpoints (all shape-inference was DELETED at WP10) |
| a state dict carrying a falsified-and-deleted branch prefix is REJECTED on BOTH read surfaces (WP9 O3b) | mantis.train.checkpoints (`KILLED_PREFIXES`, `_reject_killed_prefixes`) |
| declared-encoding is an ASSERTION (a mismatch raises); decode-override is a deliberate loud cross-decode; both together is an error | mantis.train.checkpoints (`DeclaredEncodingMismatchError`, `load_checkpoint`) |
| resume precedence: the launch config wins EXCEPT a frozen 18-key checkpoint-owned set | mantis.train.orchestrator (`RESUME_CHECKPOINT_OWNED_KEYS`, `build_resume_config_overrides`) |
| the sanctioned encoding-change path is a weights-only strip gated on WIRE-SIGNATURE equality, stamped fresh from the declared encoding and never from a loaded config | mantis.train.checkpoints (`strip_and_restamp`) |

## Pinning tests

| test | file |
|---|---|
| the envelope-v2 conformance suite (`T-CK-*`): filename/content-hash, immutable stamps, unstampable-refuse, quarantine, weights-only, killed-prefix reject, and the strip path's wire-signature gate | tests/train/test_checkpoint_conformance.py |
| `RESUME_CHECKPOINT_OWNED_KEYS` equals the exact frozen 18-key set (golden-pinned) | tests/train/test_checkpoint_conformance.py, tests/fixtures/train/resume_goldens.json |
| resume precedence end to end: launch-wins outside the owned set, checkpoint-wins inside it | tests/train/test_resume_semantics.py, tests/train/test_resume_wiring_integration.py |
| warm-start / weights-only load from a foreign lineage | tests/train/test_warmstart.py |
| an anchor whose core is missing raises `AnchorLoadError` rather than degrading | tests/train/test_anchor.py |
| the launch path writes an envelope-v2 checkpoint, resumes from it, and shuts down clean on a simulated signal (integration tier) | tests/train/test_launch_path_smoke.py |
