# W3 ADDENDUM — the Python src wave (read after WAVE_BRIEF.md)

Slimming (R368(d)/(e)/(k)) plus two correctness classes (the broad-except class, and defects 15/16/17 that
resolve here). A deletion of dead code is proven by the touched packages' tests + the import DAG (gate 9) +
pyright (gate 14) green before and after; a change that touches a WITNESS needs its planted break (brief).
Slimming NEVER grows lines net per commit; a correctness commit may.

## Rows and lanes
- Your rows are listed in your prompt with census IDs. docs/slim/LEDGER.md §6 (table) / §5 (queue) give the lane;
  docs/slim/S-A-CORE-N.md (+ S-L-DUP.md, S-L-SEAM.md) give subject/claim/Δ/witness and the `## Review` amendments;
  docs/slim/REPROBE.md gives the torch-host re-probe of the PENDING-PROBE rows (all green; TESTS-5-06 amended;
  CORE-1-08 must edit tests/train/test_trainer_seam_conformance.py's GridRouteBufferLike import + SEAM_MATRIX row
  in the SAME commit).
- LANE C rows: RE-LANE under R368(b) first (→A / →B / still C, with the ground), then implement. Protection binds
  the INVARIANT (00_MAP §3 PZ-1's listed symbols + pinning tests; PZ-2 seam members; PZ-6 ruling-named code while
  the ruling still directs the code to exist). Other code in a PZ-glob file changes normally, the file's pinning
  tests green before and after. A card never protects code (the leg that touches it updates the card). A header
  calling itself frozen is a claim.
- R368(d) ORDERS: grid/dense residue gone (arms, flags, stubs, pins); the phantom protocol members
  `TrainerLike.train_step_from_tensors` and `GridRouteBufferLike` gone; run-named symbols and pins gone; duplicated
  helpers made one (sha256 onto `mantis.util.hashing` — its own agent, last).
- R368(e) DECIDED: the sealbot adapter (src side here; vendor script/patch/pin in W4) with `find_vendor_root`
  MOVED FIRST to where its remaining users (strix) need it; src/mantis/data's old corpus pipeline except what the
  kept audit tool imports.
- R368(k): AQ-X1 "imports decide; message text follows" (tools/audit_bootstrap_corpus.py imports stdlib only, so
  data/sources + generate.py + mantis.env may go; the tool's message/record strings naming them are reworded in the
  same commit); CORE-1-24 delete, "the resolver's detector stays"; CORE-2-14 is OUT (schema key — carded).
- R368(h) OUT: schema keys and minted rows; hot-path refactors in collate (dead-code removal there is in).
- A deletion that removes a Python TEST (a test importing a deleted module) happens in the SAME commit; report the
  collected-count drop (the dispatcher moves the floor) and remove any tools/ci_gates/tier_declaration.txt row.
- A structural removal (a package like data/ or env/) needs the docs/design/repo_design.md amendment and the
  CLAUDE.md map edit in the same commit (R9) — edit ONLY the lines that name the removed thing.
- Contract docs (docs/contracts/*.md): an event field / payload / resolver a doc names moves with its doc (version
  bump per the doc's convention); gate 13 (`tools/ci_gates/contract_doc_gate.py`) checks run_config_schema.md's
  `mantis.*` symbols and keys; gate 10 checks doc path tokens (DISSOLVED_PATHS in check_tracked_refs.py for R368(e)
  deletions only). tests/config/test_every_key_has_consumer*.py resolve every CONSUMER_REGISTRY string to a symbol
  defined in src/crates — deleting a cited symbol reds it: re-point the registry string to the live reader.
- src/mantis/monitor/producer_manifest.yaml and src/mantis/config/armed_aborts.py `source_pin`s name modules,
  symbols and CODE SUBSTRINGS: gate 12 and the manifest verifier red if you edit a pinned line's text. Do not touch
  pinned lines unless a row requires it.

## The broad-except class (R368(g) last bullet), for your packages
Classify every `except Exception` / bare `except:` / `except BaseException` in src/ of your packages as:
(a) a top-level handler that logs via `logger.exception` (CLAUDE.md allows it — keep, fix the message if it
repeats the exception); (b) a production site that swallows or re-labels — FIX: catch the specific exceptions the
guarded call raises (read the callee), or re-raise (LAW-14: persistence failures are run-fatal, no silent
excepts); (c) a hot-loop site — do not restructure; list it for a card. Report the table (counts, fixed sites,
kept top-level handlers with why). A fix that changes behaviour on an error path must keep that path's test green,
or add one if none exists.

## run10 safety (R368(i)) — hard
- After ANY commit touching src/mantis/config/, src/mantis/train/, src/mantis/model/, src/mantis/run.py:
  `.venv/bin/python docs/slim/handoff/run10_resolved.py configs/run10.yaml | cmp - docs/slim/handoff/run10_resolved_base.json`
  must print nothing.
- No change to the checkpoint/stamp format (src/mantis/train/checkpoints.py save/load, eval/snapshot.py) or to
  trainer/search/eval numerics. Deleting code unreachable BY STRUCTURE is not such a change.

## Checks
- Targeted tests while iterating (the touched modules' tests + every test that greps a removed name:
  `git grep -n <name> -- src tools tests crates docs`); at the end, every test file under the tests/<pkg>/ dirs of
  your packages ONCE (default tier `-m 'not integration and not slow'`; add `-m integration` only for files whose
  subject you changed and that are integration-marked).
- `.venv/bin/python tools/check_import_dag.py src/mantis`; pyright: `bash tools/ci_gates/lint_gate.sh --self-test`
  (if node/mise resolves; else `.venv/bin/pyright --pythonpath .venv/bin/python` and say so); `ruff check .`;
  `comment_lint.py`, `r8_header_gate.py`, `encoding_io_gate.py`, `contract_doc_gate.py`, `check_tracked_refs.py`,
  `validate_configs.py`, `preflight_mint.py --audit-only` (gate 12) — all green.
- One commit per finding class (group rows of one class in one package), one-line messages, no trailers.

## Report additions
- The re-lane table: ID | old lane | new lane | ground.
- The broad-except classification table.
- Per commit: rows, Δlines net (≤ 0 for slimming), tests run, collected count after, run10 MATCH if applicable.
