# HANDOFF — finishing SLIM-FIX (R368) from W3's close to W8

You are continuing a long, gated refactor. READ IN THIS ORDER before touching anything:
1. `CLAUDE.md` (hard rules, style, commit convention: ONE-LINE messages, NO trailers of any kind).
2. `docs/governance/RULINGS.md` → the entry `### R368` (the ruling you work under; clauses (a)–(l)).
3. `docs/slim/PROGRESS.md` (where the run is; what is done / still C / carded, per wave).
4. `docs/slim/handoff/WAVE_BRIEF.md` + `W2_ADDENDUM.md` + `W3_ADDENDUM.md` (the rules every implementation leg
   followed — write a W4/W5/W6/W7 addendum in the same shape before each wave).
5. `docs/slim/00_MAP.md` §3 (protected set), `01_DEFECTS.md`, `LEDGER.md` (§4 legs, §5 queue, §6 the row table), and
   the scout file for any row you touch (`docs/slim/S-*.md`, each with a `## Review` section that amends it).
6. The reviews so far: `docs/audits/REVIEW_W1_2026-09-23.md`, `REVIEW_W2_2026-09-24.md` (format to copy).

## The branch and the host
- Branch `claude/slim-fix-r368` (NOT pushed yet), in the main checkout. Base dev = 69e1532 (run10's validated base).
- The host is the operator's desktop (16 threads, 46 GB). `/tmp` is a RAM tmpfs with a PER-USER QUOTA: never put
  worktrees or cargo targets there (it voided a gate run once: `OSError: [Errno 122] Disk quota exceeded`).
  Worktrees go under `.wt/` in the repo (listed in `.git/info/exclude`), each with its own venv via `uv sync`.
- Python: `.venv/bin/python` directly (never `uv run` inside a sandbox). Rebuild the extension after Rust changes:
  `uv sync --reinstall-package mantis-engine`. After a `uv.lock` change: `uv sync`.
- Tests: `XDG_STATE_HOME=$TMPDIR/xdg .venv/bin/python -m pytest <nodes> -p no:cacheprovider`. A bare pytest is the
  default tier (`-m 'not integration and not slow'`).

## The per-wave loop (R368 "EVERY WAVE"; do not skip steps)
1. Implement (subagents or yourself), one commit per finding class, slimming never grows lines net.
2. Integrate each commit with `bash docs/slim/handoff/integrate.sh <sha>...` from the main checkout. It
   cherry-picks, then FOLDS the floor moves into that same commit: `tools/ci_gates/test_count_floor.txt` = the
   collected count (`pytest --collect-only -q -m ''`), and each measure in `comment_length_floor.txt` lowered to
   what `tools/ci_gates/comment_lint.py` measures. Floors only fall (comment) / track the count (tests).
   - exit 3 = cherry-pick conflict: resolve, `git add`, `GIT_EDITOR=true git cherry-pick --continue`, then
     `integrate.sh --fold <remaining shas>`.
   - exit 5 = comment_lint RED: a commit re-grew a measure above a floor an EARLIER commit lowered (branches measure
     against their own unmoved floor file). Shorten a docstring/comment THAT COMMIT added by the reported number of
     lines, `git commit --amend --no-edit`, then `integrate.sh --fold <remaining shas>`. Never raise a floor.
3. After any commit touching `src/mantis/{config,train,model}` or `run.py`, run10's resolved config must MATCH:
   `.venv/bin/python docs/slim/handoff/run10_resolved.py configs/run10.yaml | cmp - docs/slim/handoff/run10_resolved_base.json`
   (prints nothing = MATCH). A mismatch, a stamp/checkpoint-format change or a numerics change is a HALT (R368(i)).
4. A FRESH read-only review agent per wave (R367(b)): the dispatcher never reviews its own wave. Give it the range,
   R367(a) + R368 + correctness (budget, determinism, seam contracts, LAW-07). File its report VERBATIM under
   `docs/audits/REVIEW_W<n>_<date>.md` with a §5 disposition; fix every MUST-FIX before exit (two loops max, then
   HALT); fix should-fix or record the ground.
5. Wave exit: the full gate set WITHOUT the slow tier, in an on-disk worktree at the wave tip:
   `git worktree add --detach .wt/gates <tip>` (or `git -C .wt/gates checkout --detach <tip>`), `cd .wt/gates && uv
   sync`, then run `make gates` there (it takes ≈ 1.5 h: 2a ≈ 20–45 min, 3b ≈ 40–60 min). It must print ALL GREEN.
   A red you cannot attribute to the wave: record it in PROGRESS, do not fix it.
6. Update `docs/slim/PROGRESS.md` (wave, rows done/refuted/still-C with grounds, exit facts) and commit it.

## What is left
- **W3 close.** Every W3 implementation leg is INTEGRATED (incl. the sha256 leg). Left: REVIEW-W3 (a fresh read-only
  agent over `8b75f984..HEAD`, src rows; see PROGRESS W3 for what was done/kept), its fixes, `make gates` at the tip.
- **W4 TOOLS** (LEDGER legs L15–L19 + every TOOLS lane-C row re-laned under R368(b)):
  - the sealbot VENDOR side (R368(e)): `tools/vendor_build_sealbot.sh`, `vendor/patches/sealbot.patch`, the sealbot
    row in `vendor/pins.toml`, the Makefile `vendor.sealbot` target (tests/test_meta_ci.py pins the target set —
    update it), `tests/tools/test_vendor_build_sealbot.py`, `tests/tools/test_vendor_pins_sealbot.py`, the
    `make vendor.sealbot` line in CLAUDE.md, the sealbot-build prose in docs/contracts/eval_instrument.md;
  - gate 14 learns to MEASURE the two classes R368(g) names — bare `R\d{2,3}` ruling cites (incl. in docstrings) and
    narrative comment runs — defect 23 (`tools/ci_gates/comment_lint.py::_RULING` misses bare cites and never reads
    docstrings); baseline the new measures into `comment_length_floor.txt` at their measured values (down-only);
  - gate 16 widens to ZERO across the tree (R368(g)): `tools/ci_gates/encoding_io_gate.py` currently enforces tools/
    and tests/ module scope; fix the remaining sites, then widen the rule;
  - TOOLS-1-14/-15 KEEP (R368(k)); TOOLS-1 NEW-2 by R368(b).
- **W5 TESTS + FIXTURES** (L20–L34 + the TESTS lane-C rows re-laned; the PENDING-PROBE test rows were re-probed
  green — `docs/slim/REPROBE.md`, TESTS-5-06 amended):
  - R368(e) fixture deletions: `tests/fixtures/graph_parity/wpa_positions.json` + its manifest rows (the re-pin is
    granted; the manifest row names the sha and the commit the file lived in, 69e15329); the unread
    `tests/fixtures/value_probes/*` files + their CHANGELOG + manifest rows. `tests/test_fixtures_manifest.py` must
    stay green; generators of LIVE goldens stay;
  - the hoists (AQ-CARD-FAKES, AQ-LIFECYCLE: a card/citation protects a file's existence and assertion, not its
    helpers); AQ-BF16 (merge the `deterministic_algorithms` / `_graph_step` twins WITH a planted fp16 swap reding
    before and after); AQ-P2 (print set-equality of the two CONSUMER_REGISTRY dicts, then retire
    tests/config/test_regime_parity_p2.py's copy — its row in docs/contracts/run_config_schema.md moves in the same
    commit); AQ-POOL-HPARAMS (delete `tests/selfplay/test_pool_hparams_arms.py`'s killed-knob duplicate and its
    recording proxy; the survivor is test_pool_hparams.py's killed-and-relocated test — R38 is already annotated);
    L-SEAM-05/-06; the ~13 `train_step_from_tensors` test stubs; the defect-28 residue (the flat `glob("*.yaml")`
    at 7 sites listed in PROGRESS W1); REVIEW-W1 notes 9/10/11; the `/tmp/mantis-*` dirs tests leak (mkdtemp without
    cleanup: mantis-abort-exit-*, mantis-root-lifecycle-*, mantis-atomic-*);
  - LAST in W5: the dense drain oracle re-base (R368(d)): capture graph drain goldens (after W1), point the
    drain-parity suite at them, prove a PLANTED drain defect reds them, then delete `pool_push.push_dense`, the
    dense arm in `pool_drain.run_stats_loop` (CORE-2-23), `pool._is_graph` and the dense PoolDims fields.
- **W6 DOCS + CONFIGS** (L35–L38): STATE.md rewritten to current facts (R368(f): each dropped paragraph class names
  where it lives — a ruling, a measurement record or a commit); CARDS.md closed rows removed; delete
  `docs/governance/archive/RULINGS_ACTIVE.md` (annotations already landed); delete configs/run6.yaml, run7.yaml,
  run8.yaml (tests binding them re-point to the census `mantis.config.census.production_configs`); stale text
  01_DEFECTS 40–51; CLAUDE.md's comment bullet becomes R368(g)'s one rule; repo_design §3 (GnnArchV2 has no base
  class); remove the "28 bench floors" counts (AQ-FLOORS28: derive at point of use). Gate 10's DISSOLVED_PATHS takes
  deleted paths a scanned doc still cites.
- **W7 STYLE PASS**: R368(g)'s two classes only (ruling/card/finding cites and narrative runs), ONE package per
  commit, the floor lowered in the same commit; carve-out markers stay.
- **W8 CLOSE**: the AFTER figure (00_MAP §1's command, per top dir and per slice); one exit paragraph in STATE.md;
  `git rm -r docs/slim` then `test ! -e docs/slim`; the FULL gate set incl. the slow tier (`make gates.exit`,
  ≈ 2 h); then STOP at READY-TO-MERGE: do NOT fast-forward or push dev — the operator (or the original session)
  does ENACTS 4 after reading the exit report.

## Hard HALTs (a halt is a success: record it in PROGRESS and stop that row)
torch or the engine absent; a planted break that stays green; an edit to a protected symbol or its pinning test beyond
import re-points without a ruling naming it; run10's resolved config / the stamp format / numerics would move; a
review must-fix still open after two loops; a gate red you cannot attribute to the wave (record, do not fix).
Never: edit configs/*.yaml values, schema keys or defaults (R368(h)); touch the box or any priced act (R367(d));
push dev.
