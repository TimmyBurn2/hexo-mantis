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

## What is left (from the 2026-09-24 close-out)

W0–W7 are EXITED (PROGRESS carries each wave's exit facts and per-row ledger). NEXT is W8, per
`docs/slim/handoff/W8_ADDENDUM.md`. That file supersedes the W8 summary below where the two differ. It opens
with W8-D, the operator's grants of 2026-09-25: run6 stays and is unbound by name, the MANIFEST prose,
pipeline.py's repr, the R336(e) annotation, the archive tombstone, and the dead Trainer.load_checkpoint.
The waves:
- **W5 TESTS + FIXTURES** — EXITED 2026-09-25 at 8297c74d (PROGRESS `## W5`: REVIEW-W5 closed in one
  loop, exit sweep green, the KNOWN-RED cured by cdbc8000, collected 4927, run10 MATCH).
- **W6 DOCS + CONFIGS** — EXITED 2026-09-25 at 0c8b9176 (PROGRESS `## W6`: REVIEW-W6 closed in one loop,
  exit sweep ALL GREEN, collected 4928, run10 MATCH). configs/run7.yaml and run8.yaml are deleted; **configs/run6.yaml
  STAYS on a HALT** that needs an operator grant (R310-frozen preflight oracle plus the ruling-named deploy_sims row;
  the measured diff is `docs/slim/handoff/halt_twin_heldout.diff` and it NARROWS the oracle, see PROGRESS).
- **W7 STYLE PASS** — EXITED 2026-09-25 at the W7 exit commit (PROGRESS `## W7`). All eight legs landed, one
  package per commit. REVIEW-W7 closed in one loop, the exit sweep was ALL GREEN, collected 4928, run10 MATCH.
  Measures fell: cite 1163→91, comment_excess 3124→2144, textfile 466→228.
- **W8 CLOSE** (see W8_ADDENDUM.md): the W8-D grants first, then the AFTER figure (00_MAP §1's command, per top dir and per slice); one exit
  paragraph in STATE.md; `git rm -r docs/slim` then `test ! -e docs/slim`; the FULL gate set incl.
  the slow tier (`make gates.exit`, ≈ 2 h); then STOP at READY-TO-MERGE: do NOT fast-forward or
  push dev — the operator (or the original session) does ENACTS 4 after reading the exit report.

### Session notes for the next dispatcher (learned 2026-09-24, W3/W4)
- **The KNOWN-RED** (`tests/monitor/test_supervisor_signal_posture.py::
  test_the_stop_handlers_are_installed_by_main_and_never_at_import`): reds in FULL default-tier
  runs roughly half the time on this host (it failed W3's and W4's sweeps, passed REVIEW-W4's run
  and two others, all at the same commits); standalone it always passes. Full account + mechanism
  in PROGRESS W3's exit: the pytest process's SIGINT disposition is transiently SIG_IGN (C-level,
  not the Python signal module) when the test spawns its probe child, and CPython preserves an
  inherited SIG_IGN across exec. CURED in W5 by cdbc8000: the probe child starts from SIG_DFL.
  REVIEW-W5 §3 H reproduced the old red, saw the new pass and found the plants still red, so the row
  is not narrowed. W5's sweep 3a had 0 failed. A red there now is a real red.
- Subagents: `reviewer` (read-only reviews — dispatch each wave's REVIEW there, never review your
  own wave), `worker` (implementation legs in their own worktrees), `scout` (read-only
  inventories). Worktrees under `.wt/<name>` with `git worktree add -b <branch> .wt/<name> HEAD`
  + their own `uv sync`; NEVER /tmp (RAM quota). Gate sweeps in a worktree: `uv sync` once, then
  `make gates` (it self-gates with UV_NO_SYNC=1).
- Floors: `integrate.sh <sha>...` cherry-picks worker branches onto the main branch AND folds
  test_count_floor + comment floors into each commit. Commits made DIRECTLY on the main branch
  (review fixes, dispatcher legs) must fold manually in the same commit (measure, write floor,
  `--amend`). All seven comment measures are gated now; a docstring edit to a scoped file can
  grow `docstring_excess_lines` — compact before committing.
- Gate 14's cite measure counts BARE R-tokens from R10 up (one-digit stays out so gate 15's R8
  headers never fight it) in comments, docstrings, Rust docs AND tools/ text-format comments; do
  not add new cites in code comments (use words), and remember CLAUDE.md is unmeasured.
- `.wt/gates` is the standing sweep worktree (detached); `git -C .wt/gates checkout --detach <tip>`
  (name the SHA — its HEAD is detached, `HEAD` there resolves to itself) + `uv sync` + `make gates`.
- The venvs are Python 3.13.14 (uv's only local interpreter); CI's floor is 3.11 — fine.
- Never push, never touch dev, never edit configs/*.yaml values; a HALT is a success: record it in
  PROGRESS and stop that row.

## Hard HALTs (a halt is a success: record it in PROGRESS and stop that row)
torch or the engine absent; a planted break that stays green; an edit to a protected symbol or its pinning test beyond
import re-points without a ruling naming it; run10's resolved config / the stamp format / numerics would move; a
review must-fix still open after two loops; a gate red you cannot attribute to the wave (record, do not fix).
Never: edit configs/*.yaml values, schema keys or defaults (R368(h)); touch the box or any priced act (R367(d));
push dev.
