# W7 ADDENDUM — the style pass (read after WAVE_BRIEF.md)

R368(g)'s ONE sanctioned pass. It covers exactly two classes and nothing else:
1. **Cites**: ruling, law, finding and card tokens in comments, docstrings, Rust docs and text-format
   comments (gate 14's `ruling_cite_lines`; the token regex is `_RULING` in
   `tools/ci_gates/comment_lint.py`).
2. **Narrative runs**: own-line comment runs longer than two lines that tell a story instead of stating
   what the code cannot (`comment_excess_lines`; `textfile_comment_excess_lines` for `tools/` text
   formats and the Makefile).

The rule the pass applies: a comment or docstring states what the code cannot, in one line; more only
for an invariant. It carries no ruling, card or finding number except in a carve-out marker.

## Hard limits (beyond WAVE_BRIEF's)
- **ONE package per commit.** The package unit is the row of the table below. A leg may do several
  packages, one commit each. The dispatcher's `integrate.sh` lowers every floor that fell into the same
  commit; implementers never edit `comment_length_floor.txt` or `test_count_floor.txt`.
- **Carve-out markers STAY**: pinned bands, planted-break markers, armed-value provenance and licence
  attribution. Also keep:
  - the in-source-only `CARD-*` markers that CARDS.md's "CARD-* that exist only as in-source markers"
    section lists. The token is the card's only home; removing it deletes the card.
  - Rust `// SAFETY:` comments and any comment that states an invariant.
  - Gate 15's R8 justification headers. Note that R8 sits below the cite regex's R10 floor.
- **A token a test reads is load-bearing.** Some tests read source text, for example
  `test_the_one_provenance_pin_still_exists_and_still_names_its_grounds` (`"adjudication A-3"`) and the
  gate self-test corpora. If a package's tests red after an edit, restore the token. Never edit the
  test.
- **On contact only, NOT this pass:** docstring length (`docstring_excess_lines`,
  `private_docstring_excess_lines`, `rust_doc_excess_lines`), missing docstrings and `Raises:`
  sections. A cite line dropped from a docstring may lower `docstring_excess_lines` as a side effect.
  That is fine, because floors only fall.
- **OUT of W7:**
  - `.github/**`: R368(h), the suspended CI workflow is the operator's switch.
  - `tools/config_templates/**`: WAVE_BRIEF, mint inputs are not edited.
  - `tests/fixtures/**`.
  - Register text: RULINGS.md, LAWS.md and falsified.md.
  - Every `docs/` file.
  - String literals and assertion messages. They are not comments. A cite in a message stays.
- **tools/ci_gates/** is ruling-named code under R368(b).
  - Comments only. Never touch a pattern register, a self-test corpus or a planted-fixture string.
  - Every gate's own self-test must stay green.
  - If a cite in a gate is the ruling that ordered it, keep it when the gate's refusal text depends on
    it, and record the ground.
- **Rust:**
  - Run `rustfmt --edition 2021` on the touched files only.
  - Run `cargo clippy -p <crate> --all-targets --locked -- -D clippy::all`.
  - Run `cargo test -p <crate>` with targeted `--test` names.
  - Comment edits do not rebuild the wheel. There are no numerics, and hot paths are safe to touch
    for comments only.
- **Gate 15 stale rule:** a file that drops to 300 lines or fewer LOSES its R8 header in the same
  commit. Re-run `tools/ci_gates/r8_header_gate.py` on every touched file.
- **run10:** `configs/**`, the schema, the stamp and the numerics stay untouched. Check run10 MATCH at
  each integration.

## Current measures per package
Derived at `8eec45a9` by summing `comment_lint.measure_source` over `git ls-files` per package. The
scratch script was the dispatcher's. Re-derive the same way at leg start and never trust these
figures. Totals at that tip: cite 1163, comment_excess 3124, textfile 466. They equal the floor file.

| package | cite | comment_excess | textfile |
|---|---:|---:|---:|
| crates/mantis-selfplay | 78 | 871 | 0 |
| crates/mantis-search | 9 | 559 | 0 |
| tools/ci_gates | 67 | 92 | 252 |
| tests/train | 137 | 129 | 0 |
| crates/mantis-core | 5 | 207 | 0 |
| crates/mantis-bridge | 5 | 189 | 0 |
| tests/config | 90 | 99 | 0 |
| src/mantis/train | 73 | 110 | 0 |
| tools/ (root files) | 39 | 8 | 92 |
| src/mantis/ (root files) | 17 | 118 | 0 |
| src/mantis/config | 39 | 87 | 0 |
| tests/tools | 84 | 36 | 0 |
| tests/selfplay | 91 | 28 | 0 |
| crates/mantis-encoding | 13 | 105 | 0 |
| crates/mantis-graph | 5 | 112 | 0 |
| src/mantis/eval | 41 | 62 | 0 |
| tests/eval | 53 | 44 | 0 |
| tests/model | 36 | 43 | 0 |
| tests/bridge | 24 | 31 | 0 |
| tests/ (root files) | 26 | 28 | 0 |
| src/mantis/selfplay | 20 | 31 | 0 |
| src/mantis/monitor | 22 | 19 | 0 |
| tests/diagnostics | 29 | 10 | 0 |
| tests/monitor | 16 | 18 | 0 |
| tests/encoding | 27 | 5 | 0 |
| src/mantis/diagnostics | 9 | 14 | 0 |
| src/mantis/model | 10 | 11 | 0 |
| src/mantis/arena | 0 | 18 | 0 |
| tools/dashboard | 16 | 0 | 0 |
| Makefile | 8 | 0 | 6 |
| src/mantis/encoding | 3 | 10 | 0 |
| src/mantis/util | 4 | 8 | 0 |
| tests/util | 4 | 8 | 0 |
| tests/arena | 8 | 3 | 0 |
| src/mantis/data | 0 | 10 | 0 |
| tools/ladder | 9 | 0 | 0 |
| src/mantis/bots | 5 | 0 | 0 |
| tests/data | 2 | 1 | 0 |
| tools/probe1 | 2 | 0 | 0 |
| tools/viewer | 1 | 0 | 0 |

Out-of-W7 rows at the same tip:
- `.github`: cite 9, textfile 34.
- `tools/config_templates`: cite 27, textfile 82.

Both stay on the floor as they are.

## Leg plan
At most four legs at once. Each leg works in its own `.wt/w7-<leg>` worktree and makes one commit per
package. The dispatcher integrates with `integrate.sh` after each leg and records the model per leg in
PROGRESS.

| leg | packages | model | why that model |
|---|---|---|---|
| L1 | crates/mantis-selfplay | opus | the largest run; hot paths, SAFETY and invariant comments, a CARD marker (`replay_sampler_seed.rs`) |
| L2 | crates/mantis-search, crates/mantis-encoding | opus | solver and registry invariants; each narrative run needs a judgment about whether it states one |
| L3 | tools/ci_gates | opus | ruling-named code (R368(b)); textfile runs in the gate scripts; self-test corpora and pattern registers must not move |
| L4 | crates/mantis-core, crates/mantis-bridge, crates/mantis-graph | sonnet | mechanical; `rustfmt` and `clippy -p` per crate |
| L5 | tests/train, tests/config, tests/tools, tests/selfplay | sonnet | mechanical cite drops; the package tests are the check (source-reading tests red on a load-bearing token) |
| L6 | the remaining tests/ packages: eval, model, bridge, root files, diagnostics, monitor, encoding, util, arena, data | sonnet | mechanical |
| L7 | src/mantis/train, src/mantis/config, src/mantis/eval, src/mantis/ root files | opus | armed-value provenance and PZ-6 manifest text sit beside ordinary comments; the manifest's string fields are data, not comments |
| L8 | the remaining src/mantis packages (selfplay, monitor, diagnostics, model, arena, encoding, util, data, bots); tools/ root files, dashboard, ladder, probe1, viewer; Makefile | sonnet | mechanical; textfile runs in tools/ root text formats and the Makefile |

Order: run L1–L4 first, then L5–L8. L5 and L6 touch only tests; L7 and L8 touch only src and tools, so
none of them overlap.

## Checks (each package commit)
- The package's targeted tests pass, in the default tier. For crates, use `cargo test -p` with the
  touched test targets.
- `comment_lint.py` prints GREEN, and every measure is at or below its floor.
- Gates 15 and 17: run `--base` against the leg's start. Also run gate 14's pyright through
  `make lint`, once per leg.
- run10 MATCH at each integration.
- Collection is unchanged. A comment pass deletes no test; if the count moves, that is a HALT.

## W7 exit
Do the same as W6's exit:
1. A fresh REVIEW-W7 (opus), filed verbatim under docs/audits/ with a §5 disposition, then its fix
   loop.
2. `make gates` in `.wt/gates` as a systemd user unit, all green.
3. run10 MATCH.
4. PROGRESS exit facts, and HANDOFF's "What is left" (W8 next).

REVIEW-W7 checks four things:
- no carve-out marker was dropped;
- no in-source-only CARD marker was lost;
- no invariant comment was cut to a story-less stub;
- each commit touches ONE package and lowers its floor.
