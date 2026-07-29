# Contract: mint preflight evidence report

- version: preflight-mint-v1 (the string the artifact carries in its own `schema` field)
- owner: tools/ci_gates/preflight_mint.py
- status: LIVE. Written by WPMINT Phase W (F-B3): the report has been produced and asserted
  since WPAX, and Phase B added a top-level field to it, but nothing under docs/contracts/
  described it. A versioned artifact with no contract is a version string that means nothing.

## Summary
The mint preflight's evidence JSON. One file per invocation, written in a `finally` so it
ALWAYS lands (LAW-14); an unwritable report is a loud fatal, never a silent skip. Named
`preflight_{run_id}_{ts}.json`. It is EVIDENCE, not a gate result: what it may claim is bounded
by what the run actually DID, and every "not proven" sentence in it is derived from the run's
own history rather than from its intent.

Two invocation modes share one artifact shape: `audit` (CI gate 12, `--audit-only` — a static
manifest audit, no boot) and `preflight` (MANUAL, a real boot plus a burst). There is NO default
mode: an unknown mode is a named internal failure, because falling back would publish some other
mode's disclaimer.

## Shape

Top-level keys, all present in every report (`_new_report`):

| key | meaning |
|---|---|
| `schema` | `"preflight-mint-v1"` |
| `tool_sha256` | sha256 of the tool file that wrote it |
| `ts_utc` | ISO-8601 Z write timestamp; also the second half of the filename |
| `mode` | `"audit"` or `"preflight"` |
| `verdict`, `rc`, `failure` | the run's own outcome; a verdict that was REACHED is never overwritten by a disclaimer |
| `config` | the audited config's identity: `path`, `sha256`, `run_id`, `encoding`, `representation` |
| `coordinator` | the RESOLVED coordinator knobs, drain caps, `stop_step` and draw-rate terms, derived from the shipped resolvers and never restated |
| `override` | the burst override actually applied; its `keys` are emitted from `OVERRIDE_KEYS`, so the tool cannot become a second run-length authority |
| `manifest` | the armed-abort manifest rows this run audited, `required` and `deferred` |
| `assertions` | `a_sync`, `b_lag`, `c_arming` — each a `{verdict, reason}` block |
| `child` | the boot child's rc, stderr tail and identity, or `null` when none was spawned |
| `events` | the event segments the post-child scan actually covered |
| `tier` | which mint tier the ACCEPTED burst belongs to, and what it does not prove |

## Who asserts what where

| assertion | where |
|---|---|
| the report is written on every path, including every failure path; an unwritable report is rc 41 and fatal (LAW-14) | tools/ci_gates/preflight_mint.py (`_write_report`, `PreflightReportUnwritableError`) |
| no report on disk claims a boot its own `child` block does not record | tools/ci_gates/preflight_mint.py (`_finalise_not_run`, run at write time, not at the call site) |
| the `not_run` reason is DERIVED from what the run did (booted vs not), never from what it intended | tools/ci_gates/preflight_mint.py (`_not_run_reason`, `NOT_BOOTED_REASON` / `BOOTED_REASON`) |
| in `audit` mode, assertions (a) and (b) can never read as green — a green rc 0 covers (c) ONLY | tools/ci_gates/preflight_mint.py (`AUDIT_STDOUT_LINE`, the `not_run` skeleton) |
| an unknown mode is REFUSED and never defaulted | tools/ci_gates/preflight_mint.py (`REPORT_MODES`, no fallback) |
| the mint TIER is derived from the config's own `_burst_floors` rows and from the burst the validators ACCEPTED — `--burst-steps` is a request, not a tier | tools/ci_gates/preflight_mint.py (`_burst_tier`, `DRAW_RATE_FLOOR_KEY`) |
| every tier has a "does not prove" sentence and there is NO default entry | tools/ci_gates/preflight_mint.py (`TIER_NOT_PROVEN`, keyed by `TIER_NONE`/`TIER_SYNC_LAG`/`TIER_FULL`) |
| both `sync_lag` and `full` are required for a mint, and `full` covers `sync_lag` — the covered/owed split is published rather than left to a reader to infer | tools/ci_gates/preflight_mint.py (`MINT_REQUIRED_TIERS`, `_tier_covered`) |
| the tier disclaimer is RE-DERIVED at write time and printed FROM the finalised report, so the sentence on the terminal is byte-identical to the one on disk | tools/ci_gates/preflight_mint.py (`_finalise_tier`, `_write_report`) |
| the child's own named outcomes propagate unchanged (rc 10–41); the reserved band 42–46 is DERIVED from the watchdog / relaunch / armed-abort authorities, never re-typed | tools/ci_gates/preflight_mint.py (`PASS_THROUGH`, `RESERVED_CODES`) |

## Pinning tests

| test | file |
|---|---|
| an audit report can never read as green for the dynamic assertions | tests/tools/test_preflight_mint.py |
| the report publishes the pins the scan ACTUALLY covered, the RESOLVED coordinator config, and the audit's own required/deferred rows | tests/tools/test_preflight_mint_process.py |
| a writable out-dir still writes exactly ONE report; a report with no config block is still NAMED | tests/tools/test_preflight_mint_process.py |
| the `not_run` reason names the mode, is derived from the report's own `child` block, and a reached verdict is never overwritten | tests/tools/test_preflight_mint_process.py |
| an unknown report mode is refused and never defaulted | tests/tools/test_preflight_mint_process.py |
| every mint tier has a NOT-PROVEN entry with no default; the tier is derived from the config's own floor rows; a production config can never be preflighted in the short tier | tests/tools/test_preflight_mint_process.py |
| a tier is COVERED only when the run reached a verdict; a refused burst publishes `tier: none` and owes BOTH tiers | tests/tools/test_preflight_mint_process.py |
| the disclaimer is re-derived at write time and the `none`-tier sentence is TRUE in mode AUDIT, not only at rc 11 | tests/tools/test_preflight_mint_process.py |

## Former gap F-B1 — CLOSED (WPCLEAN Phase RES), residual disclosed

The report now witnesses WHICH config the boot child loaded: `compose_run` publishes a
`run_boot_identity` event (the child's own post-revalidation config sha) into the JSONL
segment the parent already scans, both sides hashing through the ONE authority
(`mantis.config.loader.config_identity_sha256`). The parent copies it into
`child.booted_config_sha256` and verdicts `child.config_identity` as `match` /
`mismatch` / `unwitnessed`; a MISMATCH is a NAMED failure (`PreflightConfigIdentityError`,
rc 14) raised before the predicates — a burst on the wrong config proves nothing.

Residual, disclosed rather than papered over: a child that dies BEFORE its sink exists
publishes nothing and the report says `unwitnessed`, never a silently-assumed match — the
same reached-vs-assumed distinction the `tier` block keys on (it still keys off the
report's own outcome fields rather than the requested `--burst-steps`).
