# LAWS — the standing laws of mantis

Read this first. Each law was bought by a measured failure; the earned mechanism and the
pre-R346 wording are in docs/governance/archive/laws.md. Changing or dropping a law takes an
amendment commit and operator sign-off.

## The seventeen

- LAW-01 Prime directive. Context first, measurement mandatory: no architectural change
  without reading the design docs and the source, no perf-sensitive change without a bench.
- LAW-02 Re-validation. Never drop a candidate driver or a fix on a falsified row or a banked
  prior without testing that the prior's context transfers. Retested fresh, one driver INVERTED.
- LAW-03 Measurement-unit. Verify a founding measurement's unit before framing on it — turn
  vs ply (a compound turn places TWO stones), and WHICH cell of a multi-stone win completes it.
- LAW-04 Effective-n. A strength CI counts DISTINCT games (trajectory-hash dedupe), not the
  game count; argmax/temp-0 regimes collapse to ~2 distinct games per pairing.
- LAW-05 Falsified-register-first. Read docs/governance/falsified.md before proposing any optimization or experiment, and apply LAW-02 when citing a row from it.
- LAW-06 bf16-graph. Graph-path autocast dtype is bf16, pinned in code and by a parity test.
- LAW-07 Producer-test. No gate or monitor input without a live producer test, and the checker
  carries a mutation self-test proving it bites.
- LAW-08 Live-consumer. Every config key and every registered encoding has a live consumer; a dead knob dies with its freeze-tests in one commit.
- LAW-09 Bench discipline. Pre-registered hotspots + expected-gain bracket + abort threshold;
  one change = one commit = one IQR-gated bench; profile first; a measured floor is a finding.
- LAW-10 DELETED by R347(d) — grid-era, no producer, gating nothing. The number is retired, not
  reused: every other law keeps the number it was cited by.
- LAW-11 Identity-keys. No dense-by-default anywhere. An absent encoding/representation is an
  error, never a default; representation is a closed enum on both sides of the FFI.
- LAW-12 Checkpoint-stamp. Stamps are written once and immutable, never re-stamped from a
  loaded config; one loader; weights-only strip is the one sanctioned encoding-change path.
- LAW-13 FFI/build. panic = "unwind" so a panic crosses the FFI catchable, never as a process abort; no target-cpu in committed build config.
- LAW-14 Persistence-fatal. Persistence failures are run-fatal; `except Exception: pass` is
  lint-banned; an optional effect goes through best_effort() and requires a counter.
- LAW-15 Eval-instrument. Deploy-matched eval is the DEFAULT promotion bar and a missing deploy
  decision blocks promotion; strength bars are fixed-depth instruments, never wall-clock.
- LAW-16 Lifecycle. One subsystem, contract-tested: signals save-then-exit, self-play stall watchdog ALWAYS armed, disk guard.
- LAW-17 Structure. Zero sys.path writes; one tests/ collection root; pyo3 only in the bridge;
  configs explicit and complete; a >300-line file justifies itself and states no line count.
- LAW-18 In-run observability. A lever under test logs its own fire-rate in-run — a post-hoc probe cannot tell "starved" from "ineffective".

## The protected set

R346 §1(d): this list is the whole of what ruling-protected code means from now on. Each item
exists because a measured failure bought it. None may be weakened, disarmed, narrowed or
deleted except by a ruling that names it.

- net-param hash on the warm-start
- served-sims exactness
- the suite's conformance sections
- 1-in-1 collate checks
- arena legality
- finite-gradient guard
- resume bundle round-trip
- gate pair statistics
- F-816-37 dump-on-fire
- strength_floor
- draw-rate abort
