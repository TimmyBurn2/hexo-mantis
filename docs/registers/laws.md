# Operating laws (curated register)

Curated at migration time from the predecessor project's operating laws; the forensic
evidence chain lives in the private records archive (same convention as
docs/design/repo_design.md's provenance note). Each law carries the mechanism that earned
it. Changing or dropping a law requires an amendment commit + operator sign-off.
Companion register: docs/registers/falsified.md — read it BEFORE proposing any
optimization or experiment (LAW-05).

- **LAW-01 Prime directive.** Context first, measurement mandatory. Never propose an
  architectural change without reading the relevant design docs and source. Never land a
  performance-sensitive change (search, NN, buffers, encode) without a bench against the
  current baseline.
- **LAW-02 Re-validation discipline.** Falsifications are objective- and regime-specific.
  Never drop a candidate driver or fix by citing a falsified-register row, banked result, or
  prior finding without re-validating that the prior's context transfers. Protocol: cite the
  prior → state the exact context it was falsified in → test whether that context transfers
  to the current objective → only then keep/drop. A drop resting on an un-re-validated prior
  is rejected. (Earned: in one investigated arc every candidate driver had a tempting prior
  to drop it on; tested fresh, one driver INVERTED its prior.)
- **LAW-03 Measurement-unit law.** Verify the unit of a founding measurement before building
  a frame on it — especially turn-vs-ply (a compound turn places TWO stones; depth-1
  single-stone units undercount turn wins) and which cell of a multi-stone win is
  reachability-relevant (the COMPLETING cell that lands the win, not the first stone).
  A one-cell mislabel once mis-routed a multi-week investigation. Corollaries: a BORDERLINE
  retraction earns a cheap eval-only discriminator before any expensive lever; an
  inference/self-play lift is a necessary-condition probe — name the external kill link as an
  explicit OPEN gate and pick an instrument that cannot false-clear by construction.
- **LAW-04 Effective-n law.** A strength CI's effective sample size is the number of
  DISTINCT games (trajectory-hash dedupe), not the game count. Deterministic regimes
  (argmax / temp-0 from a fixed opening) collapse to ~2 games per pairing; a CI over the raw
  count is over-confident by sqrt(copies). Dedupe and bootstrap over distinct games before
  trusting any "CI-resolved" gap; opening/opponent DIVERSITY is as load-bearing as
  temperature under argmax regimes.
- **LAW-05 Falsified-register-first.** Read docs/registers/falsified.md before proposing any
  optimization or experiment; apply LAW-02 when citing it.
- **LAW-06 bf16-graph law.** Graph-path autocast dtype is bf16 — pinned in code and by a
  regime-parity test. (fp16 AMP produced a 0×−inf cascade in aux CE losses → NaN total loss
  and BN poisoning; see falsified register.)
- **LAW-07 Producer-test law.** No gate or monitor input without a live producer test: every
  consumer binding — display OR headless gate input — cites a live producer, and the checker
  carries a mutation self-test proving it bites. (Earned by a phantom gate input that armed
  an abort chain no producer ever fed, and by a healthy-looking canary that stayed green
  through a real collapse.)
- **LAW-08 Live-consumer law.** Every config key and every registered encoding has a live
  consumer (test-enforced); dead knobs are deleted together with their freeze-tests in one
  commit.
- **LAW-09 Bench discipline.** One optimization = pre-registered hotspot list + expected
  gain bracket + abort threshold. One change = one commit = one IQR-gated bench; parity
  oracles re-run after every hot-path change; measure the end-to-end metric, not only the
  microbench. A single-run regression without a code mechanism on the touched path requires
  fresh-bench triangulation before any verdict. A measured structural floor is a finding,
  not a failure. Profile first (flamegraph / py-spy; DHAT for allocation-rate hunting);
  profiling builds = release + debug symbols (`profiling` profile).
- **LAW-10 Threat-probe criterion.** The threat-logit probe gates each 5k-step checkpoint
  and any pre-promotion checkpoint: C1 contrast_mean ≥ max(0.38, 0.8 × baseline contrast);
  C2 ext_in_top5_pct ≥ 25; C3 ext_in_top10_pct ≥ 40; C1–C3 must all PASS. C4
  |ext_logit_mean drift| < 5.0 is warning-only (BCE-drift canary). For a resumed fine-tune,
  gate against the run's OWN warm-start anchor baseline, not a generic bootstrap baseline —
  a bootstrap-gated floor was measured to be nearly vacuous against a 12×-sharper anchor.
  (Numeric thresholds re-anchor when the probe ports; the criterion structure is the law.)
- **LAW-11 Identity-keys law.** No dense-by-default anywhere. An absent
  encoding/representation is an error, never a default. Representation is a closed enum on
  both sides of the FFI; no wildcard match arm on the kind.
- **LAW-12 Checkpoint-stamp law.** Stamps are written once at creation and are immutable —
  never re-stamped from a loaded config. Artifact filenames carry run-id + content hash; an
  artifact that cannot be stamped cannot be written (save fails loud; quarantine if the run
  must survive). One loader, shared by all surfaces; weights-only strip is the one
  sanctioned encoding-change path, wire-signature-gated. A fine-tune that must pin its own
  LR requires a weights-only warm-start: a full-checkpoint resume silently inherits the
  source's scheduler state (measured incident).
- **LAW-13 FFI/build law.** The workspace release profile uses `panic = "unwind"` — Rust
  panics cross the FFI as a catchable exception, never a process abort. No
  `target-cpu=native` in committed build config; native is an opt-in env flag
  (`make build.native`); built artifacts are portable by default.
- **LAW-14 Persistence-fatal law.** Persistence failures are run-fatal by default:
  event-sink and buffer save/restore errors increment `persist_errors_total` and the
  watchdog aborts on it. `except Exception: pass` is lint-banned; optional effects go
  through a `best_effort()` wrapper that requires a counter. A replay buffer must be
  explicitly restored or guarded by a real prefill gate — a cold buffer under a tiny
  default minimum trains on ~nothing for an entire run (measured incident).
- **LAW-15 Eval-instrument law.** Deploy-matched eval is the DEFAULT promotion bar; a
  missing deploy decision blocks promotion, never falls back to a proxy regime. Strength
  claims ship protocol + n + eff_n + per-side compute. Opening books are versioned,
  sha-pinned, paired; CI on pairs is a bootstrap percentile. Strength bars must be
  reproducible instruments (fixed-depth, not wall-clock — a wall-clock bar once flipped a
  verdict that a fixed-depth bar reversed).
- **LAW-16 Lifecycle law.** Lifecycle is one subsystem: SIGINT/SIGTERM → save-then-exit
  (second signal force-exits), self-play stall watchdog ALWAYS armed, disk guard.
  Contract-tested.
- **LAW-17 Structure laws.** Zero `sys.path` writes anywhere. Single test-collection root
  `tests/`; no directory named `tests` below it. No `#[pyclass]`/`#[pymethods]`/
  `#[pyfunction]` outside the bridge crate. Configs are explicit and complete — no
  inheritance, no base-merge, no code-side defaults; a default lives in exactly one place:
  the schema field. Files >300 lines carry a justification at the top stating why the file is
  one unit; the justification never states a line count (G-DFIX-4 / R192(e) derive-or-delete —
  a transcribed tally goes stale and is then read as evidence).
- **LAW-18 In-run observability law.** A lever under test must log its own fire-rate in-run;
  a post-hoc offline probe cannot distinguish "starved" from "ineffective" (measured
  incident: a density lever's null read was unreadable without in-run counters).
