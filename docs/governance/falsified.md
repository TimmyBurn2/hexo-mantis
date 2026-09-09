# Falsified register (curated)

Rows curated at migration time from the predecessor project's consolidated register;
forensic archives live in the private records archive. DO NOT re-litigate a row without
the re-validation protocol (LAW-02): falsifications are objective- and regime-specific —
cite the row, state its context, test transfer, only then keep/drop. Dates are original
falsification dates.

| id | hypothesis | falsified by | mechanism (curated) |
|---|---|---|---|
| F-01 | Hex-native trunk variant closes the self-play gap (2026-05-05) | MCTS-matched eval | Probe gates passed but selfplay dropped to 0–1% SealBot WR. Static probes cannot validate dynamic equivariance — only MCTS-matched eval can. |
| F-02 | Super-additive interaction of 5 smoke MCTS+exploration knobs drives 91% draws | knob-isolation follow-up | Cosine temperature alone is load-bearing (~5% → ~91% draws); dirichlet / opening plies / playout cap are synergy partners, not drivers. |
| F-03 | Padding semantics (canvas-realness + partial convolution) recovers bbox direction | ablation eval | Trains below the anchor's loss yet 0% SealBot WR; bbox direction is structural: K-aggregation as cross-cluster contrast, bbox-centroid frame instability, perception radius. |
| F-04 | A learned PMA pooling layer can replace K-cluster min/max aggregation | ablation matrix | 4.5% / 7.5% WR vs 14.5% for K-cluster min/max; min/max canonical. SCOPE (WPSC/R27): falsifies PMA-as-tested vs min/max ONLY — does NOT certify min vs mean+max→attention (run3_findings_v2 §4.1). The min/max asymmetry — value aggregation takes the worst cluster view while policy takes the best-scoring view (`aggregate_cluster_values_min`, runner/search_drive.rs) — is a flagged defect preserved pending the matched-FLOP dense arm; see the CARD-MINPIN pinning test + registry.toml `value_pool = "min"` comments. |
| F-05 | Gpool-bias as a global lever for both policy and value heads | follow-up ablation | Policy-only gpool-bias is the load-bearing mechanism; full gpool-bias is NULL on value. |
| F-06 | The v6 corpus is human-quality (uniform-weight bot mix does not contaminate) | corpus audit | ~41% bot games at source_weight=1.0; Elo weighting had degenerated to uniform via choice-on-uniform-weights. Human-only Elo-weighted corpus is canonical. |
| F-07 | More pretrain epochs improve self-play | e50-vs-e30 closeout | e50 selfplay regressed vs e30 (median plies 12 vs 17; gate marginal-fail). The value head over-fits corpus-mode signal that selfplay cannot reproduce. |
| F-08 | Legal-move radius compression at bootstrap fixes the dense-lineage selfplay collapse | R=1..R=8 smoke | Median plies identical across all radii; radius does not move bootstrap quality. |
| F-09 | The dense-lineage selfplay collapse is a bootstrap recipe issue | closeout | Loss surface normal; opening-fraction starvation refuted. The collapse lives at the argmax-degeneracy / selfplay-interaction layer, not corpus/loss. |
| F-10 | Dirichlet root noise was active on the training path | post-migration audit | Silently unported during an engine migration → 16,880 steps of carbon-copy self-play (mode collapse). Ported features need producer tests (LAW-07). |
| F-11 | FP16 AMP is numerically robust on aux losses | NaN incident | 0×−inf cascade in aux CE → NaN total loss, BN poisoning. Log-clamp + exact-entropy fix; graph amp policy is bf16 (LAW-06). |
| F-12 | Promoted weights are the evaluated weights | promotion audit | Allocator reuse made every graduation commit unvalidated weights as anchor. Provenance for the checkpoint-stamp law (LAW-12). |
| F-13 | The padding-class collapse is a broadcast-scalar-plane dependency | targeted probe | Spatial pathway not dead; the collapse is structural at K=1 inference. |
| F-14 | 18-plane input dimensionality is load-bearing | plane ablation | An 8-of-18 plane subset suffices; chain features moved to an aux sub-buffer. |
| F-15 | MCTS expansion-time forced-win short-circuit accelerates training | removed pre-baseline | The network never evaluated near-win positions → no fork learning. Quiescence value-override at leaf eval is the correct alternative. |
| F-16 | Distribution-shift fine-tune over a 5% adversarial corpus (frozen spine) recovers MCTS signal | reopen closeout | MCTS-64 0/200, Wilson95 [0%, 1.88%] — DEAD bin cleanly met; the frozen-spine class is closed. |
| F-17 | A sorted-Vec representation of the legal-move set beats the hash-set rebuild (2026-05) | bench | −32.5% sims/s. The ring loop pushes ~7× duplicate cells (overlapping radius balls); sort+dedup on the bloated array costs more than hash-with-inline-dedup insert. |
| F-18 | The residual legal-move-set self-time is lookup-dominated | flamegraph | Insert is dominant (56.8% vs 27.7% lookup). The prior fix failed by fix-design error, not by the assumed lookup mechanism. |
| F-19 | Incremental legal-coverage delta maintenance amortizes below the once-per-leaf rebuild | bench | −49.5% sims/s. The delta runs per descent STEP (apply × depth + undo × 2·depth), not per leaf — de-amortized to ~3× the rebuild's work on the hot path. The residual cost is a structural floor. Corollary in perf doctrine: build-once-per-leaf beats incremental deltas on descent paths. |
| F-20 | The 150k+ stall is a colony-attractor / off-window divergence (2026-06-24) | fixed-depth opponent reproduction | The trajectory FLIPPED between a wall-clock-limited bar and a reproducible fixed-depth bar → opponent-instance artifact, not a model off-window defect. The single-axis intransitivity is real but bar-dependent. Endpoint verdict: true stall (deploy-matched self-ladder flat). Bars must be reproducible instruments (LAW-15). |
| F-21 | A custom CUDA kernel (borrowed from a sibling GNN project) would speed the dense forward path (2026-07-02) | verdict + red-team | The kernel solves GNN variable-size ragged batching — a problem the dense CNN+attention path does not have (multi-window batching varies batch COUNT, not tensor SHAPE; the standard variable-N case is already handled by stock kernels). If forward throughput binds: torch.compile → smaller net → quantized eval, in that order. SCOPE: kernel/ragged-batch-perf only — this row does NOT adjudicate the axis-graph representation (separately re-opened and since adopted as the GNN lineage). |
| F-22 | Bot-corpus share 0.15 + ply-cap-value split + cosine-off is a sufficient anti-colony lever for stable training to ≥50k steps (2026-05-20) | eval trajectory | SealBot WR 8→11→12→2→2→4; colony@SealBot pinned 100% from step 40k. Buys ~1 extra promotion; does not escape the attractor. |
| F-23 | Visit-count CE policy targets (completed-Q off) escape the colony attractor (2026-05-20) | eval trajectory | Uniformly weaker at step 20k (WR −4/−15/−18pp) — not capture, slower learning of the same trapped state. Completed-Q ruled out as a colony lever. |
| F-24 | A 3-knob escalation (completed-Q on + bot share 0.30 + uniform game-length weights) supplies enough direct anti-colony force (2026-05-21) | eval trajectory | Every config-visible colony metric crushed (selfplay colony ~0.04%) yet SealBot WR still collapsed 19%→0% at 50k. The capture channel is config-invisible. |
| F-25 | A step-20k anchor swap escapes the colony attractor under the same recipe | reproduction on the new anchor | 2→0% across 10k→40k reproduces the original 18→4% decline on a different anchor — recipe-level, not anchor-level. |
| F-26 | A static bot corpus alone is a sufficient anti-colony anchor past the peak-fit point | sustained wave | Held the attractor to step ~20k; past it the policy drifted off the corpus distribution and the 30% bot batch share became off-distribution noise. Dynamic regeneration against the current model is required. |
| F-27 | Alt value-spread + dual-bank canary alone is a sufficient run-quality gate | sustained wave | The canary stayed comfortably above its sustained gate for 46k steps WHILE SealBot WR collapsed 33%→5%. Value-head discrimination on fixed banks is not a proxy for selfplay/eval performance; gates must include a WR sliding-window trajectory (LAW-07 provenance). |
| F-28 | Refresh hook + per-class temperature scope flip prevents attractor capture | sustained wave | WR collapsed to 2% by step 45k with both levers active. |
| F-29 | A long plateau (sustained mid-WR phase) is a positive sign of attractor break | sustained wave | Plateaued 16–25% WR across 10k–30k, then catastrophically collapsed to 2%. |
| F-30 | best-model promotion is a reliable signal of improvement toward readiness | sustained wave | best-model promoted AT step 45k (wr_best 69%) WHILE external WR crashed to 2%. The promotion gate rewards anchor-exploit, not external strength (LAW-15 provenance). |
| F-31 | Bot mix is the load-bearing failure variable in the colony-attractor mechanism (2026-05-27) | ablation track | Removing bot mix produced FASTER decline. |
| F-32 | Multi-aux density (diverse aux signal) prevents single-attractor lock (2026-05-27) | ablation track | Delayed the attractor ~5k steps; did not prevent it. |
| F-33 | Bootstrap weights + corpus jointly encode a colony bias before self-play | structural bias audit | Value-head delta wrong-signed and n.s.; policy plays the winning extension; corpus winning lines are extension-dominated. No pre-selfplay bias exists — the attractor is GENERATED by the training loop. |
| F-34 | MCTS search parameters (c_puct, dirichlet) are a viable anti-colony escape lever | search-dynamics audit | c_puct ×0.5/×2.0 moves colony-visit fraction <6pp; dirichlet ×4 <3pp — inside noise. Search neither amplifies nor corrects the bias; it faithfully passes through a biased value/policy head. |
| F-35 | The value blind-spot is a TARGET problem — full-spectrum class-balanced distillation on existing features separates wins from losses (2026-06-26) | distillation gates | Separation gate craters (six estimates 0.32–0.46 vs ≥0.85); in-sample fit rules out under-power. Light-trunk unfreeze also fails. The features CONFLATE win/loss in the blind-spot neighborhood once the turn-phase shortcut is controlled → FEATURE problem, not target. |
| F-36 | Richer input threat planes carry the missing win/loss signal and justify a restart (2026-06-26) | controlled feature experiment | 4-plane control vs 8-plane treatment: both crater; a linear probe on the threat summaries ceilings at held-out AUC 0.646. The bottleneck is not cheap input features. |
| F-37 | Win/loss is recoverable from the frozen representation by some readout or a richer search-derived value target (2026-06-26) | probe battery | Flexible pre-pool probes do not beat the pooled probe (pooling discards nothing recoverable); a validated richer continuous target distilled into the frozen head craters wins. Signal is decision-relevantly ABSENT. Convergent close: the cheap-lever space (target ∧ trunk ∧ features ∧ readout) is exhausted; remaining levers are architectural (deeper net / search-in-the-loop). |
| F-38 | The deep value-blind losses need the expensive frontier (deeper/wider net or MuZero-class) (2026-06-26) | bounded-search existence proof | A bounded minimax proves ALL 33 value-blind proven-core losses at depth 6–8 turns → they are bounded-tactical game-tree properties; the deeper-net frontier is NOT earned. The lever is search-in-the-loop. |
| F-39 | A cheap tactical add (one-primitive threat-space search) flips ≥40% of the traps at <10× deploy node budget (2026-06-26) | probe + red-team | Flips 3/38 = 8% (all mate-in-2; mid/deep 0%). BROADENING the candidate set REGRESSES to 0/38 at ~100× cost (wider branching collapses reachable depth). A competent pattern-guided minimax is required, not a one-primitive probe; deploy solver-backup is affordable only for the short band. |
| F-40 | Policy-guided candidate ordering with the current net lifts the cheap-search flip rate (2026-06-26) | guided probe | 0/14 flips — the net is ~0-prior-blind on exactly the refuting moves. Policy guidance helps only AFTER search-in-the-loop training; the exceedance route runs through the bootstrap, not around it. |
| F-41 | `i32::midpoint(a, b)` is byte-identical to `(a + b) / 2` for signed integers | review catch | midpoint rounds toward −∞, integer division truncates toward 0; they differ by 1 when the sum is negative-odd. Watch carried on ported midpoint sites. |
| F-42 | A compiled extension's classes report the extension module's name in `__module__` by default | smoke assertion | PyO3's default `__module__` is `'builtins'` regardless of module placement; assert accordingly or set the module name explicitly. |
| F-43 | "run5 CPU twin produces zero self-play games" (the R207/R209 premise) AND "the R199/R200 zero-games signature (game_complete:0, training_step:0, inference_dispatch age≈wall, buffer_size:0) proves no games are produced" (2026-08-04) | R209 bisect intermediate 1 + R210 | FALSIFIED on the CPU path: intermediate 1 (n_simulations=2, log_interval=1000) produced actor_sync=13, learner_step=12 — games WERE produced and training stepped — but iteration_complete:0 because `train.log_interval=1000` gates `iteration_complete` via `_run_log_interval`'s `self._train_step % cfg.log_interval != 0` early return (`coordinator/step.py:576`); `iteration_complete.games_total` is gated by log_interval and NOT production-visible before step 1000 (R210). The production-visible games signal is `actor_sync` (trainer stepping ⇒ buffer has data ⇒ games produced), NOT `iteration_complete.games_total` (gated) and NOT `game_complete` (dropped — see next). `game_complete` IS emitted at `pool_drain.py:177` (golden-pinned, C-03 test_pool_drain_parity.py:332-352 + J-05 test_selfplay_census.py:398-421) but DROPPED in production because the production `WorkerPool` is constructed with `sink=None` at `run.py:349` (R215 corrects R207's "never emitted" attribution). `inference_dispatch`/`selfplay_drain` are phantom sources that never tick in healthy runs (R207/R208); `buffer_size:0` in `system_stats` is caught between training bursts in O4 warmup. The entire "zero-games" signature is what an invisible-but-healthy run5 looks like. BOX HALF RESOLVED (R212, ratified R220): the diagnostic box burst on the RTX 5080 (log_interval=10, /tmp-class config `/tmp/run5_box_r212_diag.yaml`, out-dir `/tmp/wp12r_8b_r212_box_measure`) MEASURED `actor_sync=5 > 0` on the box's GPU path — box PRODUCES GAMES. R212 verdict rule fired verbatim (`games_total>0 OR actor_sync>0 ⇒ box produces games, ADJ-ZERO-GAMES closes as "invisible-games" on both paths`); `iteration_complete` remained 0 even at log_interval=10 on the box, confirming the R210 decoupling defect exists on BOTH paths (narration chunk scope-(i) fix required on both, as R214 stated). The "zero-games → invisible-games" re-frame is now UNCONDITIONAL on both paths (R220 — the condition R212 placed on it is satisfied); the box was never zero-games, it was invisible-games (log_interval gating + sink=None drop, R215/R216). No second distinct box defect exists alongside the invisibility defect. [R215 mechanism correction of record: R207's "game_complete is never emitted to the sink" → "emitted (pool_drain.py:177, golden-pinned) but dropped because pool._sink=None at run.py:349"; "games live in iteration_complete.games_total" → "gated by log_interval, not production-visible; production-visible games signal is actor_sync".] [R212/R220 instance-ledger entry: what = R212 box-half measurement (box produces games, actor_sync=5 > 0); who = dispatch 8B via R212 box burst; how caught = diagnostic box burst at log_interval=10 reading actor_sync from the JSONL event stream; where corrected = falsified.md F-43, STATE §2A/§7, ADJUDICATION_QUEUE.] |

## Scope annotations (append-only; the rows above are never edited)

A grave is scoped to what its bench actually measured. These notes narrow nothing and widen
nothing in the rows themselves — they record, beside the row, which neighbouring hypotheses the
measurement did NOT reach, so a later reader does not have to infer the boundary and guess wide.
Added under LAW-02's re-validation discipline: the fence is a re-litigation TAX, never a
prohibition, and citing the row + stating the regime + testing transfer is the way through it.
Each note names the ruling that authorised it; none of them re-opens its row.

- **F-17 — scope, per R288(e) (2026-08-20).** The row is scoped to its MEASURED subject:
  **legal-set maintenance on descent paths, in the dense-era regime** the bench ran in. What the
  measurement did not reach, and what this row therefore does not adjudicate: per-unique-node
  **eval caching**, **root-level incremental updates** between actual moves (amortized over a
  whole search rather than per descent step), **transposition reuse**, and **incremental
  axis-graph construction from the parent position**. The falsified mechanism is specific — the
  ring loop pushes duplicate cells, so sort+dedup on the bloated array loses to
  hash-with-inline-dedup — and it transfers only to a candidate that shares it.

- **F-19 — scope, per R288(e) (2026-08-20).** Same scoping as F-17, and for the same reason: the
  row measured **incremental legal-coverage delta maintenance on descent paths** against a
  once-per-leaf rebuild, in the dense era. Its corollary line — *build-once-per-leaf beats
  incremental deltas on descent paths* — is a statement about **that** artifact, whose per-leaf
  build is cheap; it is not a general ban on incrementality. Explicitly NOT covered: eval caching,
  root-level increments, transposition reuse, and **incremental axis-graph construction from the
  parent position** — the last of which is registered as **candidate INCR-GRAPH**, which the
  governance register carries in its perf lane. INCR-GRAPH is a candidate, not a plan: it is
  **gated on the Rust-criterion box measurement at run5 shape** (nothing is designed before that
  number exists — LAW-01), and its **pre-registered falsifier is this row's own inequality**,
  evaluated at run5's measured depth distribution: the delta wins only where
  `delta_cost × depth < build_cost`, which is exactly the arithmetic that killed the legal-set
  case (a tiny build_cost) and is an open question only where the per-leaf build is expensive.

- **F-43 — mechanism annotation, per R335(a) (2026-09-04).** The row's FINDING is untouched: the
  zero-games signature was an INVISIBLE-games signature on both paths, and that stands. What has
  moved is ONE of the two mechanisms that made a healthy run invisible. R215's correction of
  record reads *"dropped because `pool._sink=None` at `run.py:349`"*; at `736c4b5` both pool
  constructions pass `sink=_DeferredSink()` (`src/mantis/run.py:528, 538`) — a late-binding
  adapter bound in `compose_run` before `pool.start()` — so **`game_complete`, with its
  `moves_list`, reaches the stream in production today**. The OTHER mechanism, `log_interval`
  gating `iteration_complete` (R210), is NOT claimed to have moved and was not re-measured here.
  Nothing in the row is re-opened: this note records that a reader citing the `sink=None` half as
  live repo state would be citing something that is no longer true, which is exactly the drift an
  append-only register corrects by annotation rather than by edit (R9).
