# Contract: eval instrument

- version: v1
- owner: mantis.arena
- status: LIVE. <!-- AUDIT-1 F-52: this read "SKELETON — contract text lands with the
  subsystem port" over a filled contract, beside a shipped eval subsystem. The label is
  decision-adjacent: a reader deciding whether the deploy-matched bar is specified would
  have concluded it is not. -->

## Summary
deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games

## Who asserts what where

This section is the DURABLE instrument: what each term means, and which module is the single
authority for it. The run-specific choices — which rungs, which numbers, which honesty
clauses — live once in `docs/contracts/eval_decision_run5.md` and are not restated here.

- **Deploy-matched** means both sides of a comparison are built by the SAME player
  constructor at the SAME simulation count. `mantis.eval.worker` builds the candidate and the
  best snapshot through one `_build_candidate_player` call site at `eval.gate.deploy_sims`,
  and the `RegimeKey` stamped on each record carries the value that was actually used. A
  ladder entry is deploy-matched to its own per-kind simulation count, never to the gate's —
  playing at one value while stamping another is a mislabelled record, not a rounding
  difference.
- **The fixed-depth external bar.** An external opponent whose strength axis is SEARCH DEPTH,
  not simulations, is only an instrument if the depth it plays is the depth it claims. The
  adapter (`mantis.bots.sealbot`) therefore drives the engine's depth ceiling, neutralises its
  wall-clock cut with a value that provably cannot be reached inside a game, and reads the
  reached depth back as a RECEIPT after every move. Register row F-20 is why this is a receipt
  and not an assurance.

  **The receipt models the engine's own legitimate short searches, not a proxy for them.** A
  search that stops early because the engine proved a win, proved a LOSS, or found a mate
  inside its threshold band is correct, and so is one that never ran because the position was
  empty. The receipt reads those conditions from the engine — a threshold constant exported by
  the tracked vendor patch, and the engine's own reported score — rather than reconstructing
  them from the board. Reconstructing them is strictly narrower and was measured to reject
  correct play in every game it was tried on.

  **What a genuine violation costs, stated exactly, because this document outlives the work
  package that wrote it:** it raises `SealBotDepthError`, and
  `src/mantis/eval/worker.py:349-356` catches `RungUnresolvable` and nothing else — so the
  exception ends the **whole eval round**, not the one ladder entry. There is no per-entry
  "recorded broken" mechanism today; an earlier draft of this section claimed one and no such
  code exists. The trade is deliberate in this direction only: a round that dies loudly is
  recoverable, and an entry that quietly reports a bar it never played is not.
- **Vendoring, and the ONE build command.** External engines are pinned by commit sha in
  `vendor/pins.toml` and fetched by `make vendor`, which CLONES and does not build. The build
  is a separate, manual step and it must use mantis's OWN interpreter or the extension's ABI
  will not match the process that imports it:

  ```
  uv run --with pybind11 --with setuptools python setup.py build_ext --inplace
  ```

  run inside `vendor/external/sealbot/current/`. `--with` is ephemeral: a vendor build
  dependency never becomes a mantis dependency. The refusal reason a rung emits when the
  extension is absent names this command verbatim, so the log says which step to run.
- **Books** are versioned, sha-pinned and paired: `mantis.arena.books` verifies the sha256 at
  load and raises on mismatch, and every opening is played exactly twice with the colours
  swapped, so a colour advantage cancels within the pair rather than across the sample.
- **eff_n is distinct games** (LAW-04): `mantis.eval.aggregate` dedupes on the trajectory
  hash before any interval is computed, and the low-power guard counts distinct games PER
  PAIR. Games are not evidence; distinct trajectories are.
- **The interval is a bootstrap percentile** over those distinct outcomes, seeded from the
  gate seed (gate blocks) or the ladder bootstrap seed (ladder blocks). An empty sample
  degenerates to an absent interval rather than raising, so a zero-game block cannot
  manufacture a bound.
- **A rung that cannot resolve SKIPS LOUD, on four channels**, and the fourth is the one that
  answers the question the first three cannot: an `eval_rung_skipped` event, an ERROR log
  line, the round result's own skip list, and — per LAW-18 as R164 reads it — an
  `eval_rung_skip_class` counter event emitted alongside each skip, partitioning the reasons
  into a CLOSED set so "these are the skips the operator authorised" and "the box is
  misconfigured" stop looking identical while the run is still going. See
  `docs/contracts/event_manifest.md` for the payload and the class set.
- **A terminal round that yields no promotion decision is rc 48** (WP12-R Phase O / R152,
  discharging R133's "rc 0 does not certify eval health"). LAW-15's "no promotion decision =
  deliverable incomplete" is enforced at the PROCESS boundary: the terminal round's typed
  `eval_broken_reason` (`mantis.eval.errors.EvalBrokenReason`, seven members) is latched
  set-once by `drain.run_terminal_eval` and resolved to
  `monitor.heartbeat.TERMINAL_EVAL_BROKEN_EXIT_CODE` through the ONE resolver,
  `mantis.config.armed_aborts.exit_code_for_abort`. A MID-RUN broken round is deliberately
  NOT covered — rounds recur, and persistent breakage stays the heartbeat watchdog's
  jurisdiction (R133's split). Which of the seven broke is read in the ONE channel, on the
  `eval_broken` event's `reason`; the rc says only that the deliverable is incomplete.
  Pinned by `tests/train/test_terminal_eval_rc.py`.

## Pinning tests

Each row names the test that would go RED if the claim above stopped being true. A claim with
no runnable producer is not listed as covered here; where the producer cannot run in CI, the
row says so and names what does run.

| claim | pinning test | runs in CI? |
|---|---|---|
| deploy-matched gate, both sides at the same constructor and count | `tests/eval/test_gate_parity.py` | yes |
| the ladder entry plays at its per-kind simulation count, not the gate's | `tests/eval/test_rung_seat_off_window.py` | yes |
| the head answers outside the encoding window at both seats | `tests/eval/test_eval_selfplay_child_parity.py`, `tests/eval/test_rung_seat_off_window.py` | yes |
| an unimplemented declared pooling is REFUSED, never a fallthrough | `tests/eval/test_value_pool_guard.py`, `tests/eval/test_eval_decode_guard_ordering.py`, `tests/eval/test_graph_round_encoding.py` | yes |
| eff_n is trajectory-hash-distinct; the low-power guard is per pair; an empty sample degenerates rather than raising | `tests/eval/test_aggregate_regime.py` | yes |
| a pin is a commit sha, and the declared patch is tracked | `tests/tools/test_vendor_pins_sealbot.py` | yes |
| the depth adapter drives the ceiling, neutralises the time cut, and CHECKS the receipt | `tests/bots/test_sealbot_adapter.py` | yes (against a recording double) |
| the refusal reasons name exactly their own missing step, and no environment key | `tests/bots/test_sealbot_resolve.py`, `tests/bots/test_protocol.py` | yes |
| each skip-reason class counts itself in-run, on a closed set | `tests/eval/test_rung_skip_class_counter.py` | yes |
| the external win-rate field populates with no producer change | `tests/eval/test_wr_sealbot_handshake.py`, `tests/eval/test_wr_sealbot_config_only.py` | yes |
| the decision document agrees with the minted config | `tests/eval/test_eval_decision_run5_doc.py` | yes |
| the REAL vendored engine agrees on the rules, holds its depth receipt, and is deterministic | `tests/bots/test_sealbot_vendored.py` | **no** — Tier 2, `@pytest.mark.integration`; skips with a named reason and a named box counterpart, and a skip is reported as `not_run`, never as coverage |
