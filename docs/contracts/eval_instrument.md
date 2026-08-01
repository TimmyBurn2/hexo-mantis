# Contract: eval instrument

- version: v1
- owner: mantis.arena
- status: SKELETON — contract text lands with the subsystem port

## Summary
deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games

## Who asserts what where
TODO (filled by the porting work package)

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
TODO (filled by the porting work package)
