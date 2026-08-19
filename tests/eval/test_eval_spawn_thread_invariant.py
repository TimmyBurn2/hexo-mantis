"""F-816-20 item 1 — the eval spawn site refuses an off-main-thread kick.

`_worker_entry` arms `PR_SET_PDEATHSIG` (F-816-14) and the kernel signals on the death of
the THREAD that created the child. So a round kicked from a worker thread would be SIGKILLed
the moment that thread returned — a premature kill of a LIVE eval round, strictly worse than
the orphan the arming prevents. `compose_run` already declares the main-thread precondition
in prose (it calls `signal.signal`, which raises off the main thread) and nothing asserted
it; a prose precondition no test can cross is the class R71 names.

`RuntimeError` and not `assert`: `python -O` strips asserts, and this is a production safety
invariant rather than a test aid. Its sibling refusal — the `mp_ctx_name` whitelist, which
raises `ValueError` because it answers a different question (a bad argument VALUE, not a
violated invariant) — lives in `test_eval_mp_context_whitelist.py`.
"""
from __future__ import annotations

import ast
import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.pipeline import EvalPipeline

_PIPELINE_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "mantis" / "eval" / "pipeline.py"
)
_DEADLINE_SEC = 20.0


# ── fixtures, self-contained (R5 bars importing another test module) ─────────────────────
def _bare_pipeline(tmp_path: Path) -> Any:
    """A pipeline with ONLY the two attributes `_spawn_worker` reads.

    `__new__` without `__init__` is the established house shape for driving one method of this
    class (its written justification lives at `tests/eval/test_promotion_integrity.py`, cited
    for the technique — nothing is imported from it). Booting a full pipeline would start a
    poller thread and a ladder file that the refusal under test never touches.
    """
    pipe = EvalPipeline.__new__(EvalPipeline)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    pipe._work_dir = work
    pipe._mp_ctx_name = "spawn"
    return pipe


def _spec(round_id: str = "r000001_1000") -> Any:
    return SimpleNamespace(
        round_id=round_id,
        result_path=str(round_id) + "_result.json",
        to_dict=lambda: {"round_id": round_id},
    )


# ── item 1: the main-thread refusal ──────────────────────────────────────────────────────
def test_spawning_an_eval_worker_off_the_main_thread_is_REFUSED(tmp_path) -> None:
    """The refusal, driven from a real thread rather than by faking `main_thread()`.

    The message must NAME `PR_SET_PDEATHSIG` (R73 — the row's claim is the message, and the
    message is the only route an operator has to the reason). `RuntimeError` and not
    `AssertionError`: a bare `assert` vanishes under `python -O`, which would leave the
    invariant unenforced in exactly the long-running production process it protects."""
    pipe = _bare_pipeline(tmp_path)
    box: dict[str, Any] = {}

    def _attempt() -> None:
        try:
            box["proc"] = pipe._spawn_worker(_spec())
        except BaseException as exc:  # noqa: BLE001 — the exception IS the observation
            box["exc"] = exc

    worker = threading.Thread(target=_attempt, name="q3-eval-kick")
    worker.start()
    worker.join(timeout=_DEADLINE_SEC)
    assert not worker.is_alive(), "the kicking thread never returned"

    assert isinstance(box.get("exc"), RuntimeError), (
        f"an off-main-thread eval kick must raise RuntimeError; got {box.get('exc')!r}"
    )
    assert "PR_SET_PDEATHSIG" in str(box["exc"]), (
        f"the refusal must name the mechanism it protects; got {str(box['exc'])!r}"
    )
    assert list(pipe._work_dir.iterdir()) == [], (
        "the refusal must fire BEFORE the spec file is written — a refused kick that has "
        "already littered the work dir has done half a round"
    )


def test_spawning_from_the_main_thread_is_allowed(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE POSITIVE CONTROL. Without it the refusal could be "always raise" and this file
    would stay green while the eval pipeline could no longer kick a round at all.

    `multiprocessing.get_context` is patched to a fake whose `Process` is a stub, so no real
    process starts: the claim under test is that the guard LETS THE PATH THROUGH, not anything
    about a real worker."""
    pipe = _bare_pipeline(tmp_path)
    started: list[str] = []

    class _StubProcess:
        def __init__(self, **kw: Any) -> None:
            self.daemon = kw.get("daemon")
            self.pid = 4242
            self.exitcode = None

        def start(self) -> None:
            started.append("start")

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(
        multiprocessing, "get_context",
        lambda name=None: SimpleNamespace(Process=lambda **kw: _StubProcess(**kw)),
    )
    monkeypatch.setattr(
        "mantis.train.lifecycle.signals.register_child", lambda _p: None,
    )

    proc = pipe._spawn_worker(_spec())
    assert isinstance(proc, _StubProcess) and started == ["start"], (
        "a main-thread kick must reach the spawn and start the child"
    )


def test_both_spawn_call_sites_run_under_the_guarded_method(tmp_path) -> None:
    """A STATIC row: `ctx.Process(` appears in `pipeline.py` ONLY inside `_spawn_worker`.

    The refusal is a choke point, and a choke point is only worth what its exclusivity is
    worth. A future second spawn path added elsewhere in the file would bypass the guard
    silently — no test would red, and the first symptom would be a round SIGKILLed mid-flight
    on a production box."""
    tree = ast.parse(_PIPELINE_SRC.read_text(encoding="utf-8"), filename=str(_PIPELINE_SRC))
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_spawn_worker":
            guarded.update(
                inner.lineno for inner in ast.walk(node) if isinstance(inner, ast.Call)
            )
    sites: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "Process":
            sites.append(node.lineno)
    assert sites, "no `.Process(` call found at all — this census has stopped measuring"
    outside = [line for line in sites if line not in guarded]
    assert outside == [], (
        f"every process spawn in pipeline.py must go through `_spawn_worker`; unguarded "
        f"`.Process(` call(s) at line(s) {outside}"
    )
