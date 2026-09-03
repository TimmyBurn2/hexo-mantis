"""O5 — bootstrap path resolver (resolve/bootstrap.resolve_bootstrap).

CLI arg, not a config-file key -> no schema field. Validates at launch (exists-or-raise);
existence probe is injectable and called exactly once.
"""
import pytest

from mantis.config.resolve.bootstrap import (
    BootstrapNotFoundError,
    ResolvedBootstrap,
    resolve_bootstrap,
)


def test_none_is_fresh_run_no_raise():
    res = resolve_bootstrap(None)
    assert res == ResolvedBootstrap(path=None, source="none")


def test_existing_path_is_cli_source():
    res = resolve_bootstrap("exists.pt", exists=lambda _: True)
    assert res == ResolvedBootstrap("exists.pt", "cli")


def test_missing_path_raises_naming_path_and_knob():
    """The knob is `--resume-from` (AUDIT-1 F-47). This asserted `BOOTSTRAP`, which named a make
    target that does not exist — and the assertion passed for as long as the message was wrong,
    because both sides were the same fiction."""
    with pytest.raises(BootstrapNotFoundError) as exc:
        resolve_bootstrap("missing.pt", exists=lambda _: False)
    msg = str(exc.value)
    assert "missing.pt" in msg and "--resume-from" in msg


def test_error_is_file_not_found_error():
    assert issubclass(BootstrapNotFoundError, FileNotFoundError)


def test_existence_checked_exactly_once():
    calls = []

    def _exists(p):
        calls.append(p)
        return True

    resolve_bootstrap("exists.pt", exists=_exists)
    assert calls == ["exists.pt"]


# ── the guard is WIRED, and that is the half that was missing ──────────────────────────────
def test_run_main_calls_the_resolver_before_it_launches() -> None:
    """AUDIT-1 F-47's repair, pinned STRUCTURALLY rather than by running a launch.

    The resolver had ZERO callers. `mantis.run.main` parsed `--resume-from` and handed it
    straight to `launch_run`, so a mistyped path surfaced as whatever `torch.load` says about a
    missing file, deep inside `init_trainer`'s resume branch, after the composition root had
    already built a run. A guard and the flag it guards, in the same process, joined by nothing.

    Asserted by AST over `main`'s own body: the call must exist AND it must come BEFORE the
    `launch_run` call, because a validation that runs after the launch is not a launch guard.
    """
    import ast
    import inspect
    import textwrap

    import mantis.run as run_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_mod.main)))
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in ("resolve_bootstrap", "launch_run"):
                lines.setdefault(name, node.lineno)
    assert "resolve_bootstrap" in lines, (
        "`mantis.run.main` does not call `resolve_bootstrap`. The launch-time existence check "
        "for `--resume-from` is unreachable again, which is the state AUDIT-1 F-47 found"
    )
    assert "launch_run" in lines, "the pin's own premise is gone — `main` no longer launches"
    assert lines["resolve_bootstrap"] < lines["launch_run"], (
        "`resolve_bootstrap` is called AFTER `launch_run`. A stale path must fail before the "
        "composition root builds a run, not after"
    )


def test_the_error_names_a_knob_the_operator_can_actually_use() -> None:
    """The message pointed at `BOOTSTRAP=<path>` (a make target that does not exist) and
    `--checkpoint` (a flag that does not exist). An error naming a knob nobody has is worse
    than no error: it sends the reader looking for something that was never there."""
    import inspect

    with pytest.raises(BootstrapNotFoundError) as excinfo:
        resolve_bootstrap("/nonexistent/checkpoint.pt")
    message = str(excinfo.value)
    assert "--resume-from" in message, f"the error names no usable knob: {message}"
    assert "BOOTSTRAP=" not in message and "--checkpoint" not in message, (
        f"the error still names a knob that does not exist: {message}"
    )

    # …and `--resume-from` is a real flag on the real parser, not just a nicer string.
    import mantis.run as run_mod

    src = inspect.getsource(run_mod.main)
    assert '"--resume-from"' in src, (
        "the error names `--resume-from` but `main` no longer declares that flag — the message "
        "would be pointing at a knob that does not exist, which is the defect it just fixed"
    )
