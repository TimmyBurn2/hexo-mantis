"""The dispatcher: one request in, one status + body out; engines load once; every refusal is named."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def dispatch(analyzer):
    return importlib.import_module("analyzer.dispatch")


@pytest.fixture(scope="module")
def dispatcher(analyzer, dispatch, tmp_path_factory, mint_stamp):
    engines = importlib.import_module("analyzer.engines")
    d = tmp_path_factory.mktemp("run9") / "checkpoints"
    d.mkdir()
    mint_stamp(d)
    (d / "best_model.pt").write_bytes(b"")
    (d / "run9_00000100_deadbeef.ckpt").write_bytes(b"")
    disp = dispatch.Dispatcher(engines.discover([d]), device="cpu", threads=2)
    yield disp
    disp.close()


def _id(dispatcher, kind: str, run_id: str | None = None) -> str:
    return next(r.id for r in dispatcher.infos if r.kind == kind and (run_id is None or r.run_id == run_id))


def test_engines_lists_every_row_with_its_kind_and_no_host_path(dispatcher):
    out = dispatcher.handle({"op": "engines"})
    assert out["status"] == 200
    rows = out["body"]["engines"]
    assert {row["kind"] for row in rows} == {"mantis", "snapshot_gap"} and all("path" not in row for row in rows)


def test_analyze_returns_the_record_and_echoes_seq(dispatcher):
    out = dispatcher.handle({"op": "analyze", "engine": _id(dispatcher, "mantis", "an1"), "moves": "0,0;1,0", "sims": 0, "seq": 5})
    assert out["status"] == 200 and out["body"]["seq"] == 5 and out["body"]["ok"] is True
    assert out["body"]["record"]["position"]["ply"] == 2 and "value" in out["body"]["record"]["raw"]
    assert json.dumps(out["body"])


def test_trace_returns_one_row_per_ply(dispatcher):
    out = dispatcher.handle({"op": "trace", "engine": _id(dispatcher, "mantis", "an1"), "moves": [[0, 0], [1, 0]], "sims": 0, "seq": 1})
    assert out["status"] == 200 and [r["ply"] for r in out["body"]["trace"]] == [0, 1, 2]


@pytest.mark.parametrize("request_,status,needle", [
    ({"op": "analyze", "engine": "nope", "moves": "", "sims": 0, "seq": 1}, 404, "unknown engine 'nope'"),
    ({"op": "analyze", "engine": "SNAP", "moves": "", "sims": 0, "seq": 1}, 503, "snapshot format"),
    ({"op": "analyze", "engine": "EMPTY", "moves": "", "sims": 0, "seq": 1}, 503, "run9_00000100_deadbeef: "),
    ({"op": "analyze", "engine": "MANTIS", "moves": "0,0;0,0", "sims": 0, "seq": 1}, 400, "illegal move (0, 0) at ply 1"),
    ({"op": "analyze", "engine": "MANTIS", "moves": "x", "sims": 0, "seq": 1}, 400, "move 0 is not"),
    ({"op": "analyze", "engine": "MANTIS", "moves": None, "sims": 0, "seq": 1}, 400, "moves must be text"),
    ({"op": "analyze", "engine": "MANTIS", "moves": "", "sims": "many", "seq": 1}, 400, "sims must be"),
    ({"op": "bogus", "seq": 1}, 404, "unknown op"),
])
def test_refusals_are_named_with_their_status(dispatcher, request_, status, needle):
    ids = {"SNAP": _id(dispatcher, "snapshot_gap"), "MANTIS": _id(dispatcher, "mantis", "an1"),
           "EMPTY": _id(dispatcher, "mantis", "run9")}
    req = dict(request_)
    req["engine"] = ids.get(req.get("engine"), req.get("engine"))
    out = dispatcher.handle(req)
    assert out["status"] == status and out["body"]["ok"] is False and needle in out["body"]["refused"]


def test_a_failed_load_is_a_503_naming_the_exception_and_is_retried_next_time(dispatcher):
    empty = _id(dispatcher, "mantis", "run9")
    first = dispatcher.handle({"op": "analyze", "engine": empty, "moves": "", "sims": 0, "seq": 1})
    assert first["status"] == 503 and "Error" in first["body"]["refused"]
    assert empty not in dispatcher._loaded, "a failed load is not cached"  # noqa: SLF001


def test_once_prints_the_record_as_json(analyzer, dispatcher, capsys):
    cli = importlib.import_module("analyzer.cli")
    eid = _id(dispatcher, "mantis", "an1")
    d = str(Path(next(r.path for r in dispatcher.infos if r.id == eid)).parent)
    rc = cli.main(["once", "--checkpoints", d, "--engine", eid, "--moves", "0,0;1,0", "--sims", "0", "--threads", "2"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["position"]["ply"] == 2 and rec["search"] == {"absent": "sims=0 (raw only)"}


def test_help_shows_both_subcommands(analyzer, capsys):
    cli = importlib.import_module("analyzer.cli")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0 and "once" in capsys.readouterr().out
