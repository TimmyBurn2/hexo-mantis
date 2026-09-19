"""The dispatcher: one request in, one status + body out; engines load once; every refusal is named."""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture(scope="module")
def dispatch(analyzer):
    return importlib.import_module("analyzer.dispatch")


@pytest.fixture(scope="module")
def dispatcher(analyzer, dispatch, tmp_path_factory, mint_stamp):
    engines = importlib.import_module("analyzer.engines")
    d = tmp_path_factory.mktemp("ckpt")
    mint_stamp(d)
    (d / "best_model.pt").write_bytes(b"")
    disp = dispatch.Dispatcher(engines.discover([d]), device="cpu", threads=2)
    yield disp
    disp.close()


def test_engines_lists_every_row_with_its_kind(dispatcher):
    out = dispatcher.handle({"op": "engines"})
    assert out["status"] == 200
    kinds = {row["kind"] for row in out["body"]["engines"]}
    assert kinds == {"mantis", "snapshot_gap"}


def test_analyze_returns_the_record_and_echoes_seq(dispatcher):
    eid = next(r.id for r in dispatcher.infos if r.kind == "mantis")
    out = dispatcher.handle({"op": "analyze", "engine": eid, "moves": "0,0;1,0", "sims": 0, "seq": 5})
    assert out["status"] == 200 and out["body"]["seq"] == 5 and out["body"]["ok"] is True
    assert out["body"]["record"]["position"]["ply"] == 2 and "value" in out["body"]["record"]["raw"]
    assert json.dumps(out["body"])


def test_trace_returns_one_row_per_ply(dispatcher):
    eid = next(r.id for r in dispatcher.infos if r.kind == "mantis")
    out = dispatcher.handle({"op": "trace", "engine": eid, "moves": [[0, 0], [1, 0]], "sims": 0, "seq": 1})
    assert out["status"] == 200 and [r["ply"] for r in out["body"]["trace"]] == [0, 1, 2]


@pytest.mark.parametrize("request_,status,needle", [
    ({"op": "analyze", "engine": "nope", "moves": "", "sims": 0, "seq": 1}, 404, "unknown engine 'nope'"),
    ({"op": "analyze", "engine": "SNAP", "moves": "", "sims": 0, "seq": 1}, 503, "snapshot format"),
    ({"op": "analyze", "engine": "MANTIS", "moves": "0,0;0,0", "sims": 0, "seq": 1}, 400, "illegal move (0, 0) at ply 1"),
    ({"op": "analyze", "engine": "MANTIS", "moves": "x", "sims": 0, "seq": 1}, 400, "move 0 is not"),
    ({"op": "analyze", "engine": "MANTIS", "moves": "", "sims": "many", "seq": 1}, 400, "sims must be"),
    ({"op": "bogus", "seq": 1}, 404, "unknown op"),
])
def test_refusals_are_named_with_their_status(dispatcher, request_, status, needle):
    ids = {r.kind: r.id for r in dispatcher.infos}
    req = dict(request_)
    if req.get("engine") in ("SNAP", "MANTIS"):
        req["engine"] = ids["snapshot_gap" if req["engine"] == "SNAP" else "mantis"]
    out = dispatcher.handle(req)
    assert out["status"] == status and out["body"]["ok"] is False and needle in out["body"]["refused"]


def test_once_prints_the_record_as_json(analyzer, dispatcher, capsys):
    cli = importlib.import_module("analyzer.cli")
    eid = next(r.id for r in dispatcher.infos if r.kind == "mantis")
    d = str(next(r.path for r in dispatcher.infos if r.kind == "mantis")).rsplit("/", 1)[0]
    rc = cli.main(["once", "--checkpoints", d, "--engine", eid, "--moves", "0,0;1,0", "--sims", "0", "--threads", "2"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["position"]["ply"] == 2 and rec["search"] == {"absent": "sims=0 (raw only)"}
