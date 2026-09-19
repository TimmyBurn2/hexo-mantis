"""The server with a STUB handler (no torch): routes, status codes, seq, supersession, survival of a panic-class exception."""
from __future__ import annotations

import importlib
import json
import threading
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture(scope="module")
def serve(analyzer):
    return importlib.import_module("analyzer.serve")


class _Stub:
    """Answers by op; `slow` requests sleep so a later one can supersede; `boom` raises a BaseException subclass."""

    def __init__(self):
        self.seen = []

    def __call__(self, req):
        self.seen.append(req)
        op = req.get("op")
        if op == "engines":
            return {"status": 200, "body": {"engines": [{"id": "e1", "kind": "mantis"}]}}
        if req.get("moves") == "slow":
            time.sleep(0.6)
        if req.get("moves") == "boom":
            raise KeyboardInterrupt("simulated PanicException-class failure")
        if req.get("moves") == "bad":
            return {"status": 400, "body": {"seq": req.get("seq"), "ok": False, "refused": "illegal move (0, 0) at ply 1"}}
        return {"status": 200, "body": {"seq": req.get("seq"), "ok": True, "record": {"op": op, "sims": req.get("sims")}}}


@pytest.fixture()
def server(serve):
    stub = _Stub()
    analyst = serve.Analyst(stub, timeout_sec=2.0)
    analyst.start()
    httpd = serve.make_server("127.0.0.1", 0, analyst, b"<!doctype html><title>t</title>")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd, stub
    httpd.shutdown()
    analyst.stop()


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_the_page_and_the_engines_route(server):
    httpd, _stub = server
    port = httpd.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        assert resp.status == 200 and b"<title>t</title>" in resp.read()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/engines", timeout=5) as resp:
        assert json.loads(resp.read()) == {"engines": [{"id": "e1", "kind": "mantis"}]}


def test_analyze_echoes_seq_and_a_refusal_carries_its_status(server):
    httpd, _stub = server
    port = httpd.server_address[1]
    status, body = _post(port, "/analyze", {"engine": "e1", "moves": "0,0", "sims": 3, "client": "c", "seq": 9})
    assert status == 200 and body["seq"] == 9 and body["record"] == {"op": "analyze", "sims": 3}
    status, body = _post(port, "/analyze", {"engine": "e1", "moves": "bad", "sims": 0, "client": "c", "seq": 10})
    assert status == 400 and body["refused"].startswith("illegal move")
    status, body = _post(port, "/nowhere", {})
    assert status == 404
    req = urllib.request.Request(f"http://127.0.0.1:{port}/analyze", data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 400


def test_a_newer_request_from_the_same_client_supersedes_the_queued_one(server):
    httpd, stub = server
    port = httpd.server_address[1]
    results = {}

    def go(seq, moves):
        results[seq] = _post(port, "/analyze", {"engine": "e1", "moves": moves, "sims": 8, "client": "c", "seq": seq})

    t1 = threading.Thread(target=go, args=(1, "slow"))
    t1.start()
    time.sleep(0.1)
    t2 = threading.Thread(target=go, args=(2, "0,0"))
    t2.start()
    time.sleep(0.1)
    t3 = threading.Thread(target=go, args=(3, "0,0;1,0"))
    t3.start()
    for t in (t1, t2, t3):
        t.join(timeout=5)
    assert results[1][1]["ok"] is True, "the running request completes"
    assert results[2] == (200, {"seq": 2, "superseded": True})
    assert results[3][1]["record"] == {"op": "analyze", "sims": 8}
    assert [r.get("seq") for r in stub.seen if r.get("op") == "analyze"] == [1, 3]


def test_a_raw_request_does_not_supersede_a_search_request(serve):
    a = {"client": "c", "engine": "e1", "op": "analyze", "sims": 0}
    b = {"client": "c", "engine": "e1", "op": "analyze", "sims": 256}
    assert serve.supersession_key(a) != serve.supersession_key(b)
    assert serve.supersession_key(a) == serve.supersession_key({**a, "moves": "x"})


def test_a_base_exception_in_the_handler_is_a_500_and_the_analyst_survives(server):
    httpd, _stub = server
    port = httpd.server_address[1]
    status, body = _post(port, "/analyze", {"engine": "e1", "moves": "boom", "sims": 0, "client": "c", "seq": 1})
    assert status == 500 and "KeyboardInterrupt" in body["refused"]
    status, body = _post(port, "/analyze", {"engine": "e1", "moves": "0,0", "sims": 0, "client": "c", "seq": 2})
    assert status == 200 and body["ok"] is True


def test_a_timeout_is_a_504(serve):
    analyst = serve.Analyst(lambda req: time.sleep(1.0) or {"status": 200, "body": {}}, timeout_sec=0.2)
    analyst.start()
    try:
        out = analyst.submit({"op": "analyze", "seq": 4, "client": "c"})
        assert out["status"] == 504 and out["body"]["seq"] == 4 and "0.2" in out["body"]["refused"]
    finally:
        analyst.stop()


def test_the_default_bind_is_loopback(analyzer):
    cli = importlib.import_module("analyzer.cli")
    assert cli.DEFAULT_BIND == "127.0.0.1"
    args = cli.build_parser().parse_args(["serve", "--checkpoints", "x"])
    assert args.bind == "127.0.0.1" and args.port == 8766
