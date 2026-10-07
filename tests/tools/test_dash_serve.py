"""The dash server: the served Run page equals the freeze's, loopback by default, no path escapes, one snapshot per request."""
from __future__ import annotations

import importlib
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _dash_record import (
    game,
    game_rows,
    iteration_rows,
    segment_start,
    six_in_a_row_for_p1,
    trainer_rows,
    write_config,
    write_heartbeat,
    write_segment,
    write_shard,
)


@pytest.fixture(scope="module")
def serve(dash):
    return importlib.import_module("dash.serve")


@pytest.fixture(scope="module")
def record(dash):
    return importlib.import_module("dash.readers.record")


def _run(root: Path, run_id: str = "r1", beat_age: float = 0.0) -> Path:
    run = root / run_id
    write_segment(run / "logs", run_id, 1, [segment_start(run_id, 1), *trainer_rows(range(1, 80)),
                                            *iteration_rows(range(1, 80, 4)), *game_rows(60)])
    write_config(run, run_id, None)
    beat = write_heartbeat(run / "logs", run_id, time.time() - beat_age)
    os.utime(beat, (time.time() - beat_age, time.time() - beat_age))
    write_shard(run / "logs" / "games", run_id, 1, "2026100510", [game("a", six_in_a_row_for_p1())])
    return run


@pytest.fixture()
def server(serve, record, tmp_path):
    hub = serve.Hub([record.RunRecord("r1", _run(tmp_path)), record.RunRecord("r2", _run(tmp_path, "r2"))])
    hub.poll_once()
    httpd = serve.make_server("127.0.0.1", 0, hub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield hub, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, str, dict[str, str]]:
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(url, timeout=10) as resp:  # encoding-gate: ok -- an HTTP response, decoded below
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8"), dict(err.headers)


def _sections(html: str) -> list[str]:
    return re.findall(r'<section class="q".*?</section>', html, re.S)


def test_the_served_run_page_carries_the_same_sections_as_the_freeze(dash, server):
    hub, base = server
    status, html, _ = _get(f"{base}/run/r1")
    frozen = importlib.import_module("dash.views.run").page([hub.snapshots()["r1"]], runs=hub.labels, now=None)
    assert status == 200 and _sections(html) == _sections(frozen) and len(_sections(html)) == 4
    assert 'href="/static/dash.css"' in html and 'class="state live"' in html


def test_the_root_and_the_picker_redirect_to_a_run(server):
    _, base = server
    assert _get(f"{base}/")[2]["Location"] == "/run/r1"
    assert _get(f"{base}/run?run=r2&compare=r1")[2]["Location"] == "/run/r2?compare=r1"


@pytest.mark.parametrize("path", ["/static/..%2Fserve.py", "/static/%2e%2e", "/static/serve.py", "/static/dash.css/x",
                                  "/run/nope", "/nothing", "/run/r1/../../etc"])
def test_a_path_outside_the_served_names_is_a_404_never_a_file(server, path):
    _, base = server
    status, body, _ = _get(base + path)
    assert status == 404 and "import" not in body


def test_static_files_are_served_by_name_with_their_type(server):
    _, base = server
    status, body, headers = _get(f"{base}/static/charts.js")
    assert status == 200 and headers["Content-Type"].startswith("text/javascript") and "crosshair" in body


def test_a_compare_page_names_both_runs(server):
    _, base = server
    status, html, _ = _get(f"{base}/run/r1?compare=r2")
    assert status == 200 and 'class="c2"' in html and '<option value="r2" selected>' in html


def test_a_request_during_a_poll_sees_the_old_snapshot_whole(serve, record, tmp_path, monkeypatch):
    rec = record.RunRecord("r1", _run(tmp_path))
    hub = serve.Hub([rec])
    hub.poll_once()
    before = hub.snapshots()["r1"]
    entered, release = threading.Event(), threading.Event()
    real = rec.poll

    def slow_poll():
        snap = real()
        entered.set()
        release.wait(10)
        return snap

    monkeypatch.setattr(rec, "poll", slow_poll)
    thread = threading.Thread(target=hub.poll_once)
    thread.start()
    entered.wait(10)
    assert hub.snapshots()["r1"] is before, "until the swap, every reader holds the old snapshot"
    release.set()
    thread.join(10)
    assert hub.snapshots()["r1"] is not before


def test_a_failed_poll_keeps_the_previous_snapshot_and_names_the_failure(serve, record, tmp_path, monkeypatch):
    rec = record.RunRecord("r1", _run(tmp_path))
    hub = serve.Hub([rec])
    hub.poll_once()
    before = hub.snapshots()["r1"]
    monkeypatch.setattr(rec, "poll", lambda: (_ for _ in ()).throw(OSError("vanished mid-copy")))
    hub.poll_once()
    assert hub.snapshots()["r1"] is before and hub.failures["r1"].startswith("OSError")


def test_a_stale_heartbeat_is_drawn_stale_not_live(serve, record, tmp_path):
    hub = serve.Hub([record.RunRecord("r1", _run(tmp_path, beat_age=1800.0))])
    hub.poll_once()
    assert 'class="state stale"' in hub.run_page("r1", None, time.time()).decode("utf-8")


def test_the_server_binds_loopback_by_default(dash):
    cli = importlib.import_module("dash.cli")
    args = cli.build_parser().parse_args(["serve", "--run", "a=/x"])
    assert args.bind == "127.0.0.1" and cli.DEFAULT_BIND == "127.0.0.1"


@pytest.mark.parametrize("argv", [["serve", "--run", "nolabel"], ["serve", "--run", "a=/x", "--records", "b=/y"],
                                  ["freeze", "--out", "o.html"], ["serve", "--run", "a=/x", "--rule-unit", "a=u,v@0"]])
def test_a_malformed_input_is_refused_by_name(dash, argv):
    cli = importlib.import_module("dash.cli")
    with pytest.raises(SystemExit):
        cli.records_of(cli.build_parser().parse_args(argv))


def test_the_page_cache_holds_only_served_runs_whatever_the_query(serve, record, tmp_path):
    hub = serve.Hub([record.RunRecord("r1", _run(tmp_path)), record.RunRecord("r2", _run(tmp_path, "r2"))])
    hub.poll_once()
    for i in range(50):
        hub.run_page("r1", f"x{i}", time.time())
    hub.run_page("r1", "r2", time.time())
    assert set(hub._pages) == {("r1", None), ("r1", "r2")}  # noqa: SLF001


def test_a_failed_read_is_drawn_as_one_not_as_a_stopped_run(serve, record, tmp_path, monkeypatch):
    rec = record.RunRecord("r1", _run(tmp_path))
    hub = serve.Hub([rec])
    hub.poll_once()
    monkeypatch.setattr(rec, "poll", lambda: (_ for _ in ()).throw(OSError("vanished mid-copy")))
    hub.poll_once()
    html = hub.run_page("r1", None, time.time() + 7200).decode("utf-8")
    assert 'class="state unread"' in html and "last read failed" in html and 'class="state stopped"' not in html


def test_a_label_with_a_space_round_trips_through_its_own_links(serve, record, tmp_path):
    hub = serve.Hub([record.RunRecord("run eleven", _run(tmp_path))])
    hub.poll_once()
    assert serve.route_get(hub, "/").location == "/run/run%20eleven"
    assert serve.route_get(hub, "/run/run%20eleven").status == 200
