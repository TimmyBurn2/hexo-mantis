"""The read-only server: one poll thread swaps in immutable snapshots, a threaded HTTP server renders from them; no route writes."""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from .readers.record import RunRecord, RunSnapshot
from .views import run as run_view
from .views.page import WEB

_LOG = logging.getLogger(__name__)
_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
#: A rendered Run page is reused for this long before the status line is re-read against the clock.
PAGE_TTL_SEC = 20.0


@dataclass(frozen=True)
class Reply:
    """One response: status, content type, body, and a Location for a redirect."""

    status: int
    ctype: str
    body: bytes
    location: str | None = None


def text(status: int, message: str) -> Reply:
    return Reply(status, "text/plain; charset=utf-8", message.encode("utf-8"))


def as_json(status: int, body: Any) -> Reply:
    return Reply(status, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8"))


class Hub:
    """The served runs and their latest snapshots; `poll_once` re-reads every run and swaps the mapping in one assignment."""

    def __init__(self, records: list[RunRecord]) -> None:
        self.records = {r.label: r for r in records}
        self.labels = tuple(self.records)
        self._lock = threading.Lock()
        self._snaps: MappingProxyType[str, RunSnapshot] = MappingProxyType({})
        self.generation = 0
        self.failures: dict[str, str] = {}
        self._pages: dict[tuple[str, str | None], tuple[int, float, bytes]] = {}

    def poll_once(self) -> None:
        """Read what is new in every run; a run whose read fails keeps its previous snapshot and the failure is logged."""
        fresh = dict(self.snapshots())
        for label, rec in self.records.items():
            try:
                fresh[label] = rec.poll()
                self.failures.pop(label, None)
            except (OSError, RuntimeError, ValueError) as exc:
                _LOG.exception("poll of %s failed; its previous snapshot stays", label)
                self.failures[label] = f"{type(exc).__name__} at {time.strftime('%H:%M:%S')}"
        with self._lock:
            self._snaps = MappingProxyType(fresh)
            self.generation += 1

    def snapshots(self) -> MappingProxyType[str, RunSnapshot]:
        with self._lock:
            return self._snaps

    def run_page(self, label: str, compare: str | None, now: float) -> bytes:
        """The Run view of `label` (`compare` overlaid), rendered once per snapshot and re-rendered past the TTL."""
        snaps = self.snapshots()
        generation = self.generation
        key = (label, compare)
        cached = self._pages.get(key)
        if cached is not None and cached[0] == generation and now - cached[1] < PAGE_TTL_SEC:
            return cached[2]
        chosen = [snaps[label]] + ([snaps[compare]] if compare is not None and compare in snaps and compare != label else [])
        body = run_view.page(chosen, runs=self.labels, now=now).encode("utf-8")
        self._pages[key] = (generation, now, body)
        return body

    def follow(self, every_sec: float, stop: threading.Event) -> None:
        """The poll loop: until `stop` is set, one `poll_once` every `every_sec`."""
        while not stop.wait(every_sec):
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 — the poll thread must outlive one bad read; the snapshot stays
                _LOG.exception("the poll loop's read failed")


Route = Callable[["Hub", list[str], dict[str, list[str]]], Reply]


def static(name: str) -> Reply:
    """A file of the package's `web/` by its bare name only: a path, `..` or an unlisted name is a 404, never a file."""
    allowed = {p.name: p for p in WEB.iterdir() if p.is_file() and p.suffix in _TYPES}
    path = allowed.get(name)
    if path is None:
        return text(404, f"no static file {name!r}")
    return Reply(200, _TYPES[path.suffix], path.read_bytes())


def route_get(hub: Hub, raw_path: str, extra: dict[str, Callable[..., Reply]] | None = None) -> Reply:
    """GET dispatch over the path's segments; every unknown name is a 404."""
    split = urlsplit(raw_path)
    parts = [p for p in split.path.split("/") if p]
    query = parse_qs(split.query)
    if not parts:
        return Reply(302, "text/plain", b"", f"/run/{quote(hub.labels[0])}")
    if parts == ["run"]:
        run = (query.get("run") or [hub.labels[0]])[0]
        compare = (query.get("compare") or [""])[0]
        return Reply(302, "text/plain", b"", f"/run/{quote(run)}" + (f"?compare={quote(compare)}" if compare else ""))
    if parts[0] == "static" and len(parts) == 2:
        return static(parts[1])
    if parts[0] == "run" and len(parts) == 2:
        if parts[1] not in hub.snapshots():
            return text(404, f"no run {parts[1]!r}; served: {', '.join(hub.labels)}")
        compare = (query.get("compare") or [None])[0]
        return Reply(200, "text/html; charset=utf-8", hub.run_page(parts[1], compare, time.time()))
    handler = (extra or {}).get(parts[0] if parts[0] != "api" or len(parts) < 2 else f"api/{parts[1]}")
    if handler is not None:
        return handler(hub, parts, query)
    return text(404, f"no route {split.path}")


def handler_class(hub: Hub, *, get_extra: dict[str, Callable[..., Reply]] | None = None,
                  post: Callable[[str, bytes], Reply] | None = None) -> type[BaseHTTPRequestHandler]:
    """The request handler bound to one hub; POST exists only for the routes `post` answers."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, reply: Reply) -> None:
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.ctype)
            self.send_header("Content-Length", str(len(reply.body)))
            self.send_header("Cache-Control", "no-store")
            if reply.location:
                self.send_header("Location", reply.location)
            self.end_headers()
            self.wfile.write(reply.body)

        def do_GET(self) -> None:  # noqa: N802 — http.server's own name
            self._send(route_get(hub, self.path, get_extra))

        def do_POST(self) -> None:  # noqa: N802 — http.server's own name
            if post is None:
                self._send(text(404, "no POST route"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > 1_000_000:
                self._send(as_json(400, {"ok": False, "refused": "the body needs a Content-Length of 1 byte to 1 MB"}))
                return
            self._send(post(urlsplit(self.path).path, self.rfile.read(length)))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server's own signature
            return

    return Handler


def make_server(bind: str, port: int, hub: Hub, **routes: Any) -> ThreadingHTTPServer:
    """A bound, not yet serving, server; port 0 asks the OS for one (tests)."""
    return ThreadingHTTPServer((bind, port), handler_class(hub, **routes))


def run_server(args: argparse.Namespace, records: list[RunRecord], **routes: Any) -> int:
    """`serve`: the first read before the socket opens, then the poll thread and the server until interrupted."""
    hub = Hub(records)
    t0 = time.time()
    hub.poll_once()
    if hub.failures:
        print(f"dash: refused: the first read failed for {', '.join(hub.failures)}", flush=True)
        return 2
    stop = threading.Event()
    poller = threading.Thread(target=hub.follow, args=(args.poll_sec, stop), name="poll", daemon=True)
    poller.start()
    httpd = make_server(args.bind, args.port, hub, **routes)
    unsafe = "" if args.bind == "127.0.0.1" else "  (UNSAFE: not loopback — no auth, no TLS)"
    print(f"mantis dash on http://{args.bind}:{httpd.server_address[1]}/ — runs {', '.join(hub.labels)}; first read "
          f"{time.time() - t0:.1f} s; re-read every {args.poll_sec:g} s; Ctrl-C stops{unsafe}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
    return 0
