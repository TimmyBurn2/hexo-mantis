"""The loopback server: ONE analyst thread (every engine, Board and tree lives and dies on it) with supersession, four routes."""
from __future__ import annotations

import argparse
import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .dispatch import Dispatcher, refusal
from .engines import EngineInfo
from .html import render

HandleFn = Callable[[dict[str, Any]], dict[str, Any]]
Item = tuple[dict[str, Any], Future[dict[str, Any]]]
_ROUTES = {"/analyze": "analyze", "/trace": "trace"}


def supersession_key(request: dict[str, Any]) -> tuple[Any, ...]:
    """`(client, engine, op, raw|search)`: a newer request with the same key replaces a queued one; `engines` never."""
    if request.get("op") == "engines":
        return ("engines", id(request))
    try:
        tier = "raw" if int(request.get("sims", 0) or 0) == 0 else "search"
    except (TypeError, ValueError, OverflowError):
        tier = "search"
    return request.get("client"), request.get("engine"), request.get("op"), tier


class Analyst:
    """The one worker thread; `submit` waits on a Future up to `timeout_sec` and answers 504 past it."""

    def __init__(self, handler: HandleFn, *, timeout_sec: float, on_stop: Callable[[], None] | None = None) -> None:
        self._handler = handler
        self._on_stop = on_stop
        self.timeout_sec = float(timeout_sec)
        self._queue: queue.Queue[Item | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="analyst", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Queue the stop sentinel; `on_stop` runs ON the analyst thread (the engines must die where they lived)."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=self.timeout_sec)

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Enqueue and wait; the reply is the handler's `{status, body}` or a 504 with the wait named."""
        fut: Future[dict[str, Any]] = Future()
        self._queue.put((request, fut))
        try:
            return fut.result(timeout=self.timeout_sec)
        except TimeoutError:
            return refusal(request.get("seq"), 504, f"the analyst did not answer within {self.timeout_sec} s")

    def _drain(self, first: Item) -> list[Item]:
        pending = [first]
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)
                break
            pending.append(item)
        survivors: dict[tuple[Any, ...], Item] = {}
        for item in pending:
            key = supersession_key(item[0])
            if key in survivors:
                old_req, old_fut = survivors[key]
                old_fut.set_result({"status": 200, "body": {"seq": old_req.get("seq"), "superseded": True}})
            survivors[key] = item
        return [item for item in pending if survivors.get(supersession_key(item[0])) is item]

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                if self._on_stop is not None:
                    self._on_stop()
                return
            for req, fut in self._drain(item):
                try:
                    fut.set_result(self._handler(req))
                except BaseException as exc:  # noqa: BLE001 — a PanicException is a BaseException; the analyst must outlive it
                    fut.set_result(refusal(req.get("seq"), 500, f"analyst exception: {type(exc).__name__}: {exc}"))


def _handler_class(analyst: Analyst, page: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, out: dict[str, Any]) -> None:
            self._send(int(out["status"]), json.dumps(out["body"]).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 — http.server's own name
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/engines":
                self._json(analyst.submit({"op": "engines"}))
            else:
                self._json(refusal(None, 404, f"no route {path}"))

        def do_POST(self) -> None:  # noqa: N802 — http.server's own name
            path = urlsplit(self.path).path
            op = _ROUTES.get(path)
            if op is None:
                self._json(refusal(None, 404, f"no route {path}"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                self._json(refusal(None, 400, "the body is not a JSON object with a positive Content-Length"))
                return
            request["op"] = op
            self._json(analyst.submit(request))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server's own signature
            return

    return Handler


def make_server(bind: str, port: int, analyst: Analyst, page: bytes) -> ThreadingHTTPServer:
    """A bound, not yet serving, `ThreadingHTTPServer`; port 0 asks the OS for one (tests)."""
    return ThreadingHTTPServer((bind, port), _handler_class(analyst, page))


def run_server(infos: list[EngineInfo], args: argparse.Namespace) -> int:
    """`serve`: build the dispatcher, start the analyst, serve until interrupted; 0 on a clean stop."""
    dispatcher = Dispatcher(infos, device=args.device, threads=args.threads, strix=args.strix)
    analyst = Analyst(dispatcher.handle, timeout_sec=args.timeout_sec, on_stop=dispatcher.close)
    analyst.start()
    httpd = make_server(args.bind, args.port, analyst, render().encode("utf-8"))
    unsafe = "" if args.bind == "127.0.0.1" else "  (UNSAFE: not loopback — no auth, no TLS)"
    print(f"mantis analyzer on http://{args.bind}:{httpd.server_address[1]}/ — {len(dispatcher.infos)} engine row(s); "
          f"Ctrl-C stops{unsafe}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        analyst.stop()
    return 0


__all__ = ["Analyst", "make_server", "run_server", "supersession_key"]
