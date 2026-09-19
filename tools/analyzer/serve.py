"""The loopback server: ONE analyst thread (every engine, Board and tree lives on it) with supersession, three routes."""
from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .dispatch import Dispatcher

Handler = Callable[[dict[str, Any]], dict[str, Any]]
_Item = tuple[dict[str, Any], "Future[dict[str, Any]]"]
_ROUTES = {"/analyze": "analyze", "/trace": "trace"}


def supersession_key(request: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    """`(client, engine, op, raw|search)`: a newer request with the same key replaces a queued one."""
    tier = "raw" if int(request.get("sims", 0) or 0) == 0 else "search"
    return request.get("client"), request.get("engine"), request.get("op"), tier


class Analyst:
    """The one worker thread; `submit` waits on a Future up to `timeout_sec` and answers 504 past it."""

    def __init__(self, handler: Handler, *, timeout_sec: float) -> None:
        self._handler = handler
        self.timeout_sec = float(timeout_sec)
        self._queue: queue.Queue[_Item | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="analyst", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Enqueue and wait; the reply is the handler's `{status, body}` or a 504 with the wait named."""
        fut: Future[dict[str, Any]] = Future()
        self._queue.put((request, fut))
        try:
            return fut.result(timeout=self.timeout_sec)
        except TimeoutError:
            return {"status": 504, "body": {"seq": request.get("seq"), "ok": False,
                                            "refused": f"the analyst did not answer within {self.timeout_sec} s"}}

    def _drain(self, first: _Item) -> list[_Item]:
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
        survivors: dict[tuple[Any, Any, Any, str], _Item] = {}
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
                return
            for req, fut in self._drain(item):
                try:
                    fut.set_result(self._handler(req))
                except BaseException as exc:  # noqa: BLE001 — a PanicException is a BaseException; the analyst must outlive it
                    fut.set_result({"status": 500, "body": {"seq": req.get("seq"), "ok": False,
                                                            "refused": f"analyst exception: {type(exc).__name__}: {exc}"}})


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
                self._json({"status": 404, "body": {"ok": False, "refused": f"no route {path}"}})

        def do_POST(self) -> None:  # noqa: N802 — http.server's own name
            path = urlsplit(self.path).path
            op = _ROUTES.get(path)
            if op is None:
                self._json({"status": 404, "body": {"ok": False, "refused": f"no route {path}"}})
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                self._json({"status": 400, "body": {"ok": False, "refused": "the body is not a JSON object"}})
                return
            request["op"] = op
            self._json(analyst.submit(request))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server's own signature
            return

    return Handler


def make_server(bind: str, port: int, analyst: Analyst, page: bytes) -> ThreadingHTTPServer:
    """A bound, not yet serving, `ThreadingHTTPServer`; port 0 asks the OS for one (tests)."""
    return ThreadingHTTPServer((bind, port), _handler_class(analyst, page))


def run_server(infos: list[Any], args: Any) -> int:
    """`serve`: build the dispatcher, start the analyst, serve until interrupted; 0 on a clean stop."""
    from .html import render
    dispatcher = Dispatcher(infos, device=args.device, threads=args.threads, strix=args.strix)
    analyst = Analyst(dispatcher.handle, timeout_sec=args.timeout_sec)
    analyst.start()
    httpd = make_server(args.bind, args.port, analyst, render().encode("utf-8"))
    unsafe = "" if args.bind == "127.0.0.1" else "  (UNSAFE: not loopback — no auth, no TLS)"
    print(f"mantis analyzer on http://{args.bind}:{httpd.server_address[1]}/ — {len(infos)} engine row(s); "
          f"Ctrl-C stops{unsafe}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        analyst.stop()
        dispatcher.close()
    return 0


__all__ = ["Analyst", "make_server", "run_server", "supersession_key"]
