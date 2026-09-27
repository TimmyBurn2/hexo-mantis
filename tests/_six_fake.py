"""A vendor root pinning a FAKE executable `sixengine` and a network, laid out as the real Six pin is."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from mantis.util.hashing import sha256_file

_FAKE_ENGINE = '''#!{python}
import os
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
for line in {stderr!r}:
    print(line, file=sys.stderr, flush=True)
if {dies!r}:
    sys.exit(1)
(here / "env.txt").write_text(os.environ.get("LD_LIBRARY_PATH", ""))
log = (here / "received.txt").open("a")
log.write(" ".join(sys.argv[1:]) + "\\n")
for raw in sys.stdin:
    log.write(raw)
    log.flush()
    cmd = raw.split()
    if not cmd:
        continue
    if cmd[0] == "six":
        print("id name {ident}", flush=True)
        print("id version 0.1", flush=True)
        print("sixok", flush=True)
    elif cmd[0] == "isready":
        print("readyok", flush=True)
    elif cmd[0] == "go":
        print("bestmove none", flush=True)
    elif cmd[0] == "quit":
        break
'''


def fake_vendor(tmp_path: Path, *, stderr: Sequence[str] = (), net_sha: str | None = None,
                ident: str = "HexBot Net", dies: bool = False, runtime_sha: str | None = None) -> Path:
    """The vendor root; `dies` makes the engine exit 1 after its stderr lines, before `sixok`; `runtime_sha` pins a runtime."""
    root = tmp_path / "vendor"
    engine = root / "external" / "six-assets" / "release" / "engine" / "sixengine"
    net = root / "external" / "six-assets" / "gen-0030.onnx"
    engine.parent.mkdir(parents=True)
    engine.write_text(_FAKE_ENGINE.format(python=sys.executable, stderr=list(stderr), dies=dies, ident=ident),
                      encoding="utf-8")
    engine.chmod(0o755)
    net.write_bytes(b"a network")
    runtime = engine.parent / "libonnxruntime.so.1"
    runtime.write_bytes(b"a runtime")
    runtime_pin = ("" if runtime_sha is None else
                   f"[pins.six.assets.runtime]\nsha256 = \"{runtime_sha}\"\npath = \"six-assets/release/engine/libonnxruntime.so.1\"\n")
    (root / "pins.toml").write_text(
        "[pins.six]\nurl = \"https://example.invalid/six.git\"\nsha = \"" + "a" * 40 + "\"\n"
        f"[pins.six.assets.engine]\nsha256 = \"{sha256_file(engine)}\"\npath = \"six-assets/release/engine/sixengine\"\n"
        f"[pins.six.assets.gen0030]\nsha256 = \"{net_sha or sha256_file(net)}\"\npath = \"six-assets/gen-0030.onnx\"\n"
        + runtime_pin, encoding="utf-8")
    return root
