"""A vendor root pinning a FAKE executable `sixengine` and a network, laid out as the real Six pin is."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

_FAKE_ENGINE = '''#!{python}
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
for line in {stderr!r}:
    print(line, file=sys.stderr, flush=True)
if {dies!r}:
    sys.exit(1)
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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_vendor(tmp_path: Path, *, stderr: Sequence[str] = (), net_sha: str | None = None,
                ident: str = "HexBot Net", dies: bool = False) -> Path:
    """The vendor root; `dies` makes the engine exit 1 after its stderr lines, before `sixok`."""
    root = tmp_path / "vendor"
    engine = root / "external" / "six-assets" / "release" / "engine" / "sixengine"
    net = root / "external" / "six-assets" / "gen-0030.onnx"
    engine.parent.mkdir(parents=True)
    engine.write_text(_FAKE_ENGINE.format(python=sys.executable, stderr=list(stderr), dies=dies, ident=ident),
                      encoding="utf-8")
    engine.chmod(0o755)
    net.write_bytes(b"a network")
    (root / "pins.toml").write_text(
        "[pins.six]\nurl = \"https://example.invalid/six.git\"\nsha = \"" + "a" * 40 + "\"\n"
        f"[pins.six.assets.engine]\nsha256 = \"{sha256_of(engine)}\"\npath = \"six-assets/release/engine/sixengine\"\n"
        f"[pins.six.assets.gen0030]\nsha256 = \"{net_sha or sha256_of(net)}\"\npath = \"six-assets/gen-0030.onnx\"\n",
        encoding="utf-8")
    return root
