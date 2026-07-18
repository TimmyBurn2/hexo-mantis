"""unsendable (O13, LOCKED #3): `Board` (PyBoard) and `MCTSTree` (PyMCTSTree)
hold a `Send + !Sync` core `Board` and reject cross-thread access (single-thread
Python ownership); the other 9 pyclasses are send-safe. NO `unsafe impl Sync for
Board` anywhere (WP2 soundness).

The unsendable guard fires as a pyo3 assertion that crosses the FFI as a catchable
PanicException — the process survives, the object is simply refused off-thread.
"""
import threading

from mantis import _engine


def _access_on_new_thread(fn):
    """Run `fn` on a fresh thread; return the exception type name or 'ok'."""
    result = {}

    def worker():
        try:
            fn()
            result["r"] = "ok"
        except BaseException as exc:  # noqa: BLE001 — classifying the guard
            result["r"] = type(exc).__name__

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return result["r"]


def test_board_rejects_cross_thread_access(panic_exception):
    board = _engine.Board.with_encoding_name("v6")
    board.apply_move(0, 0)
    assert _access_on_new_thread(lambda: board.apply_move(1, 0)) == panic_exception.__name__
    # Same-thread use is unaffected.
    board.apply_move(1, 0)
    assert board.ply == 2


def test_mctstree_rejects_cross_thread_access(panic_exception):
    tree = _engine.MCTSTree()
    assert _access_on_new_thread(tree.root_visits) == panic_exception.__name__
    # Same-thread use is unaffected.
    assert tree.root_visits() == 0


def test_send_safe_classes_cross_thread_ok():
    """The send-safe pyclasses (Arc/atomic-backed) are usable off-thread."""
    rb = _engine.ReplayBuffer(8, "v6")
    assert _access_on_new_thread(lambda: rb.size) == "ok"
    hb = _engine.HexgBuffer(8, "gnn_axis_v1")
    assert _access_on_new_thread(lambda: hb.size) == "ok"
    ts = _engine.TacticalSolver()
    assert _access_on_new_thread(lambda: ts.__class__) == "ok"


def test_no_unsafe_impl_sync_for_board():
    """Grep-gate mirror (O13/O14): the WP2-dropped `unsafe impl Sync for Board`
    never reappears in any crate source."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    hits = []
    for rs in (repo / "crates").rglob("*.rs"):
        text = rs.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            is_comment = stripped.startswith(("//", "///", "//!", "*"))
            if "unsafe impl Sync for Board" in stripped and not is_comment:
                hits.append(f"{rs}:{i}")
    assert not hits, f"unsafe impl Sync for Board reappeared: {hits}"
