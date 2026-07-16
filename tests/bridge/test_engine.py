"""Bridge liveness: the compiled mantis._engine module and its DAG-edge surface."""
from mantis import _engine


def test_engine_importable_with_doc():
    assert _engine.__doc__


def test_hello():
    assert _engine.hello() == "mantis._engine alive"


def test_workspace_crates_set():
    assert set(_engine.workspace_crates()) == {
        "mantis-core",
        "mantis-graph",
        "mantis-encoding",
        "mantis-search",
        "mantis-selfplay",
    }
