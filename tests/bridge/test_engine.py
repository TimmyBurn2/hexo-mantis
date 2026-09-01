"""Bridge liveness + WP0 close-out re-prove (O16): the compiled mantis._engine
module imports and exposes the full assembled surface via the unified `mantis`
namespace (extend_path, no sys.path write). Full inventory + F-42 live in
test_surface.py; this pins the import path and the headline all_specs() re-prove.
"""
from mantis import _engine


def test_engine_importable_with_doc():
    assert _engine.__doc__


def test_all_specs_reprove():
    """`import mantis._engine; all_specs()` under the full pyclass surface."""
    specs = _engine.all_specs()
    assert len(specs) == 5
    assert {s.name for s in specs} == {
        "v6", "v6w25", "v6_live2_ls", "gnn_axis_v1", "gnn_axis_r8"}


def test_registry_sha_reprove():
    assert len(_engine.registry_sha()) == 32
    assert len(_engine.registry_sha_hex()) == 64
