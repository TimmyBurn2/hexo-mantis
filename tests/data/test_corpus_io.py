"""O4a — corpus_io sha round-trip + sidecar validation.

save_corpus → load_corpus round-trips arrays byte-exact; the sidecar carries the correct
sha256 / encoding_name / schema_version; a wrong ``expected_encoding`` on load raises
``CorpusMetadataError``; a tampered ``.npz`` (sha mismatch) raises ``CorpusMetadataError``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from mantis.data.corpus_io import (
    SCHEMA_VERSION,
    CorpusMetadataError,
    compute_npz_sha256,
    load_corpus,
    save_corpus,
    validate_corpus_sidecar,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "states": np.arange(2 * 4, dtype=np.float16).reshape(2, 4),
        "policies": np.linspace(0.0, 1.0, 2 * 3, dtype=np.float32).reshape(2, 3),
        "outcomes": np.array([-1.0, 1.0], dtype=np.float32),
    }


def test_save_load_round_trip_byte_exact(tmp_path) -> None:
    path = tmp_path / "corpus.npz"
    arrays = _arrays()
    save_corpus(path, arrays=arrays, encoding_name="v6")

    loaded, meta = load_corpus(path, expected_encoding="v6")
    for k, v in arrays.items():
        assert np.array_equal(loaded[k], v)
        assert loaded[k].dtype == v.dtype

    assert meta["encoding_name"] == "v6"
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["n_positions"] == 2
    assert meta["sha256"] == compute_npz_sha256(path)


def test_sidecar_contents(tmp_path) -> None:
    path = tmp_path / "corpus.npz"
    save_corpus(path, arrays=_arrays(), encoding_name="v6w25", source_manifest="m1")
    sidecar = json.loads((tmp_path / "corpus.npz.metadata.json").read_text())
    assert sidecar["encoding_name"] == "v6w25"
    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["source_manifest"] == "m1"
    assert sidecar["sha256"] == compute_npz_sha256(path)


def test_wrong_expected_encoding_raises(tmp_path) -> None:
    path = tmp_path / "corpus.npz"
    save_corpus(path, arrays=_arrays(), encoding_name="v6")
    with pytest.raises(CorpusMetadataError, match="encoding mismatch"):
        load_corpus(path, expected_encoding="v6w25")


def test_tampered_npz_raises(tmp_path) -> None:
    path = tmp_path / "corpus.npz"
    save_corpus(path, arrays=_arrays(), encoding_name="v6")
    # Corrupt the archive bytes without touching the sidecar → sha mismatch.
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CorpusMetadataError, match="sha256 mismatch"):
        validate_corpus_sidecar(path)
    with pytest.raises(CorpusMetadataError, match="sha256 mismatch"):
        load_corpus(path)


def test_missing_sidecar_warns_not_raises(tmp_path) -> None:
    path = tmp_path / "bare.npz"
    np.savez_compressed(path, states=np.zeros((1, 2), dtype=np.float16))
    with pytest.warns(DeprecationWarning):
        arrays, meta = load_corpus(path)
    assert meta == {}
    assert "states" in arrays


def test_empty_arrays_raises(tmp_path) -> None:
    with pytest.raises(CorpusMetadataError, match="arrays dict is empty"):
        save_corpus(tmp_path / "x.npz", arrays={}, encoding_name="v6")
