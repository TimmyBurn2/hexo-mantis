"""O1 — the encoding resolver: a declared name passes through, an absent one RAISES."""
import pytest

from mantis.config.resolve.encoding import AbsentEncodingError, reconcile_encoding


def test_a_declared_encoding_is_the_resolved_one():
    assert reconcile_encoding("gnn_axis_r8") == "gnn_axis_r8"


@pytest.mark.parametrize("declared", [None, ""])
def test_an_absent_encoding_raises_with_no_terminal_default(declared):
    with pytest.raises(AbsentEncodingError):
        reconcile_encoding(declared)


def test_the_absent_error_is_a_valueerror():
    assert issubclass(AbsentEncodingError, ValueError)
