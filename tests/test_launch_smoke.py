"""Launch-path smoke (integration tier): package + engine import in a fresh interpreter."""
import subprocess
import sys

import pytest

from mantis import _engine


@pytest.mark.integration
def test_launch_smoke_subprocess():
    """A FRESH interpreter imports the package and sees the SAME registry this one does.

    The count is DERIVED from the parent's live registry and handed to the child, never typed.
    It used to be a literal `4` inside this string, which made the row a second authority over
    the size of the registered set — and an invisible one: the number sits in a code string with
    no encoding name near it, so no name-based search finds it, and the row is integration-marked
    so the default tier never runs it. R328(b) registered a fifth encoding and this was the last
    place in the tree still saying four (R192(e), derive-or-delete).

    What the row actually claims is unchanged and is now what it checks: a fresh interpreter can
    import `mantis`, load the extension, and read the registry the parent already read.
    """
    expected = len(_engine.all_specs())
    assert expected > 0, "the parent's registry is empty; this row cannot show a child loaded one"
    code = (
        "import mantis; from mantis import _engine; "
        f"assert len(_engine.all_specs()) == {expected}"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
