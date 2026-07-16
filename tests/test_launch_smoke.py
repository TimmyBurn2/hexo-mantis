"""Launch-path smoke (integration tier): package + engine import in a fresh interpreter."""
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_launch_smoke_subprocess():
    res = subprocess.run(
        [sys.executable, "-c", "import mantis; from mantis import _engine; assert _engine.hello()"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
