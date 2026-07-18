"""Launch-path smoke (integration tier): package + engine import in a fresh interpreter."""
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_launch_smoke_subprocess():
    code = "import mantis; from mantis import _engine; assert len(_engine.all_specs()) == 4"
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
