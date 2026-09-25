"""`all_specs()` iterates in name order in every process, so parametrized collection is stable."""
from __future__ import annotations

import subprocess
import sys

from mantis.encoding import all_specs

# Enough fresh processes that a per-process-seeded order is caught with probability 1 - 2**-8.
_PROCESSES = 8
_PROBE = "from mantis.encoding import all_specs; print(','.join(s.name for s in all_specs()))"


def test_all_specs_is_name_sorted_in_this_process() -> None:
    names = [s.name for s in all_specs()]
    assert names == sorted(names), names


def test_all_specs_order_is_identical_across_processes() -> None:
    orders = {
        subprocess.run(
            [sys.executable, "-c", _PROBE], check=True, capture_output=True, text=True, encoding="utf-8"
        ).stdout.strip()
        for _ in range(_PROCESSES)
    }
    assert len(orders) == 1, orders
    names = orders.pop().split(",")
    assert names == sorted(names), names
