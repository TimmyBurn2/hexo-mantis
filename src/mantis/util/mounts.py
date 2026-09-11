"""Where the mount table is read from: the kernel's, or a planted one for a test drive."""
from __future__ import annotations

import os
from pathlib import Path

#: Where the kernel publishes the mount table. Parameterised so the parse has a mutation test.
MOUNTS = Path("/proc/mounts")
#: A test drive points the durability check at a planted table; the reading names the table it
#: used and `mantis.run` refuses a stamp taken from any table but `MOUNTS`, so it launches nothing.
MOUNTS_ENV = "MANTIS_PREFLIGHT_MOUNTS_TABLE"


def resolve_mounts_table() -> Path:
    """The mount table a preflight reads: `MOUNTS_ENV` when set, else the kernel's `MOUNTS`."""
    override = os.environ.get(MOUNTS_ENV)
    return Path(override) if override else MOUNTS
