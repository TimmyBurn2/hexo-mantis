"""Module entry point — `python -m mantis.encoding [audit] [...]`.

See mantis/encoding/audit.py.
"""
from __future__ import annotations

import sys

from mantis.encoding.audit import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
