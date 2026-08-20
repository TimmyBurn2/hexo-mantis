"""mantis.util — small helper submodules (leaves: import NOTHING from `mantis`).

Intentionally does NOT re-export from sibling modules, so that importing
``mantis.util`` (or ``from mantis.util.X import ...``) does NOT trigger a
torch / numpy import via sibling re-exports. In particular ``device`` imports
torch at module top; re-exporting it here would pull torch into package init, and a
torch-free package init is the property every early-import leaf here depends on.
(The founding example of such a leaf, ``mantis.util.cpu_budget``, RELOCATED to
``tests/util/_cpu_budget.py`` under R289(q) after an AST census found it had no
production consumer at all. The no-torch-in-init rule is unchanged and still load-bearing;
only that one example moved.)

Callers should always use the fully-qualified submodule path:
    from mantis.util.constants import HISTORY_LEN
    from mantis.util.coordinates import axial_to_flat
    from mantis.util.device import best_device   # explicit torch-consumers only
"""
