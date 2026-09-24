"""mantis.util — small helper submodules (leaves: import NOTHING from `mantis`).

Intentionally does NOT re-export from sibling modules, so that importing
``mantis.util`` (or ``from mantis.util.X import ...``) does NOT trigger a
torch / numpy import via sibling re-exports. In particular ``device`` imports
torch at module top; re-exporting it here would pull torch into package init, and a
torch-free package init is the property every early-import leaf here depends on.

Callers should always use the fully-qualified submodule path:
    from mantis.util.constants import DRAW_RATE_WINDOW
    from mantis.util.coordinates import axial_distance
    from mantis.util.device import best_device   # explicit torch-consumers only
"""
