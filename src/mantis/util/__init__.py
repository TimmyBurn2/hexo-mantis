"""mantis.util — small helper submodules (leaves: import NOTHING from `mantis`).

Intentionally does NOT re-export from sibling modules, so that importing
``mantis.util`` (or ``from mantis.util.X import ...``) does NOT trigger a
torch / numpy import via sibling re-exports. In particular ``device`` imports
torch at module top; re-exporting it here would pull torch into package init and
break the contract that ``mantis.util.cpu_budget`` must be importable BEFORE
numpy / torch (it sets OMP_NUM_THREADS et al.).

Callers should always use the fully-qualified submodule path:
    from mantis.util.constants import HISTORY_LEN
    from mantis.util.coordinates import axial_to_flat
    from mantis.util.cpu_budget import apply_auto_thread_budget
    from mantis.util.device import best_device   # explicit torch-consumers only
"""
