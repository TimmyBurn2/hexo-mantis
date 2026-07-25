"""`StrictModel` — the ONE base every schema section subclasses (§10 package split).

Lives in its own leaf module (no imports from `.core`/`.train`/`.selfplay`/`.monitor`) so the
package's internal import graph stays a DAG: `.core` imports `.train` (RunConfig needs
TrainConfig) and `.train` needs `StrictModel` — importing it from `.core` would create
`.core -> .train -> .core`, the exact top-level cycle `tools/check_import_dag.py` (CI gate 9)
rejects. Every section module (`.core`, `.train`, `.selfplay`, `.monitor`) imports `StrictModel`
from here instead.
"""
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every config model: unknown key = hard error, no coercion, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
