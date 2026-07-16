"""Typed surface of the compiled mantis._engine extension (crates/mantis-bridge).

Twin of src/mantis/_engine.pyi (the type-checker-visible copy in the editable src
package). This copy ships in the mantis-engine wheel beside the compiled module and
also keeps the PEP 420 namespace dir (python/mantis/) tracked so fresh clones build.
Keep both stubs identical when the bridge API changes.
"""

def hello() -> str: ...
def workspace_crates() -> list[str]: ...
