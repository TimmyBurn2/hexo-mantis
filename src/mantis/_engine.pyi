"""Typed surface of the compiled mantis._engine extension (crates/mantis-bridge).

Twin of crates/mantis-bridge/python/mantis/_engine.pyi (the wheel-shipped copy): this
copy makes the extension visible to type checkers against the editable src package —
a regular package shadows namespace portions, so the wheel copy alone is not seen.
Keep both stubs identical when the bridge API changes. Invisible to the runtime
importer (.pyi files are never imported); the real module is the compiled .so.
"""

def hello() -> str: ...
def workspace_crates() -> list[str]: ...
