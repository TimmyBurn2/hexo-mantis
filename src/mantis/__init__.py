"""mantis: AlphaZero-style self-play bot for Hex Tac Toe (public package root)."""
# The compiled extension (mantis._engine, built from crates/mantis-bridge) is installed
# by a separate wheel into the same package namespace. Extend the package __path__ so the
# editable src copy and the installed extension resolve as one package. This is a
# __path__ extension, NOT a sys.path write (those are banned repo-wide).
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
