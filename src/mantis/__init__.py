"""mantis: AlphaZero-style self-play bot for Hex Tac Toe (public package root)."""
# mantis._engine is a separate wheel in this namespace: extending __path__ makes the src copy and
# it resolve as one package (a __path__ extension, NOT a banned sys.path write).
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
