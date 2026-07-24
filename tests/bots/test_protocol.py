"""⊕ WP11-A bots — BotProtocol + RandomBot + resolve_bot (design §a.2, §b bots suite).

RED-at-import until IMPL writes `mantis.bots.protocol` / `mantis.bots.random_bot` /
`mantis.bots.resolve`. ORACLE-FIRST (⊕): the top-level import raises ModuleNotFoundError
before any port code exists.

Rung-resolution census at HEAD (DESIGN.md census verdict): 0 of 6 ladder rungs resolve
locally — sealbot/kraken/strix are WP12-R property. `resolve_bot` must raise
`RungUnresolvable` for each, with a reason string that DISTINGUISHES "env key unset" from
"env key set but no adapter installed" (dispatch: env keys / vendor pins only, no host
path, no default endpoint — a set-but-unadapted env key is not a silent fallback to a
default host).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis.bots import BotProtocol, RandomBot, RungUnresolvable, resolve_bot

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis" / "bots"

_KNOWN_KINDS = ("random", "sealbot", "kraken", "strix")
_ENV_KEYS = {
    "sealbot": "MANTIS_BOT_SEALBOT",
    "kraken": "MANTIS_BOT_KRAKEN",
    "strix": "MANTIS_BOT_STRIX",
}


class _FakeBoard:
    """Minimal duck-typed stand-in for `mantis._engine.Board.legal_moves()`."""

    def __init__(self, legal: list[tuple[int, int]]) -> None:
        self._legal = list(legal)

    def legal_moves(self) -> list[tuple[int, int]]:
        return list(self._legal)


def test_random_bot_is_legal_and_seed_deterministic():
    legal = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    board = _FakeBoard(legal)
    bot_a = RandomBot(seed=1234)
    bot_b = RandomBot(seed=1234)
    moves_a = [bot_a.select_move(board) for _ in range(10)]
    moves_b = [bot_b.select_move(board) for _ in range(10)]
    assert moves_a == moves_b, "same seed + same position sequence must replay identically"
    assert all(m in legal for m in moves_a), "every selected move must be legal"
    bot_c = RandomBot(seed=5678)
    moves_c = [bot_c.select_move(board) for _ in range(10)]
    assert moves_c != moves_a, "a different seed must (overwhelmingly likely) diverge"


def test_random_bot_satisfies_bot_protocol():
    bot = RandomBot(seed=1)
    assert isinstance(bot, BotProtocol)
    assert isinstance(bot.name(), str)
    bot.new_game()  # must not raise — stateless reset


def test_resolver_resolves_random_locally():
    factory = resolve_bot("random", depth=None, opponent_sims=None)
    bot = factory()
    assert isinstance(bot, BotProtocol)
    board = _FakeBoard([(0, 0), (1, 1)])
    move = bot.select_move(board)
    assert move in [(0, 0), (1, 1)]


@pytest.mark.parametrize("kind", ["sealbot", "kraken", "strix"])
def test_external_kinds_unresolvable_at_head_with_reason(kind, monkeypatch):
    env_key = _ENV_KEYS[kind]
    monkeypatch.delenv(env_key, raising=False)
    with pytest.raises(RungUnresolvable) as unset_exc:
        resolve_bot(kind, depth=5, opponent_sims=128)
    assert unset_exc.value.rung == kind
    unset_reason = unset_exc.value.reason
    assert "env key" in unset_reason and "unset" in unset_reason

    # Env key SET but no WP12-R adapter package installed at HEAD — a DIFFERENT reason,
    # never a silent fallback to a default host/path (dispatch law: env keys / vendor
    # pins only).
    monkeypatch.setenv(env_key, "some_adapter_module:build")
    with pytest.raises(RungUnresolvable) as set_exc:
        resolve_bot(kind, depth=5, opponent_sims=128)
    assert set_exc.value.rung == kind
    set_reason = set_exc.value.reason
    assert "no adapter installed" in set_reason or "WP12-R" in set_reason
    assert set_reason != unset_reason, (
        "env-key-unset and env-key-set-but-unadapted must be DISTINGUISHABLE reasons"
    )


def test_no_host_path_tokens_in_bots_sources():
    for path in _SRC.rglob("*.py"):
        text = path.read_text()
        assert "/home/" not in text, f"{path} carries a host-specific /home/ path literal"
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v.startswith("/") and len(v) > 1 and "/" in v[1:]:
                    raise AssertionError(
                        f"{path} carries an absolute path literal: {v!r} — bot endpoints "
                        "must resolve only through env keys / vendor pins"
                    )


def test_unknown_bot_kind_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        resolve_bot("mystery_bot", depth=None, opponent_sims=None)
    msg = str(exc.value)
    for kind in _KNOWN_KINDS:
        assert kind in msg, f"ValueError should name the known kind set, missing {kind!r}"
