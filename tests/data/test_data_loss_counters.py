# >300 justify (R8). ONE closure: the counter, its producer at every arm, and its destination
# are the same claim; split apart, a producer file passes while the sink key is gone and a
# census passes while nothing feeds it. The census and its mutation self-test sit beside the
# arms they police so adding an arm reds the file that owns it.
"""Every `data/**` skip/truncate loss is COUNTED and READ.

Each test DRIVES the failure — nothing is asserted from source text — and then asserts the
value reaches its destination, since a counter only a test can see is half-wired. The
offline arms (`corpus_analysis`, `corpus_metrics`, `generate`, `human_seeding`,
`sources/human`) feed `PIPELINE_COUNTERS` and reach `log_pipeline_losses`.

The mutation pin is a mechanical census proving `src/mantis/data/**` holds zero blind-except
swallow arms outside the sanctioned wrapper, derived by scanning the source rather than from
a transcribed line list, with its own self-test proving the detector bites a planted swallow.
"""
from __future__ import annotations

import ast
import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import mantis.data
import mantis.data.loss_counters as lc
from mantis.data.corpus_analysis import load_all_games
from mantis.data.corpus_metrics import analyse_opening_diversity
from mantis.data.generate import _play_one_game, load_cached_bot_games
from mantis.data.human_seeding import _build_file_index
from mantis.data.sources.base import GameRecord
from mantis.data.sources.human import HumanGameSource
from mantis.monitor.best_effort import BestEffortCounters

_DATA_ROOT = Path(mantis.data.__file__).resolve().parent

# A game whose third move replays the first cell — the engine raises `cell already
# occupied`, which is the illegal-move truncation every replayer swallows.
_ILLEGAL = [(0, 0), (1, 0), (0, 0), (2, 0)]
def _delta(counters: BestEffortCounters, label: str, fn: Callable[[], object]) -> int:
    """Return the counts added under ``label`` by running ``fn``.

    A delta, not an absolute: the registries are process-global and accumulate across a run.
    """
    before = counters.get(label)
    fn()
    return counters.get(label) - before


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_unreadable_cached_bot_game_is_counted_and_skipped(tmp_path: Path) -> None:
    _write(tmp_path / "ok.json", {"moves": [{"x": 0, "y": 0}]})
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    games: list = []
    n = _delta(lc.PIPELINE_COUNTERS, "data.generate.cached_game_unreadable_skipped",
               lambda: games.extend(load_cached_bot_games(tmp_path)))
    assert n == 1 and len(games) == 1


def test_unreadable_human_source_game_is_counted_and_skipped(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    src = HumanGameSource(tmp_path)
    n = _delta(lc.PIPELINE_COUNTERS, "data.sources.human.game_unreadable_skipped",
               lambda: list(src))
    assert n == 1


def test_unreadable_human_seeding_index_entry_is_counted(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert _delta(lc.PIPELINE_COUNTERS, "data.human_seeding.index_game_unreadable_skipped",
                  lambda: _build_file_index(str(tmp_path), 1)) == 1


def test_malformed_bot_and_injected_corpus_games_are_counted(tmp_path: Path) -> None:
    human = tmp_path / "human"
    human.mkdir()
    (tmp_path / "bot" / "sealbot_fast").mkdir(parents=True)
    (tmp_path / "bot" / "sealbot_fast" / "bad.json").write_text("{no", encoding="utf-8")
    (tmp_path / "inj").mkdir()
    (tmp_path / "inj" / "bad.json").write_text("{no", encoding="utf-8")

    def _load() -> None:
        load_all_games(human, bot_games_dir=tmp_path / "bot",
                       injected_dir=tmp_path / "inj", include_bot_games=True)

    bot = _delta(lc.PIPELINE_COUNTERS,
                 "data.corpus_analysis.bot_game_malformed_skipped", _load)
    inj = _delta(lc.PIPELINE_COUNTERS,
                 "data.corpus_analysis.injected_game_malformed_skipped", _load)
    assert (bot, inj) == (1, 1)


def test_corpus_metrics_illegal_move_truncations_are_counted() -> None:
    records = [GameRecord(game_id_str="g", moves=_ILLEGAL, winner=1, source="human")]
    assert _delta(
        lc.PIPELINE_COUNTERS,
        "data.corpus_metrics.opening_diversity_illegal_move_truncated_replay",
        lambda: analyse_opening_diversity(records, encoding_name="gnn_axis_r8"),
    ) == 1


class _BotThatExplodes:
    def reset(self) -> None: ...
    def name(self) -> str: return "explodes"
    def get_move(self, state, board):
        raise RuntimeError("bot is broken (test)")


def test_generate_seeding_fallback_and_bot_move_error_are_counted(tmp_path: Path) -> None:
    """Drive both `generate.py` in-game arms: seeding falls back to random, then the bot raises."""
    seed_label = "data.generate.human_seeding_failed_fallback_random"
    bot_label = "data.generate.bot_move_error_truncated_game"
    before = (lc.PIPELINE_COUNTERS.get(seed_label), lc.PIPELINE_COUNTERS.get(bot_label))
    result = _play_one_game(_BotThatExplodes(), 0, encoding_name="gnn_axis_r8", use_human_seeding=True,
                            human_corpus_dir=str(tmp_path))
    after = (lc.PIPELINE_COUNTERS.get(seed_label), lc.PIPELINE_COUNTERS.get(bot_label))
    assert (after[0] - before[0], after[1] - before[1]) == (1, 1)
    assert result is None, "a bot-truncated game with no winner is still dropped"


class _BotWithBadMoveShape:
    def reset(self) -> None: ...
    def name(self) -> str: return "bad_shape"
    def get_move(self, state, board): return (1, 2, 3)


def test_the_wrapper_covers_everything_the_old_try_covered(tmp_path: Path) -> None:
    """Prove the wrapper covers everything the old inline `try` blocks did.

    Those blocks spanned more than the raising call — the move unpack and the eligibility
    comparison — so wrapping only the call would turn a counted skip into a live TypeError.
    """
    _write(tmp_path / "weird.json", {"moveCount": "twenty", "moves": []})
    n = _delta(lc.PIPELINE_COUNTERS, "data.human_seeding.index_game_unreadable_skipped",
               lambda: _build_file_index(str(tmp_path), 1))
    assert n == 1, "a non-comparable moveCount must be a COUNTED skip, not a raise"

    m = _delta(lc.PIPELINE_COUNTERS, "data.generate.bot_move_error_truncated_game",
               lambda: _play_one_game(_BotWithBadMoveShape(), 0, encoding_name="gnn_axis_r8"))
    assert m == 1, "a bot returning a non-pair must be a COUNTED skip, not a raise"


def test_log_pipeline_losses_publishes_the_snapshot(caplog) -> None:
    lc.PIPELINE_COUNTERS.increment("data.sources.human.game_unreadable_skipped")
    with caplog.at_level(logging.INFO, logger="mantis.data.loss_counters"):
        snapshot = lc.log_pipeline_losses("test")
    assert snapshot["data.sources.human.game_unreadable_skipped"] >= 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("data_pipeline_losses" in m and "game_unreadable_skipped" in m
               for m in messages), messages


def test_offline_entry_points_flush_their_losses(tmp_path: Path, caplog) -> None:
    """Prove the offline registry has a live consumer at the entry point, not only in this file."""
    (tmp_path / "bad.json").write_text("{no", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="mantis.data.loss_counters"):
        load_cached_bot_games(tmp_path)
    flushes = [r.getMessage() for r in caplog.records if "data_pipeline_losses" in r.getMessage()]
    assert any("load_cached_bot_games" in m for m in flushes), flushes


_BLIND = {"Exception", "BaseException"}


def _blind_except_sites(root: Path) -> list[str]:
    """Return every blind `except` handler under ``root``, derived by scanning the tree.

    The sanctioned wrapper `mantis.monitor.best_effort` lives outside this root, so the
    expected count here is exactly zero.
    """
    sites: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            exc = node.type
            if exc is None or (isinstance(exc, ast.Name) and exc.id in _BLIND):
                sites.append(f"{path.relative_to(root)}:{node.lineno}")
    return sites


def test_data_holds_zero_blind_except_swallow_arms() -> None:
    sites = _blind_except_sites(_DATA_ROOT)
    assert sites == [], (
        "every optional effect in data/** goes through `best_effort` (counted) or fails "
        f"loud; blind excepts found: {sites}"
    )


def test_the_census_bites_a_planted_swallow(tmp_path: Path) -> None:
    """Prove the census is not vacuous: each planted swallow form, including bare `except:`, bites."""
    (tmp_path / "planted_pass.py").write_text(
        "def f():\n    try:\n        g()\n    except Exception:\n        pass\n",
        encoding="utf-8")
    (tmp_path / "planted_continue.py").write_text(
        "def f():\n    for x in y:\n        try:\n            g()\n"
        "        except Exception:  # noqa: BLE001\n            continue\n",
        encoding="utf-8")
    (tmp_path / "planted_bare.py").write_text(
        "def f():\n    try:\n        g()\n    except:\n        pass\n", encoding="utf-8")
    assert len(_blind_except_sites(tmp_path)) == 3


def test_the_census_does_not_fire_on_a_narrow_except(tmp_path: Path) -> None:
    """Prove the census leaves narrow handlers alone; firing on them would push fixes toward blind ones."""
    (tmp_path / "narrow.py").write_text(
        "try:\n    import rich\nexcept ImportError:\n    rich = None\n", encoding="utf-8")
    assert _blind_except_sites(tmp_path) == []
