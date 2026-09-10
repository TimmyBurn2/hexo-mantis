# >300 justify (R8). ONE unit because it is ONE closure: the counter, its producer at every
# arm, and its destination are the same claim, and splitting them is exactly the half-wired
# state LAW-07 forbids — a producer file that passes while the sink key is gone, or a census
# that passes while nothing feeds it, would each be green in isolation and wrong together.
# The census and its mutation self-test must also sit beside the arms they police, or the
# next person to add an arm reds a file they never opened.
"""Item 8 — every `data/**` skip/truncate loss is COUNTED and READ (LAW-14 / LAW-18 / LAW-07).

`data/**` used to lose training rows and corpus games through twelve blind
`except Exception` arms plus three un-excepted off-window row drops. All of them were
invisible: a corpus where one file is corrupt and a corpus where EVERY game truncates at
ply 3 produced byte-identical logs.

Every test below DRIVES the failure — nothing is asserted from source text — and then
asserts the value reaches its DESTINATION, because a counter only a test can see is the
half-wired state LAW-07 forbids:

  - IN-RUN: the three dense replayers and their `REPLAY_COUNTERS` registry are DELETED with
    the grid path (R346(f)), and so is the `monitor_gates.data_loss_counters` key they fed —
    a registry with no producer publishes an always-empty mapping that reads as "nothing was
    lost" forever. What is left below is the OFFLINE half. The
    drives below call `_emit_monitor_gates` directly, so no cadence key is constructed here
    and the R242 split cannot silently re-point what these assert.
  - OFFLINE (`corpus_analysis`, `corpus_metrics`, `generate`, `human_seeding`,
    `sources/human`) → `PIPELINE_COUNTERS` → `log_pipeline_losses` on the data logger.

Plus the item's MUTATION PIN: a mechanical census proving `src/mantis/data/**` holds ZERO
blind-except swallow arms outside the sanctioned wrapper, derived by scanning the source
(never a transcribed line-number list — line numbers rot), carrying its own self-test
proving the detector BITES a planted bare `except Exception: pass`.
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
    """Counts added under ``label`` by running ``fn``. Delta, not absolute: the registries
    are process-global by design (they accumulate across a whole run)."""
    before = counters.get(label)
    fn()
    return counters.get(label) - before


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ══ OFFLINE producers ════════════════════════════════════════════════════════════════
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
    """Both `generate.py` in-game arms in one drive: an empty human corpus makes seeding
    raise (fallback to random), then the bot raises and ends the game."""
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
    """Behaviour-exactness of the conversion itself. The old inline `try` blocks covered
    MORE than the one raising call — the unpack of the bot's move, and the eligibility
    comparison against a `moveCount` that parses but is not comparable. Wrapping only the
    call would have let those escape as a live TypeError and killed a corpus job that used
    to skip one game, which is a REGRESSION dressed as a safety fix."""
    _write(tmp_path / "weird.json", {"moveCount": "twenty", "moves": []})
    n = _delta(lc.PIPELINE_COUNTERS, "data.human_seeding.index_game_unreadable_skipped",
               lambda: _build_file_index(str(tmp_path), 1))
    assert n == 1, "a non-comparable moveCount must be a COUNTED skip, not a raise"

    m = _delta(lc.PIPELINE_COUNTERS, "data.generate.bot_move_error_truncated_game",
               lambda: _play_one_game(_BotWithBadMoveShape(), 0, encoding_name="gnn_axis_r8"))
    assert m == 1, "a bot returning a non-pair must be a COUNTED skip, not a raise"


# ══ OFFLINE consumer — the counters reach the pipeline log ════════════════════════════
def test_log_pipeline_losses_publishes_the_snapshot(caplog) -> None:
    lc.PIPELINE_COUNTERS.increment("data.sources.human.game_unreadable_skipped")
    with caplog.at_level(logging.INFO, logger="mantis.data.loss_counters"):
        snapshot = lc.log_pipeline_losses("test")
    assert snapshot["data.sources.human.game_unreadable_skipped"] >= 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("data_pipeline_losses" in m and "game_unreadable_skipped" in m
               for m in messages), messages


def test_offline_entry_points_flush_their_losses(tmp_path: Path, caplog) -> None:
    """LAW-08: the offline registry has a LIVE consumer at the entry point that owns the
    arms — not only in this test file."""
    (tmp_path / "bad.json").write_text("{no", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="mantis.data.loss_counters"):
        load_cached_bot_games(tmp_path)
    flushes = [r.getMessage() for r in caplog.records if "data_pipeline_losses" in r.getMessage()]
    assert any("load_cached_bot_games" in m for m in flushes), flushes


# ══ THE CENSUS + its mutation self-test ══════════════════════════════════════════════
_BLIND = {"Exception", "BaseException"}


def _blind_except_sites(root: Path) -> list[str]:
    """Every `except:` / `except Exception:` / `except BaseException:` handler under ``root``.

    Derived by SCANNING the tree, never from a transcribed line list. The sanctioned
    wrapper `mantis.monitor.best_effort` owns the one blind except in the repo's optional-
    effect path and lives outside this root, so the expected count here is exactly zero.
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
    """LAW-07 mutation self-test: the detector is not vacuous. Each planted form is the exact
    shape item 8 removed — including the bare `except:` a future edit could reach for."""
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
    """The discriminating negative: `data/**`'s legitimate narrow handlers (optional-import
    `except ImportError`, `except (OSError, JSONDecodeError)`) are NOT swallow arms and the
    census must leave them alone — otherwise it would push a real fix toward `except Exception`."""
    (tmp_path / "narrow.py").write_text(
        "try:\n    import rich\nexcept ImportError:\n    rich = None\n", encoding="utf-8")
    assert _blind_except_sites(tmp_path) == []
