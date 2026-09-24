"""`resolve_eval_model_sims`: the config value wins; an unknown opponent or a None value raises."""
import pytest

from mantis.config.resolve.nsims import resolve_eval_model_sims


def test_random_reads_config_value():
    assert resolve_eval_model_sims("random", 96) == 96


def test_the_retired_sealbot_opponent_is_unknown():
    with pytest.raises(ValueError, match="unknown eval opponent 'sealbot'"):
        resolve_eval_model_sims("sealbot", 128)


def test_unknown_opponent_raises():
    with pytest.raises(ValueError):
        resolve_eval_model_sims("mystery", 96)


def test_none_value_raises_delta_rebuild():
    # Δ-REBUILD: frozen returned 96; the config field is required, no code default to fall to.
    with pytest.raises(ValueError):
        resolve_eval_model_sims("random", None)
