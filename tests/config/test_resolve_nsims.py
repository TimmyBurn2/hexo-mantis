"""O2 — eval model_sims decision-equivalence (resolve/nsims.resolve_eval_model_sims).

REBUILD: the code-side {random:96, sealbot:128} default dict DIES; the per-opponent value
is a required schema field the resolver READS. Config value always wins; unknown opponent
raises; None raises (Δ-REBUILD — frozen fell to the code default).
"""
import pytest

from mantis.config.resolve.nsims import resolve_eval_model_sims


def test_random_reads_config_value():
    assert resolve_eval_model_sims("random", 96) == 96


def test_sealbot_reads_config_value():
    assert resolve_eval_model_sims("sealbot", 128) == 128


def test_config_value_always_wins():
    assert resolve_eval_model_sims("random", 64) == 64


def test_unknown_opponent_raises():
    with pytest.raises(ValueError):
        resolve_eval_model_sims("mystery", 96)


def test_none_value_raises_delta_rebuild():
    # Δ-REBUILD: frozen returned 96; the config field is required, no code default to fall to.
    with pytest.raises(ValueError):
        resolve_eval_model_sims("random", None)
