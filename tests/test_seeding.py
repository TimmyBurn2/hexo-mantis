"""Seeding doctrine pins (repo_design §8): autouse reseed + seeded-lib visibility."""
import importlib.util
import os
import random

PYTEST_SEED = int(os.environ.get("PYTEST_SEED", "20260716"))


def test_reseed_gives_identical_rng_state():
    # The autouse conftest fixture reseeds before every test, so the first draw here
    # must equal the first draw of a fresh generator seeded the same way.
    assert random.random() == random.Random(PYTEST_SEED).random()


def test_seeded_libs_mirror_availability(seeded_libs):
    assert "random" in seeded_libs
    assert ("numpy" in seeded_libs) == (importlib.util.find_spec("numpy") is not None)
    assert ("torch" in seeded_libs) == (importlib.util.find_spec("torch") is not None)
