"""The smallest GSPRT a round can play, two pairs read at one check, for tests whose subject is not the rule."""
from __future__ import annotations

TWO_PAIR_GSPRT: dict = {"mu0": 0.42, "mu1": 0.52, "alpha": 0.05, "beta": 0.1, "check_every_pairs": 2,
                        "min_pairs": 2, "max_pairs": 2, "at_max_pairs": "promote"}
