"""Corpus IO / replay / augmentation LUTs / sources / pipeline metrics.

The lightweight, most-used surface is re-exported here. The heavier CLI /
analysis modules (``corpus_metrics``, ``corpus_analysis``, ``corpus_reporter``,
``generate``, ``human_seeding``) are imported via their fully-qualified paths so
that ``import mantis.data`` does not pull in optional matplotlib/rich or trigger
the report-dir side effect.
"""
from mantis.data.augment import get_policy_scatters
from mantis.data.corpus_io import (
    SCHEMA_VERSION,
    CorpusMetadataError,
    compute_npz_sha256,
    load_corpus,
    save_corpus,
    validate_corpus_sidecar,
)
from mantis.data.pipeline_metrics import CorpusMetrics, SourceMetrics
from mantis.data.replay import (
    ReplayTriples,
    replay_game_to_triples,
    replay_game_to_triples_ls,
    replay_game_to_triples_v6,
)
from mantis.data.replay_v6w25 import replay_game_to_triples_v6w25
from mantis.data.sources import CorpusSource, GameRecord

__all__ = [
    "CorpusMetadataError",
    "CorpusMetrics",
    "CorpusSource",
    "GameRecord",
    "ReplayTriples",
    "SCHEMA_VERSION",
    "SourceMetrics",
    "compute_npz_sha256",
    "get_policy_scatters",
    "load_corpus",
    "replay_game_to_triples",
    "replay_game_to_triples_ls",
    "replay_game_to_triples_v6",
    "replay_game_to_triples_v6w25",
    "save_corpus",
    "validate_corpus_sidecar",
]
