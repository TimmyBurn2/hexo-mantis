"""Corpus IO / sources / pipeline metrics.

The lightweight, most-used surface is re-exported here. The heavier CLI /
analysis modules (``corpus_metrics``, ``corpus_analysis``, ``corpus_reporter``,
``generate``, ``human_seeding``) are imported via their fully-qualified paths so
that ``import mantis.data`` does not pull in optional matplotlib/rich or trigger
the report-dir side effect.

The dense replayers and the D6 policy-scatter LUTs went with the grid path (R346(f)); the
graph corpus encoder is `mantis.data.bootstrap_encode`.
"""
from mantis.data.corpus_io import (
    SCHEMA_VERSION,
    CorpusMetadataError,
    compute_npz_sha256,
    load_corpus,
    save_corpus,
    validate_corpus_sidecar,
)
from mantis.data.pipeline_metrics import CorpusMetrics, SourceMetrics
from mantis.data.sources import CorpusSource, GameRecord

__all__ = [
    "CorpusMetadataError",
    "CorpusMetrics",
    "CorpusSource",
    "GameRecord",
    "SCHEMA_VERSION",
    "SourceMetrics",
    "compute_npz_sha256",
    "load_corpus",
    "save_corpus",
    "validate_corpus_sidecar",
]
