# pyright: basic
"""Corpus distribution analysis (CLI residual).

Characterises the game corpus along several dimensions (game length, win rate,
move entropy, opening diversity, cluster count, ply coverage). The pure metric /
plot functions live in ``corpus_metrics``; rich Table console output in
``corpus_reporter``; this file is the argparse + main + corpus-loading glue.

Run as ``python -m mantis.data.corpus_analysis --human-dir <dir> ...``. All
corpus paths are explicit CLI arguments (no code-side default path — CLAUDE.md
R1; default-path resolution is a config-resolver concern). ``rich`` is optional
(lazily imported): console output degrades to no-ops when it is absent. pyright
is set to `basic` (untyped rich + json/stat dicts; DESIGN §f N7).
"""

from __future__ import annotations

import json
from pathlib import Path

from mantis.data._log import get_logger

# Re-export back-compat: external callers import analyse_ply_coverage etc. from
# this module path. main()/CLI uses corpus_metrics + corpus_reporter directly.
from mantis.data.corpus_metrics import (
    ELO_LABELS,
    REPORT_DIR,
    SOURCE_LABELS,
    _compute_per_game_entropies,
    _stratify,
    analyse_elo_stratified,
    analyse_ply_coverage,  # noqa: F401 — re-exported for back-compat
    analyse_quality_distribution,
    compute_quality_scores,
    run_analysis,
)
from mantis.data.corpus_reporter import (
    _print_elo_stratified_table,
    _print_summary_table,
)
from mantis.data.loss_counters import PIPELINE_COUNTERS, log_pipeline_losses
from mantis.data.sources.base import GameRecord
from mantis.data.sources.human import HumanGameSource
from mantis.monitor.best_effort import best_effort

log = get_logger(__name__)


def _rich_console():
    """Return a rich Console, or None when rich is unavailable (lazy import)."""
    try:
        from rich.console import Console  # type: ignore
    except ImportError:  # pragma: no cover - rich is an optional dependency
        return None
    return Console()


def _read_game_json(path: Path) -> dict:
    """Load one game JSON. Raises on unreadable/malformed input — the CALLER decides
    whether that is a counted skip (`best_effort`) or fatal."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bot_record(path: Path, source_label: str) -> GameRecord:
    data = _read_game_json(path)
    return GameRecord(
        game_id_str=path.stem,
        moves=[(m["x"], m["y"]) for m in data["moves"]],
        winner=data.get("winner", 0),
        source=source_label,
        metadata={"bot_name": data.get("bot_name", "unknown")},
    )


def _injected_record(path: Path) -> GameRecord:
    data = _read_game_json(path)
    return GameRecord(
        game_id_str=path.stem,
        moves=[(m["x"], m["y"]) for m in data["moves"]],
        winner=data.get("winner", 0),
        source="injected",
        metadata={
            "bot_name": data.get("bot_name", "unknown"),
            "injection_point": data.get("injection_point"),
            "human_moves": data.get("human_moves"),
            "bot_moves": data.get("bot_moves"),
        },
    )


def load_all_games(
    human_dir: str | Path,
    *,
    bot_games_dir: str | Path | None = None,
    injected_dir: str | Path | None = None,
    include_bot_games: bool = False,
) -> list[GameRecord]:
    """Load all games from available corpus sources.

    Args:
        human_dir: directory of scraped human game JSON files.
        bot_games_dir: root of bot self-play game JSON (searched for the
            ``sealbot_fast``/``sealbot_strong`` subdirs). Required when
            ``include_bot_games`` is True.
        injected_dir: directory of human-seed bot-continuation game JSON.
        include_bot_games: also load bot + injected games.
    """
    records: list[GameRecord] = []

    # Human games
    human_src = HumanGameSource(human_dir)
    human_count = len(human_src)
    log.info("loading_human_games", count=human_count)
    for rec in human_src:
        records.append(rec)

    human_total = len(records)
    injected_count = 0

    # Bot games — distinguish fast and strong by directory
    if include_bot_games:
        bot_dir = Path(bot_games_dir) if bot_games_dir is not None else None
        if bot_dir is not None:
            for depth_dir in ["sealbot_fast", "sealbot_strong"]:
                source_label = "bot_fast" if "fast" in depth_dir else "bot_strong"
                sub_dir = bot_dir / depth_dir
                if not sub_dir.exists():
                    continue
                bot_count = 0
                for game_file in sorted(sub_dir.glob("*.json")):
                    ok, rec = best_effort(
                        "data.corpus_analysis.bot_game_malformed_skipped",
                        lambda p=game_file, s=source_label: _bot_record(p, s),
                        counters=PIPELINE_COUNTERS,
                    )
                    if ok and rec is not None:
                        records.append(rec)
                        bot_count += 1
                log.info("loaded_bot_games", depth=depth_dir, count=bot_count)

        # Injected games (human-seed bot-continuation)
        inj_dir = Path(injected_dir) if injected_dir is not None else None
        if inj_dir is not None and inj_dir.exists():
            for game_file in sorted(inj_dir.glob("*.json")):
                ok, rec = best_effort(
                    "data.corpus_analysis.injected_game_malformed_skipped",
                    lambda p=game_file: _injected_record(p),
                    counters=PIPELINE_COUNTERS,
                )
                if ok and rec is not None:
                    records.append(rec)
                    injected_count += 1
            log.info("loaded_injected_games", count=injected_count)

    log.info("games_loaded", total=len(records), human=human_total,
             bot=len(records) - human_total - injected_count if include_bot_games else 0,
             injected=injected_count if include_bot_games else 0)
    log_pipeline_losses("data.corpus_analysis.load_all_games")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Corpus distribution analysis")
    # AUDIT-1 F-34: the corpus's own geometry. REQUIRED and undefaulted — this tool had no
    # encoding at all, so every Board it built took the engine defaults and its diversity and
    # cluster numbers described a board the corpus was not generated on.
    parser.add_argument("--encoding", type=str, required=True,
                        help="the registered encoding the corpus was generated under")
    parser.add_argument("--human-dir", type=str, required=True,
                        help="Directory of scraped human game JSON files")
    parser.add_argument("--bot-games-dir", type=str, default=None,
                        help="Root of bot self-play game JSON (with include-bot-games)")
    parser.add_argument("--injected-dir", type=str, default=None,
                        help="Directory of human-seed bot-continuation game JSON")
    parser.add_argument("--quality-scores-path", type=str, default=None,
                        help="Output path for the per-game quality-score sidecar JSON")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Output directory for summary JSON (default: reports/corpus_analysis)")
    parser.add_argument("--include-bot-games", action="store_true",
                        help="Include bot self-play games")
    parser.add_argument("--stratify-by-source", action="store_true",
                        help="Produce separate statistics for human / bot_fast / bot_strong")
    parser.add_argument("--compute-quality-scores", action="store_true",
                        help="Compute per-game quality scores and write sidecar file")
    parser.add_argument("--include-human-games", action="store_true",
                        help="Add Elo-stratified breakdown for human games")
    args = parser.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("corpus_analysis_start", include_bot_games=args.include_bot_games,
             stratify=args.stratify_by_source)

    records = load_all_games(
        args.human_dir,
        bot_games_dir=args.bot_games_dir,
        injected_dir=args.injected_dir,
        include_bot_games=args.include_bot_games,
    )
    if not records:
        log.error("no_games_found")
        return

    total_positions = sum(len(r.moves) for r in records)
    log.info("corpus_summary", total_games=len(records), total_positions=total_positions)

    console = _rich_console()
    if console is not None:
        console.rule("[bold]Corpus Distribution Analysis")

    # Always run combined analysis
    combined_results = run_analysis(
        records, "all", cluster_sample=500, encoding_name=args.encoding)
    _print_summary_table(combined_results, "Combined")

    # Print win rate by Elo band for combined
    wr = combined_results["win_rates"]
    if console is not None and any(
        wr["by_elo_band"].get(bl, {}).get("games", 0) > 0 for bl in ELO_LABELS
    ):
        from rich.table import Table  # type: ignore
        band_table = Table(title="P1 Win Rate by Elo Band (Combined)", show_header=True)
        band_table.add_column("Elo Band", style="bold")
        band_table.add_column("Games", justify="right")
        band_table.add_column("P1 Win Rate", justify="right")
        for bl in ELO_LABELS:
            bd = wr["by_elo_band"].get(bl, {"games": 0, "p1_win_rate": None})
            n = bd["games"]
            rate = bd["p1_win_rate"]
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            band_table.add_row(bl, str(n), rate_str)
        console.print(band_table)

    # Elo-stratified human game breakdown
    elo_stratified: dict = {}
    if args.include_human_games:
        if console is not None:
            console.rule("[bold]Elo-Stratified Human Games")
        elo_stratified = analyse_elo_stratified(records)
        _print_elo_stratified_table(elo_stratified)

    # Stratified analysis
    strata_results: dict[str, dict] = {}
    if args.stratify_by_source:
        strata = _stratify(records)
        if console is not None:
            console.rule("[bold]Stratified Analysis")
        for src, src_records in strata.items():
            if console is not None:
                console.rule(f"[bold cyan]{SOURCE_LABELS.get(src, src)}")
            result = run_analysis(
                src_records, src, cluster_sample=500, encoding_name=args.encoding)
            strata_results[src] = result
            _print_summary_table(result, src)

    # Quality scores
    quality_scores: dict[str, dict] = {}
    quality_stats: dict = {}
    if args.compute_quality_scores or args.stratify_by_source:
        if console is not None:
            console.rule("[bold]Quality Scores")
        per_game_entropy = _compute_per_game_entropies(records)
        quality_scores = compute_quality_scores(records, per_game_entropy)
        quality_stats = analyse_quality_distribution(quality_scores)

        # Write sidecar file (if a path was supplied)
        if args.quality_scores_path:
            scores_path = Path(args.quality_scores_path)
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, "w") as f:
                json.dump(quality_scores, f, indent=2)
            log.info("quality_scores_written", path=str(scores_path),
                     count=len(quality_scores))

    # Write summary JSON
    all_results = {
        "combined": combined_results,
        "strata": strata_results,
        "quality_stats": quality_stats,
        "elo_stratified": elo_stratified,
    }
    suffix = "stratified" if args.stratify_by_source else (
        "combined_summary" if args.include_bot_games else "summary")
    with open(report_dir / f"{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("results_written", path=str(report_dir / f"{suffix}.json"))

    log.info("corpus_analysis_complete", total_games=len(records),
             total_positions=total_positions)


if __name__ == "__main__":
    main()
