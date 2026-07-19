# pyright: basic
"""Rich console reporters for corpus distribution analysis.

Pure presentation: rich Table rendering of stratum summaries and the Elo-band
breakdown. No metric computation, no file I/O beyond console.print.

``rich`` is an optional dependency (not a mantis runtime requirement) and is
imported lazily inside each helper: when it is absent every ``_print_*`` helper
is a no-op, so importing this module never fails. pyright is set to `basic`
(rich is untyped in the bare env).
"""

from __future__ import annotations

from mantis.data.corpus_metrics import (
    MANIFEST_BAND_ORDER,
    SOURCE_LABELS,
)


def _print_elo_stratified_table(elo_data: dict) -> None:
    """Print Elo-stratified breakdown as a rich table."""
    try:
        from rich.console import Console  # type: ignore
        from rich.table import Table  # type: ignore
    except ImportError:  # pragma: no cover - rich is an optional dependency
        return

    table = Table(title="Human Games — Elo-Stratified Breakdown", show_header=True)
    table.add_column("Elo Band", style="bold")
    table.add_column("Games", justify="right")
    table.add_column("Median Length", justify="right")
    table.add_column("Top Opening (count)", justify="left")

    for band in MANIFEST_BAND_ORDER:
        data = elo_data.get(band, {})
        count = data.get("game_count", 0)
        median = data.get("median_compound_moves", 0)
        top = data.get("top_openings", [])
        top_str = ""
        if top:
            first = top[0]
            moves_str = " ".join(f"({q},{r})" for q, r in first["moves"][:5])
            top_str = f"{moves_str}... ({first['count']})"
        table.add_row(band, str(count), str(median), top_str)

    Console().print(table)


def _print_summary_table(results: dict, label: str) -> None:
    """Print a rich summary table for one stratum."""
    try:
        from rich.console import Console  # type: ignore
        from rich.table import Table  # type: ignore
    except ImportError:  # pragma: no cover - rich is an optional dependency
        return

    table = Table(title=f"Corpus Analysis — {SOURCE_LABELS.get(label, label)}",
                  show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Flag", justify="center")

    gl = results["game_lengths"]
    wr = results["win_rates"]
    me = results["move_entropy"]
    od = results["opening_diversity"]
    cc = results["cluster_counts"]

    table.add_row("Games", str(results["game_count"]), "")
    table.add_row("Positions", str(results["total_positions"]), "")
    table.add_row("Median length", str(gl["median"]), "")
    table.add_row("P10 / P90 length", f"{gl['p10_threshold']} / {gl.get('p90', 'N/A')}", "")

    p1_flag = "[red]> 60%[/red]" if wr["p1_advantage_flag"] else "[green]OK[/green]"
    table.add_row("P1 win rate", f"{wr['overall_p1_win_rate']:.1%}", p1_flag)

    table.add_row("Mean entropy",
                  f"{me['mean_entropy_nats']:.2f} ± {me['std_entropy_nats']:.2f} nats", "")

    dupe = od.get("dupe_rate_first_10", 0)
    dupe_flag = "[red]HIGH[/red]" if dupe > 0.50 else "[green]OK[/green]"
    table.add_row("Dupe rate (first 10)", f"{dupe:.1%}", dupe_flag)
    table.add_row("First-move entropy", f"{od.get('first_move_entropy', 0):.2f} nats", "")
    table.add_row("Unique @ move 3/5/10/20",
                  f"{od.get('unique_at_move_3', 0)} / {od.get('unique_at_move_5', 0)} / "
                  f"{od.get('unique_at_move_10', 0)} / {od.get('unique_at_move_20', 0)}", "")

    table.add_row("Median K (clusters)", str(cc["median_cluster_count"]), "")
    table.add_row("Frac K > 2", f"{cc['frac_k_gt2']:.1%}", "")
    table.add_row("Max K", str(cc["max_cluster_count"]), "")

    pc = results.get("ply_coverage", {})
    if pc:
        late_frac = pc.get("late_game_fraction", 0.0)
        late_flag = pc.get("late_game_flag", False)
        late_str = f"{late_frac:.1%}"
        pc_flag = "[red]< 10% (underrepresented)[/red]" if late_flag else "[green]OK[/green]"
        table.add_row("Positions at ply ≥ 40", late_str, pc_flag)

    Console().print(table)
