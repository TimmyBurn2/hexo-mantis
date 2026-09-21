"""Reading 4 (P-B2's falsifier): the tactics solver over mirrored ring roots — proof rate, novelty, ms per root."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from mantis._engine import TacticalSolver
from mantis.diagnostics.ring_reader import Ring

from .rings import full_arm_rows, reconstructed, target_argmax

WIN, LOSS, UNKNOWN = 1, -1, 0


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def argmax_keeps_win(board: Any, argmax: tuple[int, int], solver: TacticalSolver, depth: int, node_budget: int) -> bool:
    """Whether the TARGET's argmax keeps a proven win provable: the six itself, a WIN with the mover still to move, or the opponent's LOSS."""
    after = board.clone()
    after.apply_move(int(argmax[0]), int(argmax[1]))
    if after.check_win():
        return True
    result, _line, _nodes = solver.prove(after, depth, node_budget)
    return result == (WIN if after.current_player == board.current_player else LOSS)


def proof_rate(ring: Ring, *, rows: int, seed: int, depth: int, node_budget: int,
               cand_cap: int = 40) -> dict[str, Any]:
    """Over `rows` rebuilt full-arm roots: WIN/LOSS/UNKNOWN shares, novelty (plain and strict), z agreement, nodes and ms per root; net-free, no window."""
    solver = TacticalSolver(window_half=None, cand_cap=cand_cap)
    picked = full_arm_rows(ring, seed=seed, n=rows)
    boards = reconstructed(ring, picked)
    results, ms, nodes, novel, strict, wins_z, loss_z = [], [], [], 0, 0, [], []
    for i, board, _moves in boards:
        t0 = time.perf_counter()
        result, line, spent = solver.prove(board, depth, node_budget)
        ms.append((time.perf_counter() - t0) * 1000.0)
        nodes.append(int(spent))
        results.append(int(result))
        if result == WIN:
            argmax = target_argmax(ring, i)
            first = (int(line[0][0]), int(line[0][1])) if line else None
            if first is not None and argmax is not None and first != argmax:
                novel += 1
                # A won root can hold several winning first stones: strict novelty is the argmax NOT keeping the win.
                strict += int(not argmax_keeps_win(board, argmax, solver, depth, node_budget))
            if ring.value_valid[i]:
                wins_z.append(float(ring.outcome[i]))
        elif result == LOSS and ring.value_valid[i]:
            loss_z.append(float(ring.outcome[i]))
    res = np.asarray(results)
    n, n_win, n_loss = int(res.size), int((res == WIN).sum()), int((res == LOSS).sum())
    return {
        "rows_drawn": int(picked.size), "rows_rebuilt": n, "depth_plies": depth, "node_budget": node_budget,
        "cand_cap": cand_cap, "window_half": None,
        "proof_rate": _rate(n_win, n), "loss_rate": _rate(n_loss, n), "unknown_rate": _rate(n - n_win - n_loss, n),
        "n_win": n_win, "n_loss": n_loss,
        "novelty_among_wins": _rate(novel, n_win), "n_novel": novel,
        "strict_novelty_among_wins": _rate(strict, n_win), "n_strict_novel": strict,
        "strict_novel_share_of_roots": _rate(strict, n),
        # z is the mover's realised outcome: a proven WIN whose game the mover lost is the solver contradicting the record.
        "win_z_mean": (float(np.mean(wins_z)) if wins_z else None), "win_z_agree": _rate(sum(z > 0 for z in wins_z), len(wins_z)),
        "loss_z_mean": (float(np.mean(loss_z)) if loss_z else None), "loss_z_agree": _rate(sum(z < 0 for z in loss_z), len(loss_z)),
        "ms_per_root_mean": (float(np.mean(ms)) if ms else None), "ms_per_root_p50": (float(np.median(ms)) if ms else None),
        "ms_per_root_p90": (float(np.quantile(ms, 0.9)) if ms else None),
        "nodes_mean": (float(np.mean(nodes)) if nodes else None), "budget_hit_share": _rate(sum(x >= node_budget for x in nodes), n),
        "by_moves_remaining": {
            str(k): {"n": int(sum(1 for (i, _b, _m), r in zip(boards, results, strict=True) if ring.moves_remaining[i] == k)),
                     "proof_rate": _rate(sum(1 for (i, _b, _m), r in zip(boards, results, strict=True)
                                             if ring.moves_remaining[i] == k and r == WIN),
                                         sum(1 for (i, _b, _m) in boards if ring.moves_remaining[i] == k))}
            for k in (1, 2)},
    }


__all__ = ["proof_rate"]
