"""The teacher's validation against the engine itself: the argmax of our ONNX Runtime read v Six's own `go nodes 1` stone."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mantis.bots.six import SixEngine, locate_six, parse_reply
from mantis.bots.strix import find_vendor_root

from .corpus import SRC_RING
from .label import row_moves
from .planes import crop_cells, root_tactics_act, six_planes, stones_left_before
from .teacher import Teacher

#: Settings that leave only the network and the immediate-threat rule at Six's root.
ENGINE_SETTINGS = (("rootThreatNodes", 0), ("leafThreatNodes", 0))
#: |score/1000 + v(child)| within this reads as the same value (the score is rounded to thousandths).
VALUE_TOL = 0.002


def net_argmax(teacher: Teacher, moves: np.ndarray, radius: int) -> tuple[tuple[int, int], np.ndarray, float]:
    """Our read's argmax cell over Six's legal plane, the planes, and the value."""
    planes, center = six_planes(moves, radius)
    out = teacher.evaluate(planes[None])
    legal = planes[3].reshape(-1) > 0.5
    logits = np.where(legal, out.policy[0], -np.inf)
    best = int(np.argmax(logits))
    cq, cr = crop_cells(center)
    return (int(cq[best]), int(cr[best])), planes, float(out.value[0])


def engine_stones(engine: SixEngine, moves: np.ndarray, *, radius: int, nodes: int) -> tuple[tuple[tuple[int, int], ...], int | None]:
    """The engine's stones and `info` score for a fresh search of `moves`; Raises: RuntimeError — the engine's reply is a failure."""
    engine.new_game()
    lines = engine.search([(int(m[0]), int(m[1])) for m in moves], radius=radius, nodes=nodes)
    reply = parse_reply(lines)
    if reply.failure is not None:
        raise RuntimeError(f"sixengine failed: {reply.failure}")
    score = None
    for ln in lines:
        words = ln.split()
        if words[:1] == ["info"] and "score" in words:
            score = int(words[words.index("score") + 1])
    return reply.stones, score


def validate(corpus: dict[str, np.ndarray], teacher: Teacher, cpu_teacher: Teacher, *, n: int, seed: int,
             radius: int, log: Callable[[str], None] = print) -> dict[str, Any]:
    """V1 on the first `n` screened held-out ring rows in a seeded order, with the second-stone, gross and value rows beside; Raises: SixEngineError, RungUnresolvable — the engine does not start or answer; RuntimeError — a failed search."""
    rows = np.nonzero((corpus["source"] == SRC_RING) & corpus["heldout"])[0]
    order = rows[np.random.default_rng(seed).permutation(len(rows))]
    engine = SixEngine(locate_six(find_vendor_root(), teacher.variant), device="cpu")
    try:
        for name, value in ENGINE_SETTINGS:
            engine.set_option(name, value)
        res = _walk(order, corpus, teacher, cpu_teacher, engine, n=n, radius=radius, log=log)
    finally:
        engine.close()
    res.update(engine_settings=dict(ENGINE_SETTINGS), seed=seed, radius=radius, teacher=teacher.record(),
               engine_device="cpu", pass_line=int(np.ceil(0.99 * n)),
               passed=bool(res["agree"] >= int(np.ceil(0.99 * n)) and res["screened"] == n))
    return res


def _walk(order: np.ndarray, corpus: dict[str, np.ndarray], teacher: Teacher, cpu_teacher: Teacher,
          engine: SixEngine, *, n: int, radius: int, log: Callable[[str], None]) -> dict[str, Any]:
    agree = screened = skipped_tactics = skipped_empty = gross_agree = gross_n = 0
    second_n = second_agree = value_n = value_agree = 0
    dv_cpu = dp_cpu = 0.0
    disagreements: list[dict[str, Any]] = []
    value_err: list[float] = []
    for i in order:
        if screened >= n:
            break
        moves = row_moves(corpus, int(i))
        if len(moves) == 0:
            skipped_empty += 1
            continue
        tactics = root_tactics_act(moves)
        mine, planes, _v = net_argmax(teacher, moves, radius)
        stones, _score = engine_stones(engine, moves, radius=radius, nodes=1)
        if gross_n < n:
            gross_n += 1
            gross_agree += int(stones[0] == mine)
        if tactics:
            skipped_tactics += 1
            continue
        screened += 1
        hit = stones[0] == mine
        agree += int(hit)
        cpu_out, gpu_out = cpu_teacher.evaluate(planes[None]), teacher.evaluate(planes[None])
        dp_cpu = max(dp_cpu, float(np.abs(cpu_out.policy - gpu_out.policy).max()))
        dv_cpu = max(dv_cpu, float(np.abs(cpu_out.value - gpu_out.value).max()))
        if not hit and len(disagreements) < 20:
            disagreements.append({"row": int(i), "ply": len(moves), "engine": list(stones[0]), "ours": list(mine)})
        child = np.vstack([moves, np.asarray([stones[0]])])
        if hit and len(stones) == 2 and not root_tactics_act(child):
            second_n += 1
            second_agree += int(net_argmax(teacher, child, radius)[0] == stones[1])
        if hit and stones_left_before(len(moves)) == 1 and not root_tactics_act(child):
            _s2, score2 = engine_stones(engine, moves, radius=radius, nodes=2)
            if score2 is not None:
                v_child = net_argmax(teacher, child, radius)[2]
                value_n += 1
                err = abs(score2 / 1000.0 + v_child)
                value_err.append(err)
                value_agree += int(err <= VALUE_TOL)
        if screened % 100 == 0:
            log(f"V1 {agree}/{screened} agree, {skipped_tactics} tactics rows skipped")
    return {"screened": screened, "agree": agree, "skipped_tactics": skipped_tactics, "skipped_empty": skipped_empty,
            "gross_n": gross_n, "gross_agree": gross_agree, "second_n": second_n, "second_agree": second_agree,
            "value_n": value_n, "value_agree": value_agree, "value_tol": VALUE_TOL,
            "value_err_median": float(np.median(value_err)) if value_err else None,
            "value_err_max": float(np.max(value_err)) if value_err else None,
            "ort_cuda_v_cpu_max_dpolicy": dp_cpu, "ort_cuda_v_cpu_max_dvalue": dv_cpu,
            "disagreements": disagreements}
