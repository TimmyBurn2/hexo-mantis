"""The Games panel in sentences: the result, the facts, and per position the threat line, what the bot thought, its candidates."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from ..readers.chances import Point
from ..readers.games import GameView
from ..readers.hexlogic import first_of_turn, owner, turn_of
from ..readers.tactics import Reading
from .fmt import ZONE, cell, esc, num, pct, short

NAME = ("Light", "Dark")
KIND = {"selfplay": "Self-play", "promotion": "Gate", "external": "External", "random_floor": "Random"}
_ARM = {"full": "Full search", "fast": "Fast search", "opening": "An opening stone"}
#: Candidates listed per searched stone.
CANDIDATES = 7


def cells(cs: list[tuple[int, int]] | tuple[tuple[int, int], ...], join: str) -> str:
    items = [cell(q, r) for q, r in cs]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {join} " + items[-1]


def headline(g: GameView) -> str:
    """The result in one sentence, from the record's own result and termination, with any finding beside it."""
    if g.result in ("p1", "p2") and g.termination in ("six_in_a_row", "win"):
        text = f"<strong>{NAME[0 if g.result == 'p1' else 1]} wins</strong> with six in a row on turn {g.turns}."
    elif g.result in ("p1", "p2"):
        text = f"<strong>{NAME[0 if g.result == 'p1' else 1]} wins</strong> by {esc(g.termination.replace('_', ' '))} on turn {g.turns}."
    elif g.termination == "ply_cap":
        text = f"<strong>No winner</strong>: the game hit the cap after {num(len(g.moves))} stones."
    elif g.result == "draw":
        text = f"<strong>Drawn</strong> by {esc(g.termination.replace('_', ' '))} after {num(len(g.moves))} stones."
    else:
        text = "<strong>The result is unknown</strong>: the engine could not decode the winner."
    return text + (f" <span class=\"finding\">Finding: {esc(g.finding)}.</span>" if g.finding else "")


def hour_of(shard_name: str) -> str | None:
    """The shard's UTC hour (`games_<run>_seg<n>_<YYYYMMDDHH>.jsonl`) in Central European time."""
    stem = shard_name.rsplit("_", 1)[-1].split(".")[0]
    if len(stem) != 10 or not stem.isdigit():
        return None
    start = datetime.strptime(stem, "%Y%m%d%H").replace(tzinfo=UTC).astimezone(ZONE)
    return f"{start.day} {start:%b}, the hour from {start:%H:%M} {start.tzname()}"


def facts(g: GameView, run_label: str, hour: str | None) -> list[tuple[str, str]]:
    """`(label, value)` rows for the game: kind and opponent, seats, the net that played, the search, when."""
    rows = [("Kind", KIND.get(g.channel, g.channel) + {"promotion": " against the anchor", "external": f" against {g.rung}",
                                                        "random_floor": " against a random player"}.get(g.channel, ""))]
    if g.candidate is not None:
        rows.append(("Seats", f"the candidate plays {NAME[0 if g.candidate == 1 else 1]}"))
    if g.step is None:
        net = "not recorded"
    elif g.step < 0:
        net = "before the actor's first sync"
    else:
        net = f"{run_label} at {short(g.step)}" + (", the actor's copy" if g.step_kind == "actor" else "")
    rows.append(("Net", net))
    search = {"recorded": "recorded at the searched stones", "absent": "not sampled for this game",
              "none": "no player exposed a search root", "empty": "recorded, but the candidate never moved"}[g.stats_field]
    rows.append(("Search", search + (f", {num(g.served_sims)} sims served" if g.served_sims else "")))
    if hour or g.worker is not None:
        rows.append(("Recorded", ", ".join(x for x in (hour, None if g.worker is None else f"worker {g.worker}") if x)))
    return rows


def threat(t: Reading) -> str:
    """The tactics line for the side to move, from the engine's tactics reader."""
    me, them = NAME[t.mover], NAME[1 - t.mover]
    if t.cls == "win":
        return f"<strong>{me} can win now</strong> at {cells(t.cells, 'or')}."
    fours = "a four" if t.fours == 1 else f"{t.fours} fours"
    if t.cls == "block":
        tail = ", with either stone" if t.k == 2 else ""
        return f"<strong>{them} has {fours}.</strong> {me} must block {cells(t.cells, 'or')} this turn{tail}."
    if t.cls == "lost1":
        return f"<strong>{them} has {fours}.</strong> {me} cannot block them all this turn."
    return "Nothing is forced this turn."


def thought(g: GameView, ply: int, cls_cells: set[tuple[int, int]], wins: bool) -> dict[str, Any]:
    """What the recorded search did at `ply`: a sentence, Light's chance, the candidates with tags; or why nothing is shown."""
    entry = g.stats.get(ply)
    if entry is None:
        why = {"absent": "This game was not sampled for search stats; tick With search to list the games that were.",
               "none": "No player exposed a search root in this game.",
               "empty": "The candidate never moved in this game."}.get(g.stats_field, "No search recorded at this stone.")
        if g.arms is not None and g.arms[ply] == "opening":
            why = "An opening stone: the runner placed it without a search."
        return {"text": why, "light": None, "cands": [], "second": False}
    visits = sorted(([int(v[0]), int(v[1]), int(v[2])] for v in entry.get("visits") or [] if len(v) == 3), key=lambda v: -v[2])
    total = sum(v[2] for v in visits)
    played = g.moves[ply]
    share = {(v[0], v[1]): v[2] / total for v in visits} if total else {}
    arm = _ARM.get(g.arms[ply], "Search") if g.arms is not None else "Search"
    by = f" by the {esc(entry['by'])}" if entry.get("by") else ""
    if not visits:
        text = f"{arm}{by}: the root's support was empty, so no visit is recorded."
    elif played == (visits[0][0], visits[0][1]):
        text = f"{arm}, {num(total)} visits{by}. It played {cell(*played)}, its most-visited move."
    elif played in share:
        text = f"{arm}, {num(total)} visits{by}. It played {cell(*played)}, with {pct(share[played])} of the visits."
    else:
        text = f"{arm}, {num(total)} visits{by}. It played {cell(*played)}, outside the recorded visits (—)."
    v = entry.get("root_value")
    light = None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        mine = (max(-1.0, min(1.0, float(v))) + 1) / 2
        light = round(mine if owner(ply) == 0 else 1 - mine, 4)
    tag = "wins" if wins else "blocks"
    cands = [[q, r, round(n / total, 4) if total else 0.0,
              [t for t, on in (("played", (q, r) == played), (tag, (q, r) in cls_cells)) if on]]
             for q, r, n in visits[:CANDIDATES]]
    return {"text": text, "light": light, "cands": cands, "second": not first_of_turn(ply)}


def where(g: GameView, ply: int) -> str:
    """The transport's line: which turn, who places which stone, or the final position."""
    if ply >= len(g.moves):
        return f"<b>Final position</b>, turn {g.turns}"
    stone = 1 if first_of_turn(ply) else 2
    return f"<b>Turn {turn_of(ply)}</b> of {g.turns}, {NAME[owner(ply)]} places stone {stone} of {1 if ply == 0 else 2}"


def turning(tp: Point | None, game_id: str) -> str:
    """The turn that cost its mover the most chance, as a sentence linking to it; empty when no turn cost enough."""
    if tp is None:
        return ""
    href = esc("?" + urlencode({"g": game_id, "ply": tp.ply}))
    return (f'The search saw the game turn on <a href="{href}" data-ply="{tp.ply}">turn {tp.turn}</a>: it cost '
            f"{NAME[owner(tp.ply)]} {round((tp.cost or 0.0) * 100)} points of win chance.")
