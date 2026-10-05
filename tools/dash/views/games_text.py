"""The Games panel in sentences: the result, the facts, and per position the threat line, what the bot thought, its candidates."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from ..readers.chances import Point
from ..readers.games import GameView
from ..readers.hexlogic import first_of_turn, owner, turn_of, turn_size
from ..readers.tactics import Reading
from .fmt import ZONE, cell, esc, num, pct, short

NAME = ("Light", "Dark")
KIND = {"selfplay": "Self-play", "promotion": "Gate", "external": "External", "random_floor": "Random"}
_ARM = {"full": "Full search", "fast": "Fast search", "opening": "Opening stone"}
#: Candidates listed per searched stone.
CANDIDATES = 7


def cells(cs: list[tuple[int, int]] | tuple[tuple[int, int], ...], join: str) -> str:
    """Cells as a phrase: `(1, 2)`, `(1, 2) or (3, 4)`, `(1, 2), (3, 4) or (5, 6)`."""
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
    if len(stem) != 10 or not (stem.isascii() and stem.isdigit()):
        return None
    start = datetime.strptime(stem, "%Y%m%d%H").replace(tzinfo=UTC).astimezone(ZONE)
    return f"{start.day} {start:%b}, {start:%H}:00–{(start.hour + 1) % 24:02d}:00 {start.tzname()}"


def facts(g: GameView, run_label: str, hour: str | None) -> list[tuple[str, str]]:
    """`(label, value)` rows for the game: kind and opponent, seats, the net that played, the search, when."""
    rows = [("Kind", KIND.get(g.channel, g.channel) + {"promotion": " against the anchor", "external": f" against {g.rung}",
                                                        "random_floor": " against a random player"}.get(g.channel, ""))]
    if g.candidate is not None:
        rows.append(("Seats", f"candidate plays {NAME[0 if g.candidate == 1 else 1]}"))
    if g.step is None:
        net = "not recorded"
    elif g.step < 0:
        net = "before the first sync"
    else:
        net = f"{run_label} at {short(g.step)}" + (" (self-play copy)" if g.step_kind == "actor" else "")
    rows.append(("Net", net))
    search = {"recorded": "recorded", "absent": "not sampled", "none": "no search root exposed",
              "empty": "recorded, candidate never moved"}[g.stats_field]
    rows.append(("Search", search + (f", up to {num(g.served_sims)} sims a stone" if g.served_sims else "")))
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
        why = {"absent": "Not sampled for search stats. Tick With search to list the games that were.",
               "none": "No search root exposed in this game.",
               "empty": "The candidate never moved in this game."}.get(g.stats_field, "No search recorded at this stone.")
        if g.arms is not None and g.arms[ply] == "opening":
            why = "Opening stone, placed without a search."
        return {"text": why, "light": None, "cands": [], "second": False}
    visits = sorted(([int(v[0]), int(v[1]), int(v[2])] for v in entry.get("visits") or [] if len(v) == 3), key=lambda v: -v[2])
    total = sum(v[2] for v in visits)
    played = g.moves[ply]
    share = {(v[0], v[1]): v[2] / total for v in visits} if total else {}
    arm = _ARM.get(g.arms[ply], "Search") if g.arms is not None else "Search"
    by = f" by the {esc(entry['by'])}" if entry.get("by") else ""
    how = f"{arm}, {num(total)} sims{by}."
    if not visits:
        pick, how = f"{arm}{by}: no visits recorded", ""
    elif played == (visits[0][0], visits[0][1]):
        pick = f"{cell(*played)}, the top move"
    elif played in share:
        pick = f"{cell(*played)}, {pct(share[played])} of the visits"
    else:
        pick = f"{cell(*played)}, not among the recorded visits"
    text = f"{pick}, {num(total)} sims ({arm.lower()})." if how else f"{pick}."
    pick += "."
    v = entry.get("root_value")
    light = None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        mine = (max(-1.0, min(1.0, float(v))) + 1) / 2
        light = round(mine if owner(ply) == 0 else 1 - mine, 4)
    tag = "wins" if wins else "blocks"
    cands = [[q, r, round(n / total, 4) if total else 0.0,
              [t for t, on in (("played", (q, r) == played), (tag, (q, r) in cls_cells)) if on]]
             for q, r, n in visits[:CANDIDATES]]
    return {"text": text, "pick": pick, "how": how, "light": light, "cands": cands, "second": not first_of_turn(ply)}


def where(g: GameView, ply: int) -> str:
    """The transport's line: at a turn's start whose turn and how many stones, mid-turn which stone, or the final position."""
    if ply >= len(g.moves):
        return f"<b>Final position</b>, turn {g.turns}"
    if first_of_turn(ply):
        return (f"<b>Turn {turn_of(ply)}</b> of {g.turns}, {NAME[owner(ply)]} to place "
                f"{'one stone' if turn_size(ply) == 1 else 'two stones'}")
    return f"<b>Turn {turn_of(ply)}</b> of {g.turns}, {NAME[owner(ply)]} places stone 2 of 2"


def turn_thought(g: GameView, start: int, cls_cells: set[tuple[int, int]], wins: bool) -> dict[str, Any]:
    """The turn the bot played from `start`: each stone's search in a sentence, the first stone's candidates and chance."""
    plies = [p for p in range(start, start + turn_size(start)) if p < len(g.moves)]
    first = thought(g, start, cls_cells, wins)
    second = thought(g, start + 1, set(), False) if len(plies) == 2 else None
    # Two stones searched alike say the search once, under both picks.
    shared = second is not None and first.get("how") and first.get("how") == second.get("how")
    texts = [["Stone" if len(plies) == 1 else "Stone 1", first["pick"] if shared else first["text"]]]
    if second is not None:
        texts.append(["Stone 2", second["pick"] if shared else second["text"]])
    return {"texts": texts, "how": first["how"].replace(" sims", " sims each") if shared else "", "light": first["light"],
            "cands": first["cands"], "second": first["second"], "stones": [list(g.moves[p]) for p in plies]}


def turning(tp: Point | None, game_id: str) -> str:
    """The turn that cost its mover the most chance, as a sentence linking to it; empty when no turn cost enough."""
    if tp is None:
        return ""
    href = esc("?" + urlencode({"g": game_id, "ply": tp.ply}))
    return (f'The search saw the game turn on <a href="{href}" data-ply="{tp.ply}">turn {tp.turn}</a>: it cost '
            f"{NAME[owner(tp.ply)]} {round((tp.cost or 0.0) * 100)} points of win chance.")
