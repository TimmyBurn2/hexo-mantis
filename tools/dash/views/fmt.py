"""Formatting for the pages: every record string escaped, numbers with thin-space thousands, times in Central European time."""
from __future__ import annotations

import html
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

#: The operator's zone: every wall time on a page is CET or CEST, labelled.
ZONE = ZoneInfo("Europe/Berlin")
MINUS = "−"
NBSP = "\u202f"


def esc(value: object) -> str:
    """Any value as HTML text, quotes included: nothing from a record reaches a page unescaped."""
    return html.escape(str(value), quote=True)


def script_json(value: object) -> str:
    """JSON for an inline `<script>`: `<`, `>` and `&` escaped, so no record string can close the element or open a comment."""
    text = json.dumps(value, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def num(value: float | int | None, digits: int = 0) -> str:
    """A number with a narrow no-break space between thousands and a real minus sign; None is an em dash."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    text = f"{abs(value):,.{digits}f}".replace(",", NBSP)
    return (MINUS if value < 0 and float(text.replace(NBSP, "")) != 0 else "") + text


def pct(value: float | None, digits: int = 0) -> str:
    """A share as a percentage with a narrow space before the sign; None is an em dash."""
    return "—" if value is None else f"{num(100.0 * value, digits)}{NBSP}%"


def short(step: float | int | None) -> str:
    """A step count the way the axes read it: 45k, 50.8k, 900."""
    if step is None:
        return "—"
    if abs(step) < 1000:
        return num(step)
    text = f"{step / 1000:.1f}".rstrip("0").rstrip(".")
    return text.replace("-", MINUS) + "k"


def sig(value: float | None) -> str:
    """A loss-like value at three significant places: 0.517, 2.17, 20.4, 1 234."""
    if value is None or not math.isfinite(value):
        return "—"
    mag = abs(value)
    return num(value, 0 if mag >= 100 else 1 if mag >= 10 else 2 if mag >= 1 else 3)


def signed(value: float, digits: int = 2) -> str:
    """A difference with its sign always shown: +0.17, −0.04; one that rounds to zero carries none."""
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return ("+" if value >= 0 else MINUS) + text


def cell(q: int, r: int) -> str:
    """A board cell as the pages name it: (−6, −1)."""
    return f"({num(q)}, {num(r)})"


def when(ts: float | None) -> str:
    """A wall time as `21 Sept, 10:23 CEST`; None is an em dash."""
    if ts is None:
        return "—"
    moment = datetime.fromtimestamp(ts, ZONE)
    month = ("Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec")[moment.month - 1]
    return f"{moment.day} {month}, {moment:%H:%M} {moment.tzname()}"
