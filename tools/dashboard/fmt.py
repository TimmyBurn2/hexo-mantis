"""Number and text formatting shared by every panel."""
from __future__ import annotations

import html
import math
from typing import Any

#: A narrow no-break space, used as the thousands separator.
THIN = " "
MINUS = "−"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def num(value: float | int | None, digits: int = 3) -> str:
    """A number for reading: thousands grouped, `digits` significant figures below 1 000."""
    if value is None:
        return "—"
    f = float(value)
    if not math.isfinite(f):
        return "−∞" if f < 0 else "∞" if f > 0 else "NaN"
    if abs(f) >= 1000:
        text = f"{f:,.0f}".replace(",", THIN)
    elif f == int(f):
        text = str(int(f))
    else:
        text = f"{f:.{digits}g}"
    return text.replace("-", MINUS)


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f} %"


def hours(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} h"


def signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace("-", MINUS)
