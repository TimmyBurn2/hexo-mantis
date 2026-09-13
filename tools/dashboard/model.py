"""Shared vocabulary: a panel, a hero cell, and the gap register every tier writes to."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fmt import esc


@dataclass(frozen=True)
class Panel:
    """One rendered block: its title, the events it reads, its body, an optional note."""

    title: str
    reads: str
    body: str
    slug: str
    note: str = ""


@dataclass(frozen=True)
class HeroCell:
    """One tier-1 number: label, the value, a one-line detail, and its state colour."""

    label: str
    value: str
    detail: str
    state: str = "neutral"


@dataclass
class Gaps:
    """Every absence the page states; tier 1/2 carry a short marker, tier 3 the full note."""

    items: list[tuple[str, str]] = field(default_factory=list)

    def mark(self, panel: str, statement: str) -> str:
        """Register `statement` under `panel` and return the short marker for tiers 1 and 2."""
        self.items.append((panel, statement))
        return '<p class="gap">not measured — see notes</p>'


GAP_MARKER = "not measured — see notes"


def no_rows(event_name: str) -> str:
    """The ONE way an empty series is drawn: named, never as a zero or a flat line."""
    return (f'<p class="absent">No rows in this record. This panel is drawn from '
            f'<code>{esc(event_name)}</code>; the record carries none, which is an absence '
            'and is not a zero.</p>')


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")
