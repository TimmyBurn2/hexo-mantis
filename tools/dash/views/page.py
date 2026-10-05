"""The page shell: tokens, the top bar with the three views, the theme toggle; linked when served, inlined when frozen."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fmt import esc

WEB = Path(__file__).resolve().parent.parent / "web"
VIEWS = (("run", "Run"), ("games", "Games"), ("analyzer", "Analyzer"))
_MARK = ('<svg width="20" height="22" viewBox="0 0 20 22" aria-hidden="true"><path d="M10 1 19 6v10l-9 5-9-5V6z" '
         'fill="none" stroke="currentColor" stroke-width="2"/><circle cx="10" cy="11" r="3.2" fill="currentColor"/></svg>')
#: Runs before first paint, so a stored or preferred theme never flashes the other one.
_THEME = ("<script>(function(){var t=null;try{t=localStorage.getItem('mantis-theme')}catch(e){}"
          "if(t)document.documentElement.dataset.theme=t;})();</script>")


@dataclass(frozen=True)
class Shell:
    """What the top bar needs: the view, the runs and the chosen ones, and whether the page is a frozen file."""

    view: str
    runs: tuple[str, ...] = ()
    run: str | None = None
    compare: str | None = None
    frozen: bool = False


def _href(view: str, run: str | None) -> str:
    if view == "analyzer":
        return "/analyzer"
    return f"/run/{esc(run)}" + ("/games" if view == "games" else "") if run else "/"


def _bar(shell: Shell) -> str:
    links = []
    for key, label in VIEWS:
        if shell.frozen and key != "run":
            continue
        current = ' aria-current="page"' if key == shell.view else ""
        links.append(f'<a href="{"#" if shell.frozen else _href(key, shell.run)}"{current}>{label}</a>')
    controls = []
    if shell.view == "run" and not shell.frozen and shell.run:
        picks = "".join(f'<option value="{esc(r)}"{" selected" if r == shell.run else ""}>{esc(r)}</option>'
                        for r in shell.runs)
        others = "".join(f'<option value="{esc(r)}"{" selected" if r == shell.compare else ""}>{esc(r)}</option>'
                         for r in shell.runs if r != shell.run)
        controls.append(f'<form class="pick" method="get" action="/run"><label>Run <select name="run" class="btn">{picks}'
                        f'</select></label><label>Compare <select name="compare" class="btn"><option value="">none'
                        f'</option>{others}</select></label><noscript><button class="btn">Show</button></noscript></form>')
    elif shell.run and shell.view == "games":
        controls.append(f'<span class="muted">{esc(shell.run)}</span>')
    controls.append('<button class="btn" type="button" id="theme" aria-label="Switch colour theme">Theme</button>')
    return (f'<header class="top"><div class="bar"><div class="mark">{_MARK}mantis</div>'
            f'<nav class="views" aria-label="Views">{"".join(links)}</nav>'
            f'<div class="controls">{"".join(controls)}</div></div></header>')


def render(title: str, shell: Shell, body: str, *, scripts: tuple[str, ...] = (), app: bool = False) -> str:
    """A whole page; served pages link `/static/` files, a frozen page carries the stylesheet and scripts inline."""
    if shell.frozen:
        style = f"<style>{(WEB / 'dash.css').read_text(encoding='utf-8')}</style>"
        code = "".join(f"<script>{(WEB / s).read_text(encoding='utf-8')}</script>" for s in scripts)
    else:
        style = '<link rel="stylesheet" href="/static/dash.css">'
        code = "".join(f'<script src="/static/{s}" defer></script>' for s in scripts)
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{esc(title)}</title>{_THEME}{style}</head>"
            f'<body class="{"app" if app else "doc"}">{_bar(shell)}{body}{code}</body></html>')
