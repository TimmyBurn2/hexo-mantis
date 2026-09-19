"""The page: the viewer's tokens verbatim, both static files inlined, every key in the header, the perspective stated once."""
from __future__ import annotations

import importlib
import re

import pytest


@pytest.fixture(scope="module")
def html(analyzer):
    return importlib.import_module("analyzer.html")


def test_the_page_inlines_its_css_and_js_and_carries_the_viewers_tokens(html):
    page = html.render()
    for token in ("--bg", "--fg", "--mute", "--line", "--p1", "--p2", "--hi", "--win", "--heat", "--pos", "--neg"):
        assert f"{token}:" in page, token
    assert "<script src=" not in page and '<link rel="stylesheet"' not in page
    assert "prefers-color-scheme:dark" in page.replace(" ", "")
    assert "fetch('/engines')" in page and "'/analyze'" in page and "'/trace'" in page
    assert "window.HexBoard=" in page and page.index("window.HexBoard=") < page.index("HexBoard.draw("), "board.js first"


def test_the_header_prints_every_key_and_the_perspective_once(html):
    page = html.render()
    for key in ("u", "h", "p", "q", "t", "s", "v", "l", "c", "?"):
        assert re.search(rf"<kbd>{re.escape(key)}</kbd>", page), key
    assert page.count("values for the side to move") == 1


def test_the_title_is_escaped(html):
    assert "&lt;b&gt;" in html.render("<b>")


def test_the_page_has_a_slot_control_and_a_card_template_with_a_marker_class(html):
    page = html.render()
    assert 'id="addslot"' in page and '<template id="cardtpl">' in page and 'class="cardengine"' in page
    assert ".mark{" in page.replace(" ", "") and ".tracesvg[hidden]{display:none}" in page.replace(" ", "")
