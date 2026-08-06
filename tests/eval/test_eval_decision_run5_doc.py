# >300 justify (R8).
# O-A16/O-A17/O-A19 are ONE claim — `docs/contracts/eval_decision_run5.md` says only true
# things and says the not-yet-true ones in the ONE form R169 and R151 permit — over one
# artefact. They share the section parser and the doc loader; a split forks those into
# copies that drift while both stay green, and it would separate the drift gate (O-A16)
# from the two honesty clauses it exists to keep honest. Executable content is a minority;
# the rest is the per-row "what defect is this the only witness to" rationale LAW-07 asks
# each row to carry.
"""⊕ WP12-R Phase A / O-A16, O-A17, O-A19 (DESIGN_A §3, PREREG_A §1) — the decision doc.

`docs/contracts/eval_decision_run5.md` is a MINT INPUT (R147 consequence 3) and the artefact
that carries R151's control-arm honesty clause. It lives in the repo, not the workspace,
precisely so it can have a producer test — a workspace file cannot (DESIGN_A §3.1).

The defect each group is the ONLY witness to:

- **O-A16 (6 rows)** — a decision that states a number the config does not. The comparison
  direction is DOC-AGAINST-CONFIG, deliberately: a re-mint then REDS the doc rather than
  silently agreeing with it, which is the correct direction for a drift gate (gate 13's own
  discipline). Every expected value is DERIVED — from `configs/run5.yaml`, from the book
  manifest, from the live `_R139_SKIP_GROUNDS` mapping — and never transcribed into this
  file, because a transcribed list makes the oracle agree with the doc for the same reason
  the doc is wrong. MUTATION (M-A12): state `promotion_winrate` as 0.60.
- **O-A17 (2 rows)** — R151's clause softening. Arm (a) is WITHDRAWN as a new test: the
  refusal itself is already executed against the REAL `lookup("v6_live2_ls")` by four
  producers (DESIGN_A §3.8 P-1..P-4), one of them FROZEN, and a fifth would be a second
  authority over a frozen oracle (R79). What remains is doc-side and nothing existing covers
  it. MUTATION (M-A13): delete `not_run`, or add a win-rate number.
  **O-A16 cannot see M-A13's second half** — a fabricated dense win rate has no config
  counterpart to compare against — and that coverage boundary is why arm (c) exists apart.
- **O-A19 (5 rows)** — R96's in-place upgrade decaying into two states at once, and R169's
  liveness claim decaying into a bare one. Arm 3 asserts the DISCRIMINATOR: only the
  liveness clause carries a dated `*Status:*` line, and without an assertion a later edit
  could quietly give the R151 clause one and collapse the distinction. Arm 4 is abort 11's
  instrument. MUTATION (M-A19): append a STATE 2 sentence below STATE 1 instead of replacing
  it; (M-A19b): a bare liveness claim elsewhere in the doc.

**W-6, stated at the instrument rather than only in the prereg:** arm 4 is a TOKEN SCAN and
is defeatable by paraphrase — *"both sealbot rungs are exercised against the real engine"*
asserts liveness and matches none of its tokens. It is a lower bound on detection, not a
proof of absence. The residual is reviewer-enforced.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DOC = _REPO / "docs" / "contracts" / "eval_decision_run5.md"
_RUN5 = _REPO / "configs" / "run5.yaml"
_BOOKS = _REPO / "src" / "mantis" / "arena" / "books" / "manifest.toml"

#: DESIGN_A §3.4 rule 1: the two clauses never share a paragraph, a table or a bullet list.
#: These are the section headings the rule names, and they are the anchors every row below
#: locates by — never a line number, which would retire the oracle within a phase.
_CONTROL_ARM_HEADING = "Control arm: wiring owed (R151)"

#: DESIGN_A §3.6's FOUR state templates, keyed by their heading lead and carrying the ONE
#: state marker each is allowed to leave behind.
_STATE_HEADINGS = {
    "Ladder liveness: unverified in CI (R169)": "not_run",
    "Ladder liveness: verified at box preflight (R169)": "covered",
    "Ladder liveness: unverified — box preflight did not complete (R169)": "not_run",
    "Ladder liveness: MEASURED AND FAILED at box preflight (R169)": "FAILED",
}
_STATE_MARKERS = ("not_run", "covered", "FAILED")

#: PREREG_A §8 abort 11: the sanctioned wording and nothing else.
_R169_SANCTIONED = "2/6 resolve locally"

#: O-A19 arm 4 / abort 11's instrument.
_LIVENESS_TOKENS = ("live", "liveness", "is running", "plays")
_RUNG_TOKENS = ("sealbot_d5", "sealbot_d6", "rung")

_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_STATUS_RE = re.compile(r"\*Status:")


def _doc() -> str:
    assert _DOC.is_file(), (
        f"{_DOC.relative_to(_REPO)} does not exist. DESIGN_A §3.1 rules the decision into "
        f"the REPO rather than the migration workspace, because a mint-blocking decision "
        f"reachable only from a sibling directory is a provenance hole of the LAW-12 shape "
        f"— and because an in-repo doc can have a producer test, which is this file."
    )
    return _DOC.read_text()


def _run5() -> dict:
    import yaml

    return yaml.safe_load(_RUN5.read_text())


def _sections(doc: str) -> dict[str, str]:
    """Markdown heading -> body. The doc is addressed by stable heading anchors."""
    out: dict[str, str] = {}
    current = ""
    body: list[str] = []
    for line in doc.splitlines():
        if line.startswith("#"):
            out[current] = "\n".join(body)
            current = line.lstrip("#").strip()
            body = []
        else:
            body.append(line)
    out[current] = "\n".join(body)
    return out


def _section_named(doc: str, needle: str) -> str:
    matches = [body for title, body in _sections(doc).items() if needle in title]
    assert len(matches) == 1, (
        f"expected exactly ONE section whose heading contains {needle!r}; found "
        f"{len(matches)}. DESIGN_A §3.4 rule 1: separate, NAMED sections — a merged "
        f"'known gaps' table is the forbidden formulation (PREREG_A §8 abort 12)."
    )
    return matches[0]


def _lines_disagreeing(doc: str, key: str, value: object) -> list[str]:
    """Every line that names `key` AND carries a number must carry `value`.

    Compared NUMERICALLY, not by string rendering, and the distinction is not cosmetic: at
    ORACLE-WRITE the string form of this comparator reported a document stating `0.60` as
    disagreeing with the config value `0.60`, because `str(0.60)` is `"0.6"`. A drift gate
    that fires on formatting is a gate the next reader turns off.

    A line naming the key with no number at all is prose and is not a disagreement; a line
    naming the key with a DIFFERENT number is the drift M-A12 injects. The residual, stated:
    a line that happens to carry the right number for an unrelated reason is a false
    negative — this is a lower bound on detection, like O-A19 arm 4 (W-6).
    """
    mentions = [line for line in doc.splitlines() if key in line]
    assert mentions, f"the decision never mentions {key!r}; a bar it does not state is not a bar"
    target = float(value)  # type: ignore[arg-type]
    disagreeing: list[str] = []
    for line in mentions:
        numbers = [float(token) for token in _NUMBER_RE.findall(line)]
        if numbers and not any(number == target for number in numbers):
            disagreeing.append(line)
    return disagreeing


# ── O-A16 — the drift gate, six rows, every expectation derived ─────────────────────────
def test_decision_names_exactly_run5s_minted_rung_set() -> None:
    """O-A16. Two-sided: every minted rung must appear, and no rung-shaped token that is not
    minted may. A decision naming a rung run5 does not carry is a bar nobody plays."""
    doc = _doc()
    minted = {rung["name"] for rung in _run5()["eval"]["ladder"]["rungs"]}
    missing = sorted(name for name in minted if name not in doc)
    assert missing == [], f"the decision omits minted rungs: {missing}"

    claimed = set(re.findall(r"\b(?:sealbot|kraken|strix)_[A-Za-z0-9]+\b", doc))
    assert claimed <= minted, f"the decision names rungs run5 does not mint: {sorted(claimed - minted)}"


@pytest.mark.parametrize(
    "knob",
    ["stride", "screen_games", "confirm_games", "promotion_winrate", "screen_confirm_lo",
     "deploy_sims", "bootstrap_resamples", "min_distinct_per_pair", "seed_base"],
)
def test_decision_gate_knob_matches_the_minted_config(knob: str) -> None:
    """O-A16, the LAW-15 bar itself (S-1). Parametrized per knob so a single drifted number
    is attributable rather than reported as 'the gate row disagrees'."""
    doc = _doc()
    value = _run5()["eval"]["gate"][knob]
    disagreeing = _lines_disagreeing(doc, knob, value)
    assert disagreeing == [], (
        f"the decision states {knob} disagreeing with configs/run5.yaml ({value!r}):\n"
        + "\n".join(disagreeing)
    )


def test_decision_book_id_and_sha_match_the_pinned_manifest() -> None:
    """O-A16 (S-5). The book is versioned, sha-pinned and verified at load
    (`books.py:51-55`); a decision quoting a stale sha describes openings nobody played."""
    doc = _doc()
    books = tomllib.loads(_BOOKS.read_text())["books"]
    book_id = _run5()["eval"]["gate"]["opening_book"]
    assert book_id in doc, f"the decision does not name the gate's book {book_id!r}"
    assert books[book_id]["sha256"] in doc, (
        f"the decision does not carry {book_id}'s pinned sha256 — the property that makes "
        f"the openings reproducible rather than merely named"
    )


@pytest.mark.parametrize("key", ["random_model_sims", "sealbot_model_sims", "random_floor_games"])
def test_decision_per_side_compute_matches_the_minted_config(key: str) -> None:
    """O-A16 (S-12/S-14). `random_floor_games` is included on purpose: at run5 as minted it
    is 0, so the floor plays ZERO games, and a decision that presented the floor as part of
    the live bar without stating that would be exactly the over-read S-12 exists to stop."""
    doc = _doc()
    disagreeing = _lines_disagreeing(doc, key, _run5()["eval"][key])
    assert disagreeing == [], "\n".join(disagreeing)


def test_decision_quotes_r139s_grounds_from_the_live_mapping() -> None:
    """O-A16, two-sided with O-A2 (M-A2's second observer). The expected strings are read
    from the SHIPPED mapping, so the drift gate and the resolver oracle cannot disagree
    silently — a paraphrase in either place reds both."""
    from mantis.bots.resolve import _R139_SKIP_GROUNDS

    doc = _doc()
    missing = sorted(g for g in _R139_SKIP_GROUNDS.values() if g not in doc)
    assert missing == [], (
        f"the decision does not carry R139's grounds verbatim: {missing}. R143 calls these "
        f"skips OPERATOR-AUTHORIZED; the grounds are the words that say so."
    )


def test_the_drift_comparator_rejects_a_disagreeing_document() -> None:
    """O-A16's detector self-test. Without it the five rows above are satisfied by a
    comparator that finds nothing to compare (R81/R86), and gate 13's own lesson is that a
    derived check with no self-test is a check nobody has run."""
    synthetic = "The gate promotes at promotion_winrate 0.60 after 80 screen games.\n"
    assert _lines_disagreeing(synthetic, "promotion_winrate", 0.55) != [], (
        "the comparator accepted a document stating 0.60 where the config says 0.55"
    )
    assert _lines_disagreeing(synthetic, "promotion_winrate", 0.60) == [], (
        "the comparator rejected a document that AGREES — a gate that always fires is noise"
    )


# ── O-A17 — R151's honesty clause, doc-side ─────────────────────────────────────────────
def test_control_arm_row_is_marked_not_run_and_carries_no_win_rate() -> None:
    """O-A17 arms (b)+(c). `not_run` and `covered` are the difference between 'we did not
    look' and 'we looked' — R151's clause is that NO dense eval result exists, so a number
    in this section is a fabrication with nothing to compare it against."""
    section = _section_named(_doc(), _CONTROL_ARM_HEADING)
    assert "not_run" in section, (
        "the control-arm row must carry the coverage word `not_run` explicitly. A blank "
        "result cell is read as zero or as pending, and it is neither (DESIGN_A §3.3)."
    )
    assert "covered" not in section, "the control arm is not covered; it is refused by design"
    win_rates = [m for m in _NUMBER_RE.findall(section) if "." in m and 0.0 <= float(m) <= 1.0]
    assert win_rates == [], (
        f"the control-arm row carries win-rate-shaped numbers {win_rates}. No dense eval "
        f"result exists and none can be produced at HEAD; O-A16 cannot see this, because a "
        f"fabricated number has no config counterpart to disagree with."
    )


def test_control_arm_row_cites_the_site_that_actually_refuses() -> None:
    """O-A17 arm (d). `v6_live2_ls` declares `representation="grid"` (IMPLEMENTED) and
    `value_pool="min"` (IMPLEMENTED); only `policy_pool="legal_set_scatter_max"` is not, so
    the ONE refusing site is `_assert_policy_pool_implemented` at `worker.py:87-98`.
    Attributing the refusal to representation or to the value pool is the exact error
    DESIGN_A rev-2 made and rev-3 re-pointed."""
    section = _section_named(_doc(), _CONTROL_ARM_HEADING)
    assert "worker.py:87-98" in section, (
        "the control-arm row must cite the single site that refuses this encoding"
    )
    for wrong in ("representation", "value_pool"):
        assert wrong not in section, (
            f"the control-arm row attributes the refusal to {wrong!r}; both are IMPLEMENTED "
            f"for `v6_live2_ls` and saying otherwise describes a refusal that does not happen"
        )


# ── O-A19 — the decision is in a legal state ────────────────────────────────────────────
def test_the_liveness_line_matches_exactly_one_of_the_four_state_templates() -> None:
    """O-A19 arm 1a. FOUR templates, not three: 3a (`not_run`, not completed) and 3b
    (`FAILED`, measured and wrong) are the same distinction as R151's `not_run` vs
    `covered`, and DESIGN_A §8.4 calls it the finest in the design and the easiest to lose
    in transit."""
    doc = _doc()
    present = [heading for heading in _STATE_HEADINGS if heading in doc]
    assert len(present) == 1, (
        f"expected exactly ONE of DESIGN_A §3.6's four state headings; found {present}. "
        f"Two states visible at once is the drift R96 forbids (PREREG_A §8 abort 10)."
    )


def test_the_liveness_section_carries_exactly_one_state_marker() -> None:
    """O-A19 arm 1b — M-A19's observer. R96's upgrade is IN PLACE: the line is REPLACED,
    never appended to. Without this row, 'in place' is a sentence in a design doc with no
    observer at all."""
    doc = _doc()
    heading = next(h for h in _STATE_HEADINGS if h in doc)
    section = _section_named(doc, "Ladder liveness")
    found = [marker for marker in _STATE_MARKERS if marker in section]
    assert found == [_STATE_HEADINGS[heading]], (
        f"the liveness section carries markers {found}; its heading declares "
        f"{_STATE_HEADINGS[heading]!r} and exactly that one may survive the edit"
    )
    assert _R169_SANCTIONED in section, (
        f"R169's sanctioned wording {_R169_SANCTIONED!r} is absent from the liveness section"
    )


def test_the_two_honesty_clauses_are_separate_and_no_not_run_is_bare() -> None:
    """O-A19 arm 2. Two not-yet-true claims from two rulings, and they are different in
    KIND: R151's is a design decision with an owner (the code refuses, deliberately);
    R169's is an unrun measurement with a scheduled instrument (the code is fine; the
    environment has not run it). Collapsing them is what R169's dispatch forbids."""
    doc = _doc()
    control = _section_named(doc, _CONTROL_ARM_HEADING)
    liveness = _section_named(doc, "Ladder liveness")
    assert control != liveness, "the two clauses resolved to the same section body"

    bare = [
        line for line in doc.splitlines()
        if "not_run" in line and re.search(r"not_run`?\s*[—-]", line) is None
    ]
    assert bare == [], (
        "every `not_run` is qualified in the same sentence — `not_run — refused by design, "
        f"adapter owed` versus `not_run — environment-conditional, verified at box "
        f"preflight`. A bare `not_run` in either section is a drift. Offenders: {bare}"
    )


def test_the_control_arm_section_carries_no_status_line() -> None:
    """O-A19 arm 3 — the discriminator, ASSERTED. The marker WORD cannot discriminate: both
    clauses carry `not_run` by DESIGN_A §3.4's own table. Only the liveness clause carries a
    dated `*Status: … on <date> at commit <sha>*`; the R151 clause carries none, because it
    has nothing dated to report. Without this row a later edit gives it one and the
    convention decays into nothing."""
    doc = _doc()
    assert _STATUS_RE.search(_section_named(doc, "Ladder liveness")) is not None, (
        "the liveness clause must carry the dated *Status:* line that discriminates it"
    )
    control = _section_named(doc, _CONTROL_ARM_HEADING)
    assert _STATUS_RE.search(control) is None, (
        "the R151 control-arm section grew a *Status:* line, collapsing the ONE asymmetry "
        "that tells the two honesty clauses apart (DESIGN_A §3.6 rule 3)"
    )


def test_no_bare_liveness_claim_anywhere_in_the_decision() -> None:
    """O-A19 arm 4 — abort 11's instrument. R169: *'2/6 live rungs' is NOT claimable bare;
    a liveness claim with no runnable producer is the R69/LAW-07 class.* Scanned over the
    WHOLE doc outside the §3.6 liveness line, because the phrase can re-enter through a
    summary paragraph as easily as through the line itself."""
    doc = _doc()
    assert re.search(r"\d\s*/\s*6\s+live", doc) is None, (
        "the phrase '2/6 live rungs' (or a rung-count liveness claim in any form) appears "
        "in the decision. The sanctioned wording is R169's and nothing else."
    )

    liveness = _section_named(doc, "Ladder liveness")
    liveness_lines = set(liveness.splitlines())
    offenders = [
        line for line in doc.splitlines()
        if line not in liveness_lines
        and any(re.search(rf"\b{re.escape(t)}\b", line) for t in _LIVENESS_TOKENS)
        and any(t in line for t in _RUNG_TOKENS)
    ]
    assert offenders == [], (
        f"a liveness predicate is applied to the rungs outside the §3.6 line: {offenders}. "
        f"Until the box rider runs, the correct word is `not_run`, and `not_run` is a "
        f"RESULT, not an absence."
    )


#: ⊕ G-A4 / RED-TEAM F-RT-6. Word window for the whitespace-normalised scan below. MEASURED at
#: the re-point, never chosen: the claim abort 11 forbids puts its tokens **2** words apart
#: ("both sealbot rungs **are live** in run5"), and the CLOSEST innocuous pair in the shipped
#: decision is **11** apart ("…sealbot rungs resolve against the vendored engine pinned in
#: `vendor/pins.toml`; see the liveness clause below…"). 6 sits with 3x headroom below the
#: malicious distance and almost 2x above it on the innocuous side, so neither direction is
#: near the boundary.
_LIVENESS_PROXIMITY_WORDS = 6


def test_no_bare_liveness_claim_survives_a_LINE_REWRAP() -> None:
    """⊕ G-A4 — abort 11's instrument, made independent of where paragraphs happen to wrap.

    **Why the row above is not enough, measured by RED-TEAM in both directions.** It scans per
    PHYSICAL LINE and requires both tokens on the same one. The identical claim therefore reds
    on one line and passes across two — so the shipped document passes partly *because of its
    formatting*, and it sits one editor re-wrap from a false RED on innocuous prose and one
    newline from a **false GREEN** on a real claim. `PREREG_A` W-6 discloses arm 4's limit for
    PARAPHRASE; the line-unit limit is a different failure mode and was disclosed nowhere.

    This row normalises all whitespace first, so a newline is worth exactly one space, and then
    asks a PROXIMITY question rather than a line or sentence one. A sentence window would be
    the obvious choice and it is wrong here: the shipped doc's closest innocuous pair sits in
    ONE sentence, joined by a semicolon, so a sentence window would false-RED on it. The unit
    has to be distance, and the distance is measured rather than guessed (see the constant).

    **Additive, not a replacement.** The per-line row above still runs and still owns its own
    kill; this one closes the seam between its lines. Neither subsumes the other: the per-line
    row catches a token pair that a wrap would push outside this window's reach.
    """
    doc = _doc()
    liveness = _section_named(doc, "Ladder liveness")
    words = " ".join(doc.replace(liveness, " ").split()).split(" ")

    live_at = [
        i for i, w in enumerate(words)
        if any(re.fullmatch(rf"\W*{re.escape(t)}\W*", w, re.IGNORECASE) for t in _LIVENESS_TOKENS)
    ]
    rung_at = [i for i, w in enumerate(words) if any(t in w.lower() for t in _RUNG_TOKENS)]

    offenders = [
        " ".join(words[max(0, min(a, b) - 2): max(a, b) + 3])
        for a in live_at for b in rung_at
        if abs(a - b) <= _LIVENESS_PROXIMITY_WORDS
    ]
    assert offenders == [], (
        f"a liveness predicate sits within {_LIVENESS_PROXIMITY_WORDS} words of a rung, "
        f"outside the §3.6 line, once the document's line breaks are normalised away: "
        f"{offenders}. R169: the claim is not claimable bare, and where a paragraph wraps is "
        f"not a defence."
    )
