"""Line-ending discipline - the pin for the `.gitattributes` protection (P0-02).

Byte-significant files are hashed, embedded, or compared byte-for-byte. `core.autocrlf` is a
PER-DEVELOPER git setting that defaults to `true` on Git for Windows, so without repository-level
attributes the bytes in a working tree depended on who cloned it. Measured before `.gitattributes`
existed: 33 of 65 manifest-pinned fixtures failed their sha256 on a Windows checkout, and
`crates/mantis-encoding/src/registry.toml` - whose sha256 IS the cross-language identity handshake
- hashed to `1c439678...` instead of `86be0cf1...`, so the compiled extension carried a registry
identity that no Linux build can produce.

Two independent pins here, because they fail for different reasons:

* `test_gitattributes_marks_every_byte_significant_path` guards the CAUSE. It fails when a rule is
  weakened, or when a new byte-significant file is added outside the covered globs - the drift that
  would otherwise be discovered months later as an unexplainable digest mismatch.
* `test_no_byte_significant_file_was_crlf_converted` guards the SYMPTOM, and names it. The failure
  it replaces ("sha256 mismatch") sent the reader toward re-minting the manifest, which is the trap:
  the manifest is right, the checkout is wrong, and a re-mint breaks Linux CI.

Deliberately NOT "assert no fixture contains b'\\r\\n'": 3 of the 65 fixtures legitimately do
(`small_cnn_scalar.pt`, `small_cnn_aux_chain.pt`, `b6_hotpath.npz` - chance byte pairs inside
pickled/compressed streams). That rule would red on correct files. The comparison below is against
git's own stored blob, so it fires only when the working tree differs from the repository *by line
endings alone*, and stays silent on ordinary content edits.

This file is deliberately pure ASCII. Its assertion messages are read in a terminal, and a
non-ASCII message mojibakes on a cp1252 console - which is precisely the platform whose default
setting causes the defect being diagnosed.
"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = FIXTURES_ROOT / "manifest.toml"

#: Byte-significant beyond the fixture manifest: both are `include_str!`-ed into the engine
#: (`crates/mantis-encoding/src/registry/mod.rs:25`, `manifests.rs:13`), and registry.toml's
#: sha256 is exported across the FFI as `_engine.registry_sha()` and handshaken by CI gate 8.
EMBEDDED_BYTE_SIGNIFICANT = (
    "crates/mantis-encoding/src/registry.toml",
    "crates/mantis-encoding/src/manifests.toml",
)


def _git_bytes(*args: str, stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def byte_significant_paths() -> list[str]:
    """Every repo-relative path whose exact bytes are load-bearing.

    Derived, never hand-listed: the fixture half is read out of `manifest.toml` itself, so a row
    added tomorrow is covered by both pins on the next run without anyone remembering to edit this
    file. Same derive-don't-transcribe rule the manifest checker follows.
    """
    rows = tomllib.loads(MANIFEST_PATH.read_bytes().decode("utf-8"))["required"]
    paths = [f"tests/fixtures/{row['path']}" for row in rows]
    paths.extend(EMBEDDED_BYTE_SIGNIFICANT)
    return paths


def _head_blobs(paths: list[str]) -> dict[str, bytes]:
    """`path -> bytes as git stores them`, for paths present in HEAD.

    One `git cat-file --batch` call rather than N `git show`s: 67 subprocess spawns is a
    measurable cost on Windows and this runs in the default tier.
    """
    stdin = "".join(f"HEAD:{p}\n" for p in paths).encode("utf-8")
    out = _git_bytes("cat-file", "--batch", stdin=stdin)
    blobs: dict[str, bytes] = {}
    pos = 0
    for path in paths:
        newline = out.index(b"\n", pos)
        header = out[pos:newline].decode("utf-8", "replace")
        if header.endswith(("missing", "ambiguous")):
            pos = newline + 1
            continue
        size = int(header.rsplit(" ", 1)[1])
        start = newline + 1
        blobs[path] = out[start : start + size]
        pos = start + size + 1  # git appends a newline after the payload
    return blobs


def crlf_diagnosis(path: str, disk: bytes, blob: bytes) -> str | None:
    """Return a named diagnosis iff `disk` is `blob` with LF expanded to CRLF, else None.

    The `disk != blob` guard keeps ordinary content edits out: only a pure line-ending expansion is
    reported, so this cannot red on a legitimately re-minted golden.
    """
    if disk == blob:
        return None
    if disk.replace(b"\r\n", b"\n") != blob:
        return None
    injected = len(disk) - len(blob)
    return (
        f"{path}: LINE-ENDING CORRUPTION - the working-tree copy is the committed file with "
        f"every LF expanded to CRLF ({injected} 0x0D bytes injected, {len(blob)} -> {len(disk)}).\n"
        f"  CAUSE: git converted it at checkout (core.autocrlf), so its bytes - and therefore its "
        f"sha256 - differ from what every other platform sees.\n"
        f"  DO NOT re-mint tests/fixtures/manifest.toml: the manifest is correct and a re-mint "
        f"would break Linux CI. The checkout is what is wrong.\n"
        f"  FIX: ensure .gitattributes covers this path with `-text`, then\n"
        f"       git rm --cached -r . && git reset --hard"
    )


def test_gitattributes_exists_and_sets_an_lf_default() -> None:
    """The file itself, plus the default that keeps every working tree byte-identical to CI's."""
    attrs = REPO_ROOT / ".gitattributes"
    assert attrs.is_file(), (
        ".gitattributes is missing. Without it, line endings are decided by each developer's "
        "core.autocrlf, which is `true` by default on Git for Windows."
    )
    text = attrs.read_text(encoding="utf-8")
    rules = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    default = [rule for rule in rules if rule.startswith("* ")]
    assert default, (
        "no `*` default rule: files not matched by a specific rule would fall back to "
        "core.autocrlf, which is the per-developer setting this file exists to remove"
    )
    assert "eol=lf" in default[0], f"the `*` default must pin eol=lf, got: {default[0]!r}"


def test_gitattributes_marks_every_byte_significant_path() -> None:
    """CAUSE pin: every hashed/embedded path must resolve to `-text` (git reports `unset`).

    Derived from the manifest, so a fixture added under a directory the globs miss fails here
    rather than silently inheriting the text default.
    """
    paths = byte_significant_paths()
    assert len(paths) >= 60, (
        f"census collapsed to {len(paths)} paths: the derivation is broken and this pin "
        "would be vacuous"
    )

    out = _git_bytes("check-attr", "text", "--", *paths).decode("utf-8")
    verdicts = {}
    for line in out.splitlines():
        path, _, value = line.rpartition(": text: ")
        verdicts[path] = value

    unprotected = sorted(p for p in paths if verdicts.get(p) != "unset")
    assert not unprotected, (
        f"{len(unprotected)} byte-significant path(s) are NOT marked `-text` in .gitattributes, "
        f"so git may line-ending-convert them at checkout and change their sha256:\n  "
        + "\n  ".join(unprotected)
    )


def test_no_byte_significant_file_was_crlf_converted() -> None:
    """SYMPTOM pin: the working tree must not differ from git's blob by line endings alone."""
    paths = byte_significant_paths()
    blobs = _head_blobs(paths)
    assert blobs, "no HEAD blobs resolved: the comparison would be vacuous"

    problems = []
    for path in paths:
        blob = blobs.get(path)
        if blob is None:
            continue  # not yet committed; the manifest sha test still covers its content
        diagnosis = crlf_diagnosis(path, (REPO_ROOT / path).read_bytes(), blob)
        if diagnosis is not None:
            problems.append(diagnosis)

    assert not problems, (
        f"{len(problems)} byte-significant file(s) were line-ending converted at checkout:\n\n"
        + "\n\n".join(problems)
    )


def test_shell_scripts_are_lf_in_the_working_tree() -> None:
    """`#!/usr/bin/env bash` + CRLF = `bad interpreter: ...^M`. All 5 gate scripts are shebanged."""
    scripts = sorted(REPO_ROOT.glob("tools/**/*.sh"))
    assert scripts, "no shell scripts found under tools/: this pin would be vacuous"
    crlf = [
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in scripts
        if b"\r\n" in path.read_bytes()
    ]
    assert not crlf, (
        "shell script(s) have CRLF line endings; their shebang will fail with "
        f"`bad interpreter: ...^M`: {crlf}"
    )


# --- LAW-07: the detector's own trigger, self-tested --------------------------------------


def test_crlf_diagnosis_bites_on_a_converted_file() -> None:
    """A pin that cannot fail is decoration. Prove the detector fires and names the cause."""
    blob = b"alpha\nbeta\ngamma\n"
    disk = blob.replace(b"\n", b"\r\n")
    message = crlf_diagnosis("tests/fixtures/example.tsv", disk, blob)
    assert message is not None, "detector missed a pure LF->CRLF expansion"
    assert "LINE-ENDING CORRUPTION" in message
    assert "DO NOT re-mint" in message, "the message must steer away from the manifest re-mint trap"
    assert "3 0x0D bytes injected" in message, f"byte accounting wrong: {message}"
    assert message.isascii(), "the message must render identically on a cp1252 console"


def test_crlf_diagnosis_is_silent_on_clean_and_on_content_edits() -> None:
    """No false positives: identical bytes, and ordinary edits, must both report nothing."""
    blob = b"alpha\nbeta\n"
    assert crlf_diagnosis("x", blob, blob) is None, "clean file reported as corrupted"
    assert crlf_diagnosis("x", b"alpha\ndelta\n", blob) is None, "content edit misreported as CRLF"


def test_crlf_diagnosis_is_silent_on_binaries_that_contain_crlf_natively() -> None:
    """3 of the 65 fixtures contain b'\\r\\n' legitimately (.pt / .npz compressed streams).

    Encoded as a real case rather than a comment: a naive `b'\\r\\n' not in raw` pin would red on
    correct files, and this is the arm that stops someone "simplifying" it back to that.
    """
    blob = b"PK\x03\x04\r\n\x00\x91payload\r\n"
    assert crlf_diagnosis("f.npz", blob, blob) is None, "native CRLF in a binary flagged as damage"


@pytest.mark.parametrize("path", list(EMBEDDED_BYTE_SIGNIFICANT))
def test_embedded_engine_inputs_exist(path: str) -> None:
    """The census must not silently shrink if one of these is moved or renamed."""
    assert (REPO_ROOT / path).is_file(), (
        f"{path} is byte-significant (include_str!'d into the engine) but is missing; "
        "update EMBEDDED_BYTE_SIGNIFICANT and .gitattributes together"
    )
