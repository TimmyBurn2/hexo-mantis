# >300 justify (R8): the rows here are ONE claim — the allocator posture has exactly ONE
# authority, is explicit in every config, cannot be silently absent, and cannot be read from the
# environment anywhere else in `src/` — over one apparatus: the real loader, `discover_configs`,
# one `ast` census over `src/`, and one env-precedence table taken from c10's own source.
"""The allocator posture's config authority.

THE MINT POSTURE, so the rows read correctly: `allocator_posture` is a CLOSED TOKEN SET
(`default` | `expandable_segments`) or `null`, top-level, REQUIRED, with no schema default and
no code-side default anywhere. `null` is a PLACEHOLDER, not an off state — schema-VALID so gate
7 stays green, runtime-REFUSED so a CUDA run on an unminted posture cannot boot.

The defect each row is the ONLY witness to: a key minted into some configs and not others; a
code-side default, whose worst failure mode is a cap fitted under one posture and run under
another; `null` quietly meaning "whatever the launch happens to be"; the assertion reading the
WRONG variable, since c10 reads `PYTORCH_CUDA_ALLOC_CONF` FIRST and falls back to
`PYTORCH_ALLOC_CONF`; a FOREIGN conf reading as the minted one; a second env reader in `src/`;
the eval child skipping the assertion on a `None` posture; and the armed-abort row reading a
token as disarmed or `null` as armed.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mantis.config.armed_aborts import MANIFEST, Cadence, Mechanism, Status
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.allocator_posture import (
    ALLOC_CONF_VARS,
    AllocatorPosture,
    AllocatorPostureMismatchError,
    MissingAllocatorPostureError,
    UncalibratedAllocatorPostureError,
    assert_allocator_posture,
    assert_posture_token,
    device_type_of,
    parse_alloc_conf,
    read_live_allocator_conf,
    resolve_allocator_posture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
TEMPLATES_DIR = REPO_ROOT / "tools" / "config_templates"
RESOLVER_REL = "mantis/config/resolve/allocator_posture.py"


def test_ap01_every_committed_config_declares_the_posture_key():
    """Every config `discover_configs` enumerates carries `allocator_posture` explicitly."""
    configs = discover_configs(CONFIGS_DIR)
    assert configs, "discover_configs found nothing — the sweep below would assert nothing"
    for path in configs:
        cfg = load_config(path)
        assert hasattr(cfg, "allocator_posture"), path.name


def test_ap01_both_mint_templates_declare_the_posture_key():
    """A template that omits it would mint configs that omit it, failing gate 7 one layer later."""
    import yaml

    templates = sorted(TEMPLATES_DIR.glob("*.yaml"))
    assert templates, "no mint templates found"
    for path in templates:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "allocator_posture" in data, path.name


def test_ap03_every_committed_config_MINTS_A_MEASURED_POSTURE():
    """Every committed config MINTS A MEASURED POSTURE — INVERTED from its placeholder-era
    predecessor, not relaxed. It still catches a config reverting to `null` (the run would refuse
    to boot and nobody would know until it did) and one minting a token OUTSIDE the closed
    vocabulary, read off `AllocatorPosture` itself. It does NOT pin WHICH member."""
    vocabulary = {p.value for p in AllocatorPosture}
    for path in discover_configs(CONFIGS_DIR):
        posture = load_config(path).allocator_posture
        assert posture is not None, (
            f"{path.name} carries the R119 `null` placeholder again. RECAL-SITTING-5 minted a "
            "measured posture into all seven under R326; a cuda process REFUSES to boot on a "
            "null, so this reverting is a run-fatal regression that no other row would see"
        )
        assert posture in vocabulary, (
            f"{path.name} mints {posture!r}, which is not in the closed regime vocabulary "
            f"{sorted(vocabulary)} — a token nobody measured is the defect this row was "
            "written for, and it survives the inversion"
        )


def test_ap01_a_config_missing_the_key_fails_to_load(tmp_path):
    """PLANTED BREAK: delete the key from a real minted config -> ValueError at load, satisfied
    by the field being REQUIRED rather than by a hand-written check that could be skipped."""
    import yaml

    raw = yaml.safe_load((CONFIGS_DIR / "run6.yaml").read_text(encoding="utf-8"))
    del raw["allocator_posture"]
    victim = tmp_path / "no_posture.yaml"
    victim.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(victim)


def test_ap02_the_schema_field_carries_no_default():
    """A default here is a code-side default for a measured regime (R1)."""
    from mantis.config.schema import RunConfig

    field = RunConfig.model_fields["allocator_posture"]
    assert field.is_required(), (
        "allocator_posture is optional-with-a-default: an unminted config would then boot "
        "under whatever the launch happened to be, which is the silent-precondition class"
    )


def _conf_strings_in_operative_position(tree: ast.AST) -> list[str]:
    """Return string constants that look like an allocator conf AND sit where one would ACT —
    assignments, comparisons, dict keys/values and returns, never a docstring and (being an AST
    walk) never a `#` comment. A raw-text census would flag the repo's own explanatory prose."""
    hits: list[str] = []

    def _consider(node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "expandable_segments:" in node.value.lower():
                hits.append(node.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
            _consider(node.value)
        elif isinstance(node, ast.Compare):
            _consider(node.left)
            for comparator in node.comparators:
                _consider(comparator)
        elif isinstance(node, ast.Dict):
            for element in [*node.keys, *node.values]:
                _consider(element)
    return hits


def test_ap02_no_module_in_src_puts_a_posture_conf_string_where_it_could_act():
    """The conf strings live in ONE module. A literal elsewhere is a second authority."""
    offenders = {}
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == f"src/{RESOLVER_REL}":
            continue
        found = _conf_strings_in_operative_position(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if found:
            offenders[rel] = found
    assert not offenders, offenders


def test_ap02_the_operative_position_census_has_both_controls(tmp_path):
    """It must SEE a planted comparison and must NOT see the prose beside it."""
    planted = ast.parse(
        "DOC = 1  # expandable_segments:True described in a comment\n"
        "NOTE = ('the sitting measured expandable_segments:True at 11.36 GiB',)\n"
        "def f(conf):\n"
        "    return conf == 'expandable_segments:True'\n"
    )
    assert _conf_strings_in_operative_position(planted) == ["expandable_segments:True"]
    prose_only = ast.parse(
        '"""expandable_segments:True is discussed here and never compared."""\n'
        "X = 1  # expandable_segments:True\n"
    )
    assert _conf_strings_in_operative_position(prose_only) == []


def test_ap03_absent_section_raises_naming_the_level():
    with pytest.raises(MissingAllocatorPostureError) as exc:
        resolve_allocator_posture({"train": {}})
    assert "allocator_posture" in str(exc.value)


def test_ap03_not_a_mapping_raises():
    with pytest.raises(MissingAllocatorPostureError):
        resolve_allocator_posture(["not", "a", "mapping"])


def test_ap03_null_raises_the_uncalibrated_subclass_with_the_mint_line():
    with pytest.raises(UncalibratedAllocatorPostureError) as exc:
        resolve_allocator_posture({"allocator_posture": None})
    message = str(exc.value)
    assert "mint_config.py" in message, "the refusal must carry the remedy, not just the fault"
    assert issubclass(UncalibratedAllocatorPostureError, MissingAllocatorPostureError)


def test_ap03_an_unknown_token_raises_rather_than_being_carried():
    with pytest.raises(MissingAllocatorPostureError):
        resolve_allocator_posture({"allocator_posture": "expandable"})


@pytest.mark.parametrize("token", [p.value for p in AllocatorPosture])
def test_ap03_every_token_resolves_and_names_its_required_conf(token):
    spec = resolve_allocator_posture({"allocator_posture": token})
    assert spec.posture.value == token
    assert isinstance(spec.required_conf, dict)


def test_ap04_the_variable_pair_is_the_pair_c10_reads():
    assert ALLOC_CONF_VARS == ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def test_ap04_cuda_specific_variable_wins_when_both_are_set():
    """c10 reads PYTORCH_CUDA_ALLOC_CONF FIRST; PYTORCH_ALLOC_CONF is the FALLBACK."""
    live = read_live_allocator_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
         "PYTORCH_ALLOC_CONF": "max_split_size_mb:128"}
    )
    assert live.source_var == "PYTORCH_CUDA_ALLOC_CONF"
    assert live.parsed == {"expandable_segments": "True"}


def test_ap04_generic_variable_is_read_when_the_cuda_one_is_absent():
    live = read_live_allocator_conf({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    assert live.source_var == "PYTORCH_ALLOC_CONF"
    assert live.parsed == {"expandable_segments": "True"}


def test_ap04_neither_set_reads_as_the_empty_conf_and_names_no_variable():
    live = read_live_allocator_conf({})
    assert live.source_var is None
    assert live.parsed == {}
    assert live.raw == ""


def test_ap04_an_empty_string_is_the_empty_conf_but_names_its_variable():
    """`PYTORCH_CUDA_ALLOC_CONF=""` is the DEFAULT posture and the sitting's own stamp."""
    live = read_live_allocator_conf({"PYTORCH_CUDA_ALLOC_CONF": ""})
    assert live.parsed == {}
    assert live.source_var == "PYTORCH_CUDA_ALLOC_CONF"


def test_ap04_both_variables_set_and_disagreeing_is_AMBIGUOUS_not_a_guess():
    """Whether an EMPTY value counts as set is the one edge c10's headers do not settle, so it is
    reported ambiguous and REFUSED rather than guessed."""
    live = read_live_allocator_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    )
    assert live.ambiguous is True
    assert live.source_var is None
    with pytest.raises(AllocatorPostureMismatchError) as exc:
        assert_posture_token(
            "expandable_segments", device_type="cuda",
            environ={"PYTORCH_CUDA_ALLOC_CONF": "",
                     "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )
    assert "ambiguous" in str(exc.value).lower() or "cannot be told" in str(exc.value)


def test_ap04_both_variables_set_and_AGREEING_is_not_ambiguous():
    """Ambiguity is a real predicate: two variables saying the same thing is not a conflict."""
    live = read_live_allocator_conf({"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": ""})
    assert live.ambiguous is False
    assert live.parsed == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("expandable_segments:True", {"expandable_segments": "True"}),
        # c10's `ConfigTokenizer` skips EVERY isspace character, wherever it appears, so this is
        # the same conf to torch and must be here too.
        ("  expandable_segments : True  ", {"expandable_segments": "True"}),
        ("expandable_ segments:Tr ue", {"expandable_segments": "True"}),
        ("a:1,b:2", {"a": "1", "b": "2"}),
        ("a:1, ,b:2", {"a": "1", "b": "2"}),
        ("", {}),
    ],
)
def test_ap04_parse_follows_c10s_own_grammar(raw, expected):
    assert parse_alloc_conf(raw) == expected


def test_ap04_case_is_LOAD_BEARING_and_is_not_normalised_away():
    """`toBool` accepts EXACTLY `True`/`False` and keys match by exact token equality, so
    `expandable_segments:true` is not a spelling variant — torch REFUSES it — and a check that
    lower-cased would bless an environment the allocator will not accept."""
    assert parse_alloc_conf("expandable_segments:true") != parse_alloc_conf(
        "expandable_segments:True"
    )
    with pytest.raises(AllocatorPostureMismatchError):
        assert_posture_token(
            "expandable_segments", device_type="cuda",
            environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:true"},
        )


def test_ap04_the_launch_hint_is_a_spelling_torch_actually_accepts():
    """The refusal names a launch line an operator will paste, and `toBool` accepts only
    `True`/`False`, so a hint carrying `true` would send them into a TORCH_CHECK_VALUE."""
    from mantis.config.resolve.allocator_posture import AllocatorPostureSpec

    hint = AllocatorPostureSpec(AllocatorPosture.EXPANDABLE_SEGMENTS).launch_hint
    assert "expandable_segments:True" in hint
    assert "true" not in hint.replace("True", "")
    # The DEFAULT posture's hint is the absence of a configuration, not a spelling of one.
    default_hint = AllocatorPostureSpec(AllocatorPosture.DEFAULT).launch_hint
    assert "unset" in default_hint
    for var in ALLOC_CONF_VARS:
        assert var in default_hint


def test_ap05_matching_posture_passes_and_records_what_it_read():
    record = assert_posture_token(
        "expandable_segments", device_type="cuda",
        environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    assert record["enforced"] is True
    assert record["source_var"] == "PYTORCH_CUDA_ALLOC_CONF"
    assert record["posture"] == "expandable_segments"


def test_ap05_posture_a_minted_process_launched_under_posture_b_raises():
    """ORACLE obligation 1 — the packet's first planted break."""
    with pytest.raises(AllocatorPostureMismatchError) as exc:
        assert_posture_token("expandable_segments", device_type="cuda", environ={})
    message = str(exc.value)
    assert "expandable_segments" in message
    assert "PYTORCH_CUDA_ALLOC_CONF" in message, "the refusal must name the variable to set"


def test_ap05_default_posture_refuses_a_set_variable():
    """The mismatch is symmetric: a cap fitted under DEFAULT is invalid under expandable."""
    with pytest.raises(AllocatorPostureMismatchError):
        assert_posture_token(
            "default", device_type="cuda",
            environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )


def test_ap05_a_foreign_key_alongside_the_required_one_refuses():
    with pytest.raises(AllocatorPostureMismatchError) as exc:
        assert_posture_token(
            "expandable_segments", device_type="cuda",
            environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"},
        )
    assert "max_split_size_mb" in str(exc.value)


def test_ap05_default_posture_accepts_an_explicitly_empty_variable():
    record = assert_posture_token(
        "default", device_type="cuda", environ={"PYTORCH_CUDA_ALLOC_CONF": ""},
    )
    assert record["enforced"] is True


@pytest.mark.parametrize("device", ["cpu", "CPU", "mps", "cpu:0"])
def test_ap05_a_non_cuda_process_is_not_enforced_and_says_why(device):
    """Scoped, not skipped: the record states the reason, so 'not enforced' is legible."""
    record = assert_posture_token(None, device_type=device, environ={})
    assert record["enforced"] is False
    assert record["reason"]


@pytest.mark.parametrize(("name", "expected"), [("cuda", "cuda"), ("cuda:0", "cuda"),
                                                ("CUDA:1", "cuda"), ("cpu", "cpu")])
def test_ap05_device_type_ignores_the_ordinal(name, expected):
    assert device_type_of(name) == expected


def test_ap03_a_cuda_process_on_a_null_posture_refuses_to_boot():
    with pytest.raises(UncalibratedAllocatorPostureError):
        assert_allocator_posture({"allocator_posture": None}, device_type="cuda", environ={})


def test_ap03_a_cpu_process_on_a_null_posture_boots():
    """The scoping is principled: no CUDA device, no CUDA caching allocator, no posture."""
    record = assert_allocator_posture(
        {"allocator_posture": None}, device_type="cpu", environ={},
    )
    assert record["enforced"] is False


def test_ap07_a_cuda_child_with_no_posture_token_raises():
    with pytest.raises(AllocatorPostureMismatchError) as exc:
        assert_posture_token(None, device_type="cuda", environ={})
    assert "cuda" in str(exc.value).lower()


def _alloc_conf_var_constants(tree: ast.AST) -> list[str]:
    """Return every string constant naming an allocator-conf ENVIRONMENT VARIABLE. STRUCTURE,
    not text, so a `#` comment is not a read. Keyed on the VARIABLE NAME rather than the
    `os.environ` access, because the authority reads the pair through `ALLOC_CONF_VARS` and a
    subscript-keyed census would find ZERO sites. THE LIMIT: a name assembled at runtime evades it."""
    keys = set(ALLOC_CONF_VARS) | {"PYTORCH_HIP_ALLOC_CONF"}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value in keys]


def test_ap06_exactly_one_module_in_src_names_the_alloc_conf_environment_variables():
    sites = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _alloc_conf_var_constants(tree):
            sites.append(path.relative_to(REPO_ROOT).as_posix())
    assert sites == [f"src/{RESOLVER_REL}"], (
        "the allocator conf environment is named outside its one authority: "
        f"{sites}. A second reader is a second answer to 'what posture is this process in', "
        "and the two diverge silently — which is exactly how `fusion_calibrate`'s provenance "
        "block came to stamp an empty posture for a fit that may have been taken under a set "
        "`PYTORCH_ALLOC_CONF`, on the one tool whose fragmentation ratio the whole partition "
        "divides by."
    )


def test_ap06_the_census_sees_an_aliased_env_read(tmp_path):
    """An allowlist never shown to reject anything is indistinguishable from one accepting all."""
    planted = tmp_path / "smuggler.py"
    planted.write_text(
        "import os as _o\n"
        "def f():\n"
        "    return _o.environ.get('PYTORCH_ALLOC_CONF', '')\n",
        encoding="utf-8",
    )
    assert _alloc_conf_var_constants(ast.parse(planted.read_text(encoding="utf-8")))


def test_ap06_the_census_does_not_fire_on_a_comment(tmp_path):
    planted = tmp_path / "innocent.py"
    planted.write_text(
        "# PYTORCH_CUDA_ALLOC_CONF is described here and never read\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )
    assert not _alloc_conf_var_constants(ast.parse(planted.read_text(encoding="utf-8")))


def test_ap06_the_one_authority_reads_the_pair_and_the_pair_is_c10s():
    """The census is only as good as the module it allows: an allowlist over a reader that reads
    one of the two variables would pass while the other went unchecked."""
    source = (REPO_ROOT / "src" / RESOLVER_REL).read_text(encoding="utf-8")
    named = set(_alloc_conf_var_constants(ast.parse(source)))
    assert set(ALLOC_CONF_VARS) <= named


def _row(name: str):
    matches = [r for r in MANIFEST if r.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} row, found {len(matches)}"
    return matches[0]


def test_ap08_the_row_exists_and_names_the_top_level_key():
    row = _row("allocator_posture_minted")
    assert row.config_path == "allocator_posture"
    assert row.mechanism is Mechanism.CONFIG_ENUM_VALUED
    assert row.cadence is Cadence.CONSTRUCTION_TIME
    assert row.cadence_paths == ()


def test_ap08_the_row_is_REQUIRED_and_therefore_unowned_and_still_pinned():
    """The row is REQUIRED and therefore unowned, and still pinned. `owner` is None, NOT ABSENT:
    `ArmedAbort` takes it positionally, so dropping the keyword is a `TypeError` at import. The
    pin stays, because a deleted refusal would make this the phantom gate input LAW-07 prevents."""
    row = _row("allocator_posture_minted")
    assert row.status is Status.REQUIRED
    assert row.owner is None, (
        "a REQUIRED row has no owner — the debt is discharged; and it must be None rather than "
        "removed, because the dataclass takes it positionally (F-RESIT-5)"
    )
    assert row.source_pin, "a required row that is not tamper-evident is worse, not better"
    assert row.exit_code is None


def test_ap08_the_row_source_pin_still_matches_the_tree():
    """The pin is tamper-evidence only while it resolves. This is that check."""
    row = _row("allocator_posture_minted")
    rel, needle = row.source_pin
    assert needle in (REPO_ROOT / rel).read_text(encoding="utf-8"), (rel, needle)


@pytest.mark.parametrize(
    ("value", "armed"),
    [(None, False), ("", False), ("default", True), ("expandable_segments", True),
     (0, False), (1, False), (True, False)],
)
def test_ap08_the_token_predicate_is_real_in_both_directions(value, armed):
    assert Mechanism.CONFIG_ENUM_VALUED.is_armed(value) is armed


@pytest.mark.parametrize("mechanism", [m for m in Mechanism
                                       if m is not Mechanism.CONFIG_ENUM_VALUED])
@pytest.mark.parametrize("value", [None, 0, 1, -1, True, False])
def test_ap08_the_existing_predicates_are_byte_unchanged_by_the_new_member(mechanism, value):
    """A new enum member must change no other row's verdict. A `KeyError` on an undeclared
    mechanism forces a new member's verdict to be STATED here rather than inherited, and the
    live-producer member's entry is the same OBJECT as the `> 0` row, not a copy of it."""
    gt_zero = (not isinstance(value, bool) and isinstance(value, (int, float))
               and float(value) > 0.0)
    expected = {
        Mechanism.CONFIG_BOOL: value is True,
        Mechanism.CONFIG_THRESHOLD_GT_ZERO: gt_zero,
        Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER: gt_zero,
        Mechanism.CONFIG_THRESHOLD_BELOW_CEILING: False,
    }[mechanism]
    assert mechanism.is_armed(value) is expected


def test_ap08_gate_12_audit_passes_with_the_row_now_REQUIRED(tmp_path):
    """The row now GATES, and the gate must be green — measured by running it, with rc read
    directly rather than through a pipe. The assertion moved from "it prints" to "it passes":
    while DEFERRED the row gated nothing, so rc 0 is now a statement about the CONFIGS."""
    proc = subprocess.run(
        [sys.executable, "tools/ci_gates/preflight_mint.py", "--audit-only"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_the_run_process_asserts_before_its_first_cuda_allocation():
    """The assertion must precede `init_trainer` in `build_run_collaborators`, read as statement
    ORDER out of the AST: an assertion after the trainer is built has already let the allocator
    be constructed under the wrong regime."""
    import ast as _ast

    source = (REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    builder = next(
        node for node in _ast.walk(_ast.parse(source))
        if isinstance(node, _ast.FunctionDef) and node.name == "build_run_collaborators"
    )
    assert_line = init_line = None
    for node in _ast.walk(builder):
        if isinstance(node, _ast.Call):
            func = node.func
            name = func.attr if isinstance(func, _ast.Attribute) else getattr(func, "id", "")
            if name == "_assert_allocator_posture" and assert_line is None:
                assert_line = node.lineno
            if name == "init_trainer" and init_line is None:
                init_line = node.lineno
    assert assert_line is not None, "build_run_collaborators does not assert the posture"
    assert init_line is not None, "init_trainer is no longer called here — re-derive this row"
    assert assert_line < init_line, (assert_line, init_line)


def test_the_eval_seam_threads_the_declared_token_and_the_child_is_what_judges_it():
    """`compose_run` THREADS the declared posture and the CHILD asserts it, because the child is
    where the device knowledge is: the seam passes the placeholder through, and the child raises
    on a `None` token whenever its own `worker_device` is cuda."""
    source = (REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    assert "_declared_allocator_posture(config.model_dump())" in source
    worker = (REPO_ROOT / "src" / "mantis" / "eval" / "worker.py").read_text(encoding="utf-8")
    assert "assert_posture_token(spec.allocator_posture" in worker


def test_the_declared_reader_passes_the_placeholder_but_refuses_garbage():
    from mantis.config.resolve.allocator_posture import declared_allocator_posture

    assert declared_allocator_posture({"allocator_posture": None}) is None
    assert declared_allocator_posture({"allocator_posture": "default"}) == "default"
    with pytest.raises(MissingAllocatorPostureError):
        declared_allocator_posture({"allocator_posture": "expandable"})
    with pytest.raises(MissingAllocatorPostureError):
        declared_allocator_posture({})

def test_the_device_token_is_spelled_once_and_the_root_asks_a_predicate():
    """`mantis.run` may not hardcode a device string. The first cut compared
    `worker_device.split(":")[0] == "cuda"` inside `compose_run` and tripped the device authority
    test; the token now lives once beside the posture and the root asks a predicate."""
    from mantis.config.resolve.allocator_posture import governs_device

    assert governs_device("cuda") and governs_device("cuda:1")
    assert not governs_device("cpu") and not governs_device("mps")
    root = (REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    assert "_posture_governs_device(config.eval.worker_device)" in root
