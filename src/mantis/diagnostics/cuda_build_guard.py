"""Refuse a CPU-only torch build, and PROVE the CUDA path with a real matmul (R328(e)).

WHY THIS EXISTS, AND WHY IT IS IN-TREE RATHER THAN ON THE BOX. `pyproject.toml` pins torch to
the PyTorch **CPU** wheel index (`[tool.uv.sources] torch = { index = "pytorch-cpu" }`), which
is correct for the parity regime that pin was written for and wrong for a GPU host. So every
`uv sync` on the box replaces a CUDA torch with `2.11.0+cpu`, and it is not a box accident that
a box script can own — it is the committed configuration doing what it says. A guard that lives
only on the box is re-lost with the box; this one rides in the repo.

WHY `torch.cuda.is_available()` IS NOT THE CHECK. It answers "did this build find a driver",
which a CPU wheel answers `False` and a broken CUDA install can answer `True`. The question that
matters before a run is whether CUDA **computes correctly**, and only an arithmetic result
answers it. The MINT-CLOSE session hit the downgrade and verified the repair with a real matmul
against a CPU reference for exactly this reason; that check is now mechanism instead of memory.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

#: Square side for the proof matmul. Large enough that a wrong kernel or a silently-CPU tensor
#: shows up in the residual, small enough to run in milliseconds on any card the project uses.
MATMUL_N = 1024

#: Max allowed |cuda - cpu| elementwise. fp32 matmul over 1024 accumulations on a TF32-capable
#: card diverges from an MKL CPU reference well inside this; a wrong RESULT does not.
MATMUL_ATOL = 1e-2

#: Deterministic input seed, so a residual is comparable between runs and hosts.
MATMUL_SEED = 328


class CudaBuildRefusal(RuntimeError):
    """The installed torch cannot be trusted to compute on a GPU. Carries the named reason."""


def torch_build() -> dict[str, object]:
    """Everything the refusal needs to name the build it refused, derived from torch itself."""
    return {
        "torch_version": torch.__version__,
        "cuda_toolkit": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        # AUDIT-1 F-28/A11. `... else 0` published a device count nothing counted: on a
        # CPU-only host the field read `0` exactly as it would on a CUDA host that saw no
        # device, and `assert_cuda_build`'s third refusal then named the wrong condition
        # ("reports CUDA available with device_count 0") for a build that reports no CUDA at
        # all. `None` = the question was not asked, and it is asked whenever it can be.
        "device_count": (int(torch.cuda.device_count())
                         if torch.cuda.is_available() else None),
    }


def assert_cuda_build() -> dict[str, object]:
    """Refuse unless torch is a CUDA build with at least one visible device.

    Returns:
        The build dict from `torch_build()`, when every check passes.

    Raises:
        CudaBuildRefusal: the build carries no CUDA toolkit, reports no CUDA, or sees no device.
    """
    build = torch_build()
    if build["cuda_toolkit"] is None:
        raise CudaBuildRefusal(
            f"torch {build['torch_version']} is a CPU-ONLY build (torch.version.cuda is None). "
            "This is what `uv sync` installs here: pyproject.toml pins torch to the pytorch-cpu "
            "index, so the downgrade recurs on every sync and re-running the box's CUDA restore "
            "is required after each one."
        )
    if not build["cuda_available"]:
        raise CudaBuildRefusal(
            f"torch {build['torch_version']} carries CUDA {build['cuda_toolkit']} but "
            "torch.cuda.is_available() is False: no usable driver or no visible device."
        )
    # Reached only past the `cuda_available` refusal above, so the count WAS asked for and
    # `None` here would mean the two disagree — named as its own refusal rather than crashing
    # in `int(None)`.
    count = build["device_count"]
    if count is None:
        raise CudaBuildRefusal(
            f"torch {build['torch_version']} reports CUDA available but no device count was "
            "taken: `torch.cuda.is_available()` and `torch.cuda.device_count()` disagree "
            "about whether the question can be asked."
        )
    if int(count) < 1:  # type: ignore[arg-type]
        raise CudaBuildRefusal(
            f"torch {build['torch_version']} reports CUDA available with device_count 0: "
            "there is nothing to run on."
        )
    return build


def assert_cuda_matmul(n: int = MATMUL_N, atol: float = MATMUL_ATOL) -> float:
    """Multiply on CUDA, multiply the same inputs on CPU, and refuse if they disagree.

    Args:
        n: square side of the operands.
        atol: max tolerated elementwise deviation from the CPU reference.

    Returns:
        The measured max elementwise deviation.

    Raises:
        CudaBuildRefusal: the CUDA result deviates beyond `atol`, or the CUDA op itself raised.
    """
    generator = torch.Generator(device="cpu").manual_seed(MATMUL_SEED)
    left = torch.randn(n, n, generator=generator, dtype=torch.float32)
    right = torch.randn(n, n, generator=generator, dtype=torch.float32)
    reference = left @ right
    try:
        measured = (left.cuda() @ right.cuda()).cpu()
    except RuntimeError as exc:
        raise CudaBuildRefusal(f"the CUDA matmul raised rather than computing: {exc}") from exc
    residual = float((measured - reference).abs().max().item())
    if not residual <= atol:
        raise CudaBuildRefusal(
            f"the CUDA matmul disagrees with the CPU reference by {residual:.6g} > atol {atol}: "
            "this build reports a GPU and does not compute correctly on it."
        )
    return residual


def check(n: int = MATMUL_N, atol: float = MATMUL_ATOL) -> dict[str, object]:
    """Run both halves and return the report.

    Returns:
        A report dict carrying the build, the measured residual and `"verdict": "PASS"`.

    Raises:
        CudaBuildRefusal: from either half; the caller renders it as the named refusal.
    """
    build = assert_cuda_build()
    residual = assert_cuda_matmul(n=n, atol=atol)
    return {**build, "matmul_n": n, "matmul_atol": atol,
            "matmul_max_abs_dev": residual, "verdict": "PASS"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.diagnostics.cuda_build_guard",
        description="Refuse a CPU-only torch build; prove CUDA with a matmul against CPU.")
    parser.add_argument("--n", type=int, default=MATMUL_N,
                        help=f"square side of the proof matmul (default {MATMUL_N})")
    parser.add_argument("--atol", type=float, default=MATMUL_ATOL,
                        help=f"max elementwise deviation from CPU (default {MATMUL_ATOL})")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = check(n=args.n, atol=args.atol)
    except CudaBuildRefusal as exc:
        detail = json.dumps({**torch_build(), "verdict": "REFUSED", "reason": str(exc)})
        print(detail if args.json else f"CUDA-BUILD-GUARD REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report) if args.json else
          f"CUDA-BUILD-GUARD PASS: torch {report['torch_version']} "
          f"cuda {report['cuda_toolkit']} devices {report['device_count']} "
          f"matmul max|dev| {report['matmul_max_abs_dev']:.3g} <= {report['matmul_atol']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
