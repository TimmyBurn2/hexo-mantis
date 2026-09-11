"""Which resume bundles the mirror has receipted (R349(b)); under `train` because the
coordinator publishes the reading and cannot import diagnostics without a package cycle."""
from __future__ import annotations

import re
from pathlib import Path

from mantis.train.bundle import BundleManifest, complete_bundles, manifest_path_for
from mantis.util.mirror_receipts import MirrorReceiptError, verify_receipt

#: `checkpoints.checkpoint_filename`'s grammar, `{run_id}_{step:08d}_{sha8}.ckpt`; the sha8 is
#: `content_sha8` of the payload, which is what makes a bare checkpoint verifiable off-box.
CHECKPOINT_NAME_RE = re.compile(r"^(?P<run_id>.+)_(?P<step>\d{8})_(?P<sha8>[0-9a-f]{8})\.ckpt$")


def bundle_member_paths(manifest: BundleManifest, directory: Path) -> list[Path]:
    """Every file a receipt must cover for one bundle: its members and the manifest itself."""
    paths = [directory / member.name for member in manifest.members()]
    paths.append(manifest_path_for(directory / manifest.checkpoint.name))
    return paths


def unreceipted_members(manifest: BundleManifest, directory: Path) -> dict[str, str]:
    """`{member name: reason}` for every file of the bundle whose receipt is absent or wrong."""
    missing: dict[str, str] = {}
    for path in bundle_member_paths(manifest, directory):
        try:
            verify_receipt(path)
        except MirrorReceiptError as exc:
            missing[path.name] = str(exc)
    return missing


def unreceipted_bundle_steps(checkpoint_dir: str | Path) -> list[int]:
    """Steps of the complete bundles not fully receipted, oldest first (two is the warning)."""
    base = Path(checkpoint_dir)
    return [manifest.step for manifest in complete_bundles(base)
            if unreceipted_members(manifest, base)]


def stamped_checkpoints(directory: str | Path) -> list[Path]:
    """Every v2-named checkpoint under `directory`, oldest step first (a clean stop writes no bundle)."""
    base = Path(directory)
    if not base.is_dir():
        return []
    found = [(int(m.group("step")), path) for path in base.glob("*.ckpt")
             if (m := CHECKPOINT_NAME_RE.match(path.name))]
    return [path for _step, path in sorted(found, key=lambda item: (item[0], item[1].name))]


__all__ = ["CHECKPOINT_NAME_RE", "bundle_member_paths", "stamped_checkpoints",
           "unreceipted_bundle_steps", "unreceipted_members"]
