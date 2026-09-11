"""Which resume bundles the mirror has receipted (R349(b)); under `train` because the
coordinator publishes the reading and cannot import diagnostics without a package cycle."""
from __future__ import annotations

from pathlib import Path

from mantis.train.bundle import BundleManifest, complete_bundles, manifest_path_for
from mantis.util.mirror_receipts import MirrorReceiptError, verify_receipt


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


__all__ = ["bundle_member_paths", "unreceipted_bundle_steps", "unreceipted_members"]
