#!/usr/bin/env python3
"""Download minimal asset sets into repo-local assets/ directories.

Downloads:
- Kubric HDRI Haven (random subset)
- Kubric KuBasic (small rigid assets)
- PartNet Mobility (requires SAPIEN token)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, Iterable, List
from urllib.parse import urlparse, quote
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = Path(os.environ.get("SAPIEN_ASSETS_ROOT", REPO_ROOT / "assets"))
HDROUT = ASSETS_ROOT / "hdri"
RIGIDOUT = ASSETS_ROOT / "rigid_assets"
PARTNETOUT = ASSETS_ROOT / "partnet_mobility" / "dataset"

HDRI_MANIFEST = "https://storage.googleapis.com/kubric-public/assets/HDRI_haven/HDRI_haven.json"
KUBASIC_MANIFEST = "https://storage.googleapis.com/kubric-public/assets/KuBasic/KuBasic.json"
GSO_MANIFEST = "https://storage.googleapis.com/kubric-public/assets/GSO/GSO.json"

DEFAULT_PARTNET_IDS = ["1000", "1001", "1002", "1004", "1005", "1006", "1007"]
DEFAULT_PARTNET_ZIP_URL = "https://sapien.ucsd.edu/api/download/partnet-mobility-v0.zip"

# Token from https://sapien.ucsd.edu/
DEFAULT_PARTNET_TOKEN = os.environ.get("SAPIEN_TOKEN", "")

def gs_to_https(uri: str) -> str:
    if uri.startswith("gs://"):
        return f"https://storage.googleapis.com/{uri[5:]}"
    return uri


def load_json_anywhere(path_or_url: str) -> Dict[str, Any]:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https", "gs"):
        with urlopen(gs_to_https(path_or_url)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return json.loads(Path(path_or_url).read_text(encoding="utf-8"))


def resolve_base_dir(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https", "gs"):
        base_path = str(Path(parsed.path).parent)
        return f"{parsed.scheme}://{parsed.netloc}{base_path}"
    return str(Path(path_or_url).expanduser().resolve().parent)


def resolve_with_base(value: str, base_dir: str) -> str:
    value = value.replace("{asset_dir}", base_dir)
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https", "gs"):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if base_dir.startswith("http") or base_dir.startswith("gs://"):
        return base_dir.rstrip("/") + "/" + value.lstrip("/")
    return str(Path(base_dir) / path)


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return parsed._replace(path=quote(parsed.path)).geturl()
    return url


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urlopen(_safe_url(url)) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def has_any_files(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(path.is_file() for path in root.rglob("*"))


def touch_tree(root: Path) -> None:
    if not root.exists():
        return
    now = time.time()
    for path in (root,):
        try:
            os.utime(path, (now, now))
        except FileNotFoundError:
            return
    for path in root.rglob("*"):
        try:
            os.utime(path, (now, now))
        except FileNotFoundError:
            continue


def fetch_and_extract_tar(asset_path: str, asset_id: str, out_dir: Path, delete_tar: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_path = gs_to_https(asset_path)
    parsed = urlparse(asset_path)
    extract_dir = out_dir / asset_id
    needs_extract = not has_any_files(extract_dir)

    if parsed.scheme in ("http", "https"):
        tar_dest = out_dir / (Path(parsed.path).name or f"{asset_id}.tar.gz")
        if needs_extract and not tar_dest.exists():
            print(f"Downloading {asset_id} -> {tar_dest.name}")
            download_file(asset_path, tar_dest)
    else:
        tar_dest = Path(asset_path).expanduser()

    if not needs_extract:
        if delete_tar and tar_dest.exists():
            tar_dest.unlink()
        return extract_dir

    if extract_dir.exists():
        print(f"Re-extracting {asset_id}: found directory without files at {extract_dir}")

    if not tar_dest.exists():
        raise FileNotFoundError(f"Asset tar not found: {tar_dest}")

    with tarfile.open(tar_dest, "r:gz") as tar:
        names = tar.getnames()
        if any(name.startswith(f"{asset_id}/") for name in names):
            tar.extractall(out_dir)
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_dir)

    # netscratch cleanup uses file mtimes, so refresh extracted content to "now".
    touch_tree(extract_dir)

    if delete_tar and tar_dest.exists():
        tar_dest.unlink()
    return extract_dir


def pick_assets(manifest: Dict[str, Any], count: int) -> List[str]:
    asset_ids = sorted(manifest.get("assets", {}).keys())
    if count <= 0 or count >= len(asset_ids):
        return asset_ids
    rng = random.Random(42)
    return rng.sample(asset_ids, count)


def download_from_manifest(manifest_url: str, out_dir: Path, count: int, delete_tar: bool) -> None:
    manifest = load_json_anywhere(manifest_url)
    assets = manifest.get("assets", {})
    if not assets:
        raise RuntimeError(f"No assets found in {manifest_url}")
    base_dir = resolve_base_dir(manifest_url)
    data_dir_val = manifest.get("data_dir", "") or manifest.get("base_path", "")
    data_dir = resolve_with_base(data_dir_val or "", base_dir)

    for asset_id in pick_assets(manifest, count):
        entry = assets[asset_id]
        asset_path_val = entry.get("path", "")
        if asset_path_val == "":
            asset_path_val = f"{asset_id}.tar.gz"
        asset_path = resolve_with_base(asset_path_val, data_dir)
        fetch_and_extract_tar(asset_path, asset_id, out_dir, delete_tar)

def download_partnet_zip(partnet_dir: Path, url: str, token: str) -> None:
    partnet_dir.mkdir(parents=True, exist_ok=True)
    full_url = f"{url}?token={token}"
    zip_dest = partnet_dir.parent / "partnet-mobility-v0.zip"
    if not zip_dest.exists():
        print(f"Downloading PartNet zip -> {zip_dest.name}")
        download_file(full_url, zip_dest)
    extract_root = partnet_dir.parent
    with zipfile.ZipFile(zip_dest, "r") as zf:
        names = zf.namelist()
        to_extract = []
        for name in names:
            if name.endswith("/"):
                (extract_root / name).mkdir(parents=True, exist_ok=True)
            else:
                dest = extract_root / name
                if not dest.exists():
                    to_extract.append(name)
        if to_extract:
            print(f"Extracting PartNet zip ({len(to_extract)} missing files, {len(names) - len(to_extract)} already present)...")
            for name in to_extract:
                zf.extract(name, extract_root)
        else:
            print("PartNet zip already fully extracted, skipping.")


def download_partnet_models(partnet_dir: Path, token: str, model_ids: Iterable[str]) -> None:
    partnet_dir.mkdir(parents=True, exist_ok=True)
    for mid in model_ids:
        url = f"https://sapien.ucsd.edu/api/download/compressed/{mid}.zip?token={token}"
        zip_dest = partnet_dir / f"{mid}.zip"
        if not zip_dest.exists():
            print(f"Downloading PartNet {mid} -> {zip_dest.name}")
            download_file(url, zip_dest)
        extract_dir = partnet_dir / mid
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_dest, "r") as zf:
                zf.extractall(extract_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download minimal assets for SAPIEN pipeline")
    parser.add_argument("--assets-root", type=str, default=str(ASSETS_ROOT))
    parser.add_argument("--hdri-count", type=int, default=0)
    parser.add_argument("--rigid-source", choices=["gso", "kubasic"], default="gso",
                        help="Rigid asset source to download.")
    parser.add_argument("--kubasic-count", type=int, default=0,
                        help="0 means download all KuBasic assets.")
    parser.add_argument("--gso-count", type=int, default=0,
                        help="Number of GSO assets to download (0 = all).")
    parser.add_argument("--partnet-ids", type=str, default=",".join(DEFAULT_PARTNET_IDS))
    parser.add_argument("--partnet-zip-url", type=str, default=DEFAULT_PARTNET_ZIP_URL,
                        help="Full dataset zip URL (token appended automatically).")
    parser.add_argument("--sapien-token", type=str, default=DEFAULT_PARTNET_TOKEN)
    parser.add_argument("--partnet-mode", choices=["zip", "models"], default="zip",
                        help="Download full zip or specific model zips.")
    parser.add_argument("--keep-tar", action="store_true",
                        help="Keep .tar.gz archives after extract (default: delete).")
    parser.add_argument("--skip-hdri", action="store_true")
    parser.add_argument("--skip-rigid", action="store_true")
    parser.add_argument("--skip-partnet", action="store_true")
    args = parser.parse_args()

    assets_root = Path(args.assets_root).expanduser()
    hdri_dir = assets_root / "hdri"
    rigid_dir = assets_root / "gso_assets"
    partnet_dir = assets_root / "partnet_mobility" / "dataset"

    delete_tar = not args.keep_tar

    if not args.skip_hdri:
        print("Downloading HDRI Haven...")
        download_from_manifest(HDRI_MANIFEST, hdri_dir, args.hdri_count, delete_tar)

    if not args.skip_rigid:
        if args.rigid_source == "gso":
            print("Downloading GSO (rigid) assets...")
            count = args.gso_count
            if count == 0:
                count = 10_000_000
            download_from_manifest(GSO_MANIFEST, rigid_dir / "gso", count, delete_tar)
        else:
            print("Downloading KuBasic (rigid) assets...")
            count = args.kubasic_count
            if count == 0:
                count = 10_000
            download_from_manifest(KUBASIC_MANIFEST, rigid_dir / "kubasic", count, delete_tar)

    if not args.skip_partnet:
        token = args.sapien_token or os.environ.get("SAPIEN_TOKEN")
        if not token:
            print("PartNet download skipped (missing SAPIEN_TOKEN or --sapien-token).")
        else:
            if args.partnet_mode == "zip":
                print("Downloading PartNet full zip...")
                download_partnet_zip(partnet_dir, args.partnet_zip_url, token)
            else:
                ids = [i.strip() for i in args.partnet_ids.split(",") if i.strip()]
                print(f"Downloading PartNet models: {ids}")
                download_partnet_models(partnet_dir, token, ids)

    print("Done.")


if __name__ == "__main__":
    main()
