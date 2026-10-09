#!/usr/bin/env python3
"""Download and extract Infinigen-Articulated categories, then strip their inertials.

Source: https://huggingface.co/datasets/princeton-vl/infinigen-articulated

    python tools/fetch_infinigen.py --categories refrigerator,dishwasher,oven \
        --dest $SAPIEN_ASSETS_ROOT/infinigen_articulated

Categories that already exist at the destination are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanitize_urdf_inertials import sanitize  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/princeton-vl/infinigen-articulated/resolve/main/urdf"
LISTING_URL = "https://huggingface.co/api/datasets/princeton-vl/infinigen-articulated/tree/main/urdf"
DEFAULT_CATEGORIES = "refrigerator,dishwasher,oven,cabinet,drawer,lamp"
DEFAULT_DEST = os.environ.get("SAPIEN_ASSETS_ROOT", "assets") + "/infinigen_articulated"


def available_categories() -> list[str]:
    with urllib.request.urlopen(LISTING_URL, timeout=60) as r:
        files = json.load(r)
    return sorted(f["path"].split("/")[-1].removesuffix(".tar.gz")
                  for f in files if f["path"].endswith(".tar.gz"))


def fetch_category(cat: str, dest: Path, max_per_category: int) -> None:
    cat_dir = dest / cat
    if cat_dir.exists() and any(cat_dir.iterdir()):
        print(f"[{cat}] already present, skipping ({cat_dir})")
        return

    tarball = dest / f"{cat}.tar.gz"
    if not tarball.exists():
        url = f"{BASE_URL}/{cat}.tar.gz"
        print(f"[{cat}] downloading {url}")
        tmp = tarball.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        tmp.rename(tarball)
        print(f"[{cat}] downloaded {tarball.stat().st_size / 1e9:.2f} GB")

    print(f"[{cat}] extracting...")
    staging = dest / f".extract_{cat}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(staging, filter="data")
    except (EOFError, tarfile.TarError):
        # Truncated/corrupt download: drop the tarball so a rerun re-downloads.
        shutil.rmtree(staging, ignore_errors=True)
        tarball.unlink(missing_ok=True)
        raise RuntimeError(f"corrupt tarball for '{cat}' (deleted; rerun to re-download)")
    # Archive layout is ./urdf/<cat>/<id>/... -> move <id> dirs under dest/<cat>/
    inner = staging / "urdf" / cat
    if not inner.exists():
        candidates = list(staging.rglob(f"{cat}"))
        inner = candidates[0] if candidates else staging
    instance_dirs = sorted(d for d in inner.iterdir() if d.is_dir())
    if max_per_category > 0:
        instance_dirs = instance_dirs[:max_per_category]
    cat_dir.mkdir(parents=True, exist_ok=True)
    for d in instance_dirs:
        shutil.move(str(d), str(cat_dir / d.name))
    shutil.rmtree(staging)

    changed = total = 0
    for urdf in cat_dir.rglob("*.urdf"):
        total += 1
        changed += sanitize(urdf)
    print(f"[{cat}] {len(instance_dirs)} instances, sanitized {changed}/{total} URDFs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", default=DEFAULT_CATEGORIES,
                    help="Comma-separated category names (see --list)")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--max-per-category", type=int, default=0,
                    help="Keep only the first N instances per category (0 = all)")
    ap.add_argument("--list", action="store_true", help="List available categories and exit")
    args = ap.parse_args()

    if args.list:
        print("\n".join(available_categories()))
        return

    have = available_categories()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in cats if c not in have]
    if unknown:
        raise SystemExit(f"Unknown categories {unknown}. Available: {', '.join(have)}")

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for cat in cats:
        try:
            fetch_category(cat, dest, args.max_per_category)
        except Exception as e:
            print(f"[{cat}] FAILED: {e}")
            failed.append(cat)
    print(f"Done. Assets under {dest}")
    if failed:
        raise SystemExit(f"Failed categories (rerun to retry): {failed}")


if __name__ == "__main__":
    main()
