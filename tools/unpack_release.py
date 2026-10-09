#!/usr/bin/env python3
"""Restore a packed scene to the full rendered layout and verify it.

    python tools/unpack_release.py --src PACKED/0001 --dst OUT/0001
    python tools/unpack_release.py --src PACKED --dst OUT --all --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import zstandard as zstd


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def unshuffle_bytes(buf: bytes, dtype: np.dtype, shape) -> np.ndarray:
    n = np.dtype(dtype).itemsize
    count = int(np.prod(shape)) if shape else 0
    if n == 1:
        return np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)
    flat = np.frombuffer(buf, dtype=np.uint8, count=count * n).reshape(n, count)
    return flat.T.copy().view(dtype).reshape(shape)


def from_jsonable(v):
    """Restore byte values encoded in the manifest."""
    if isinstance(v, dict):
        if set(v) == {"__bytes__"}:
            return bytes.fromhex(v["__bytes__"])
        return {k: from_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [from_jsonable(x) for x in v]
    return v


def load_packed(entry: dict, base: Path, dctx: zstd.ZstdDecompressor) -> np.ndarray:
    buf = dctx.decompress((base / entry["packed"]).read_bytes())
    return unshuffle_bytes(buf, np.dtype(entry["dtype"]), tuple(entry["shape"]))


def quantize_depth(depth_f32: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The packer's inverse for depths/: the exact expression the exporter used."""
    norm = (depth_f32 - lo) / (hi - lo + 1e-8)
    return (norm * 65534 + 1).clip(0, 65535).astype(np.uint16)


def unpack_scene(src: Path, dst: Path, verify: bool = True) -> list[str]:
    man = json.loads((src / "pack_manifest.json").read_text())
    seq = man["scene"]
    dctx = zstd.ZstdDecompressor()
    dst.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    for name, e in man["files"].items():
        if name == "coords_depth":
            arrays["coords_depth"] = load_packed(e, src, dctx)
            continue
        if isinstance(e, dict) and "kind" not in e:          # a directory group
            (dst / name).mkdir(exist_ok=True)
            for fn, fe in e.items():
                if fe["kind"] == "verbatim":
                    shutil.copy2(src / name / fn, dst / name / fn)
                else:
                    a = load_packed(fe, src / name, dctx)
                    np.save(dst / name / fn, a)
                    arrays[f"{name}/{fn}"] = a
                if verify and fe.get("sha256") and sha256(dst / name / fn) != fe["sha256"]:
                    problems.append(f"{name}/{fn}")
            continue
        if e["kind"] == "verbatim":
            shutil.copy2(src / name, dst / name)
        else:
            a = load_packed(e, src, dctx)
            np.save(dst / name, a)
            arrays[name] = a
        if verify and e.get("sha256") and sha256(dst / name) != e["sha256"]:
            problems.append(name)

    # --- rebuild depths/
    dinfo = man["derived"].get("depths")
    ann_info = man["derived"][f"{seq}.npy"]
    lo, hi = from_jsonable(ann_info["metadata"])["depth_range"]
    depth_f32 = arrays.get("depth_f32.npy")
    if depth_f32 is None:
        depth_f32 = np.load(dst / "depth_f32.npy", mmap_mode="r")
    if dinfo:
        (dst / "depths").mkdir(exist_ok=True)
        for i in range(dinfo["count"]):
            q = quantize_depth(np.asarray(depth_f32[i]), lo, hi)[..., None]
            p = dst / "depths" / f"{i:03d}.npy"
            np.save(p, q)
            if verify and sha256(p) != dinfo["sha256"][f"{i:03d}.npy"]:
                problems.append(f"depths/{i:03d}.npy")

    # --- rebuild {seq}.npy
    def get(name):
        return arrays.get(name) if name in arrays else np.load(dst / name)
    ann = {
        "coords": get(f"{seq}_trajs_2d.npy"),
        "coords_depth": arrays["coords_depth"],
        "visibility": get(f"{seq}_visibility.npy"),
        "depth": np.stack([quantize_depth(np.asarray(depth_f32[i]), lo, hi)[..., None]
                           for i in range(len(depth_f32))]),
        "segmentations": get("se3/rigid_map_actor.npy")[..., None],
        "coords_depth_abs": get(f"{seq}_coords_depth_abs.npy"),
        "coords_3d": get(f"{seq}_trajs_3d.npy"),
        "metadata": from_jsonable(ann_info["metadata"]),
        "camera": {k: (np.asarray(v) if isinstance(v, list) else from_jsonable(v))
                   for k, v in ann_info["camera"].items()},
    }
    ann = {k: ann[k] for k in ann_info["keys"] if k in ann}
    np.save(dst / f"{seq}.npy", ann, allow_pickle=True)
    if verify:
        # Check each rebuilt array against the digest recorded by the packer.
        want = ann_info.get("field_sha256", {})
        if not want:
            problems.append(f"{seq}.npy:no field digests in manifest")
        for k, want_sha in want.items():
            if k not in ann:
                problems.append(f"{seq}.npy:{k} missing")
                continue
            got = hashlib.sha256(np.ascontiguousarray(ann[k]).tobytes()).hexdigest()
            if got != want_sha:
                problems.append(f"{seq}.npy:{k}")
        if list(ann.keys()) != ann_info["keys"]:
            problems.append(f"{seq}.npy:key order")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--all", action="store_true", help="src holds many packed scenes")
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir():
        ap.error(f"source directory does not exist: {src}")
    if a.all:
        scenes = sorted(p for p in src.iterdir()
                        if p.is_dir() and (p / "pack_manifest.json").is_file())
        if not scenes:
            ap.error(f"no packed scenes found in {src}; expected "
                     "<src>/<scene>/pack_manifest.json. Extract the shard tar files "
                     "into --src first, or omit --all for a single scene directory.")
    else:
        if not (src / "pack_manifest.json").is_file():
            ap.error(f"missing {src / 'pack_manifest.json'}; "
                     "use --all if --src contains multiple scene directories")
        scenes = [src]
    bad = 0
    for s in scenes:
        probs = unpack_scene(s, dst / s.name if a.all else dst, a.verify)
        if probs:
            bad += 1
            print(f"{s.name}: {len(probs)} mismatched -> {probs[:5]}", flush=True)
        else:
            print(f"{s.name}: ok", flush=True)
    print(f"{len(scenes)} scenes, {bad} with mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
