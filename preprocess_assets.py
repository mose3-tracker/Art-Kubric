#!/usr/bin/env python3
"""Preprocess assets for SAPIEN Kubric-like generation.

Scans local asset directories and writes a JSON index that can be consumed by
`render_scene.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS_ROOT = Path(os.environ.get("SAPIEN_ASSETS_ROOT", REPO_ROOT / "assets"))
DEFAULT_PARTNET_REL = os.path.join("partnet_mobility", "dataset")
DEFAULT_PARTNET_ROOT = DEFAULT_ASSETS_ROOT / DEFAULT_PARTNET_REL
DEFAULT_RIGID_ROOT = DEFAULT_ASSETS_ROOT / "rigid_assets" / "gso"
DEFAULT_HDRI_ROOT = DEFAULT_ASSETS_ROOT / "hdri"
DEFAULT_RIGID_EXTS = [".urdf", ".obj", ".glb", ".gltf", ".ply"]
DEFAULT_HDRI_EXTS = [".hdr", ".exr"]


def _as_posix_relative(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel.as_posix())


def _obj_bounding_box_diagonal(obj_path: Path) -> float:
    """Read OBJ vertices and return bounding box diagonal length."""
    verts = []
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except (OSError, ValueError):
        return 0.0
    if len(verts) < 3:
        return 0.0
    arr = np.array(verts, dtype=np.float32)
    extents = arr.max(axis=0) - arr.min(axis=0)
    return float(np.linalg.norm(extents))


def _score_urdf(urdf_path: Path) -> float:
    """Score a URDF by max(child_link_diagonal * joint_range).

    Higher scores mean the asset has large movable parts.
    """
    try:
        tree = ET.parse(urdf_path)
    except (ET.ParseError, OSError):
        return 0.1

    root = tree.getroot()
    urdf_dir = urdf_path.parent

    # Build link name -> visual mesh filenames mapping
    link_meshes: Dict[str, List[str]] = {}
    for link_el in root.findall("link"):
        link_name = link_el.attrib.get("name", "")
        meshes = []
        for mesh_el in link_el.findall(".//visual//mesh"):
            fn = mesh_el.attrib.get("filename")
            if fn:
                meshes.append(fn)
        if meshes:
            link_meshes[link_name] = meshes

    best_score = 0.0
    for joint_el in root.findall("joint"):
        jtype = joint_el.attrib.get("type", "fixed")
        if jtype == "fixed":
            continue

        limit_el = joint_el.find("limit")
        if limit_el is not None:
            try:
                lo = float(limit_el.attrib.get("lower", "0"))
                hi = float(limit_el.attrib.get("upper", "0"))
            except ValueError:
                continue
        elif jtype == "continuous":
            lo, hi = 0.0, 2 * np.pi
        else:
            continue

        joint_range = abs(hi - lo)
        if joint_range < 0.01:
            continue

        # Down-weight continuous joints (fans, wheels).
        if jtype == "continuous":
            joint_range *= 0.4

        child_el = joint_el.find("child")
        if child_el is None:
            continue
        child_name = child_el.attrib.get("link", "")
        mesh_files = link_meshes.get(child_name, [])

        max_diag = 0.0
        for fn in mesh_files:
            mesh_path = (urdf_dir / fn).resolve()
            if mesh_path.exists() and mesh_path.suffix.lower() == ".obj":
                diag = _obj_bounding_box_diagonal(mesh_path)
                max_diag = max(max_diag, diag)

        score = max_diag * joint_range
        best_score = max(best_score, score)

    return best_score if best_score > 0.0 else 0.1


def _scan_partnet(partnet_dir: Path, compute_scores: bool = True) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    if not partnet_dir.exists():
        return assets
    for sub in sorted(partnet_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Accept mobility.urdf (PartNet), model.urdf (Articraft), or a single *.urdf.
        urdf = sub / "mobility.urdf"
        if not urdf.exists():
            urdf = sub / "model.urdf"
        if not urdf.exists():
            candidates = sorted(sub.glob("*.urdf"))
            if len(candidates) == 1:
                urdf = candidates[0]
        if urdf.exists():
            entry: Dict[str, Any] = {
                "id": sub.name,
                # Absolute path; the loader changes into the asset directory.
                "urdf": str(urdf.resolve()),
            }
            if compute_scores:
                entry["movability_score"] = _score_urdf(urdf)
            assets.append(entry)
    return assets


def _scan_flat_urdf(directory: Path, compute_scores: bool = True) -> List[Dict[str, Any]]:
    """Scan a directory for URDF files (flat layout, not PartNet subdirectory structure)."""
    assets: List[Dict[str, Any]] = []
    if not directory.exists():
        return assets
    for urdf in sorted(directory.glob("*.urdf")):
        if urdf.is_file():
            entry: Dict[str, Any] = {
                "id": urdf.stem,
                "urdf": str(urdf.resolve()),
            }
            if compute_scores:
                entry["movability_score"] = _score_urdf(urdf)
            assets.append(entry)
    return assets


def _dir_urdf(directory: Path) -> Path | None:
    """The asset URDF inside a directory, or None.

    Accepts mobility.urdf (PartNet), model.urdf (Articraft), or a lone *.urdf
    (Infinigen names it after the category).
    """
    for name in ("mobility.urdf", "model.urdf"):
        urdf = directory / name
        if urdf.exists():
            return urdf
    candidates = sorted(directory.glob("*.urdf"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _scan_urdf_tree(root: Path, max_depth: int = 3) -> List[Dict[str, Any]]:
    """Recursively scan for asset directories containing one URDF each.

    Handles both PartNet-style one-level layouts (<id>/mobility.urdf) and
    nested layouts like Infinigen-Articulated (<category>/<id>/<category>.urdf).
    Asset id = posix path of the asset dir relative to root.
    """
    assets: List[Dict[str, Any]] = []
    if not root.exists():
        return assets

    def walk(directory: Path, depth: int) -> None:
        urdf = _dir_urdf(directory)
        if urdf is not None:
            assets.append({
                "id": _as_posix_relative(directory, root),
                # Absolute path; the loader changes into the asset directory.
                "urdf": str(urdf.resolve()),
            })
            return
        if depth >= max_depth:
            return
        for sub in sorted(directory.iterdir()):
            if sub.is_dir():
                walk(sub, depth + 1)

    walk(root, 0)
    return assets


def _visual_geometry_aabbs(link_el: ET.Element, urdf_dir: Path):
    """Yield (min_xyz, max_xyz) AABBs of a link's visual geometries in link frame."""
    for visual_el in link_el.findall("visual"):
        origin_el = visual_el.find("origin")
        offset = np.zeros(3)
        if origin_el is not None:
            try:
                offset = np.array([float(v) for v in origin_el.attrib.get("xyz", "0 0 0").split()])
            except ValueError:
                pass
        geom_el = visual_el.find("geometry")
        if geom_el is None:
            continue
        mesh_el = geom_el.find("mesh")
        box_el = geom_el.find("box")
        cyl_el = geom_el.find("cylinder")
        sph_el = geom_el.find("sphere")
        half = None
        if mesh_el is not None:
            fn = mesh_el.attrib.get("filename")
            if not fn:
                continue
            mesh_path = (urdf_dir / fn).resolve()
            if not mesh_path.exists() or mesh_path.suffix.lower() != ".obj":
                continue
            verts = []
            try:
                with mesh_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("v "):
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (OSError, ValueError):
                continue
            if len(verts) < 3:
                continue
            arr = np.array(verts, dtype=np.float64)
            scale = np.ones(3)
            if mesh_el.attrib.get("scale"):
                try:
                    scale = np.array([float(v) for v in mesh_el.attrib["scale"].split()])
                except ValueError:
                    pass
            yield arr.min(axis=0) * scale + offset, arr.max(axis=0) * scale + offset
            continue
        if box_el is not None:
            try:
                half = np.array([float(v) for v in box_el.attrib.get("size", "0 0 0").split()]) / 2.0
            except ValueError:
                continue
        elif cyl_el is not None:
            try:
                r = float(cyl_el.attrib.get("radius", "0"))
                length = float(cyl_el.attrib.get("length", "0"))
                half = np.array([r, r, length / 2.0])
            except ValueError:
                continue
        elif sph_el is not None:
            try:
                r = float(sph_el.attrib.get("radius", "0"))
                half = np.array([r, r, r])
            except ValueError:
                continue
        if half is not None:
            yield offset - half, offset + half


def _urdf_visual_aabb_diag(urdf_path: Path) -> float:
    """Whole-asset visual AABB diagonal (meters) at zero joint pose.

    Approximation: accumulates translation-only frame offsets (joint and
    visual <origin xyz>, ignoring rpy). Good enough to drive target-diagonal
    scale normalization; generation clamps extreme scale factors anyway.
    """
    try:
        tree = ET.parse(urdf_path)
    except (ET.ParseError, OSError):
        return 0.0
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    # Translation-only link offsets via the joint tree (children of joints).
    link_offset: Dict[str, np.ndarray] = {}
    joints = root.findall("joint")
    pending = list(joints)
    for _ in range(len(joints) + 1):
        remaining = []
        for joint_el in pending:
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.attrib.get("link", "")
            child = child_el.attrib.get("link", "")
            if parent not in link_offset and any(
                    j.find("child") is not None and j.find("child").attrib.get("link") == parent
                    for j in joints):
                remaining.append(joint_el)  # parent offset not resolved yet
                continue
            origin_el = joint_el.find("origin")
            xyz = np.zeros(3)
            if origin_el is not None:
                try:
                    xyz = np.array([float(v) for v in origin_el.attrib.get("xyz", "0 0 0").split()])
                except ValueError:
                    pass
            link_offset[child] = link_offset.get(parent, np.zeros(3)) + xyz
        if not remaining:
            break
        pending = remaining

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for link_el in root.findall("link"):
        offset = link_offset.get(link_el.attrib.get("name", ""), np.zeros(3))
        for vmin, vmax in _visual_geometry_aabbs(link_el, urdf_dir):
            lo = np.minimum(lo, vmin + offset)
            hi = np.maximum(hi, vmax + offset)
    if not np.all(np.isfinite(lo)):
        return 0.0
    return float(np.linalg.norm(hi - lo))


def _scan_rigid_assets(root: Path, exts: Iterable[str]) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    if not root.exists():
        return assets

    exts = {e.lower() for e in exts}
    seen_dirs = set()

    for urdf in root.rglob("*.urdf"):
        if not urdf.is_file():
            continue
        asset_id = _as_posix_relative(urdf.with_suffix(""), root)
        assets.append({
            "id": asset_id,
            # Absolute path; the loader changes into the asset directory.
            "path": str(urdf.resolve()),
            "kind": "urdf",
        })
        seen_dirs.add(urdf.parent.resolve())

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        if path.suffix.lower() == ".urdf":
            continue
        if path.parent.resolve() in seen_dirs:
            continue
        asset_id = _as_posix_relative(path.with_suffix(""), root)
        assets.append({
            "id": asset_id,
            "path": str(path),
            "kind": "mesh",
        })

    return assets


def _scan_hdri(root: Path, exts: Iterable[str]) -> List[str]:
    if root is None or not root.exists():
        return []
    exts = {e.lower() for e in exts}
    hdris = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            # Skip the dimmed environment-map cache files written during generation.
            if "_env" in path.stem:
                continue
            hdris.append(str(path))
    return hdris


def _cap_assets(entries: List[Dict[str, Any]], max_assets: int) -> List[Dict[str, Any]]:
    """Deterministic evenly-strided subset, preserving category spread of sorted ids."""
    if max_assets <= 0 or len(entries) <= max_assets:
        return entries
    idx = np.linspace(0, len(entries) - 1, max_assets).round().astype(int)
    return [entries[i] for i in sorted(set(idx.tolist()))]


def _build_from_config(config_path: Path, only: List[str], skip: List[str]) -> Dict[str, Any]:
    """Build the merged multi-source index described by a YAML config."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("--config requires PyYAML (pip install pyyaml)") from e

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources_meta: Dict[str, Any] = {}
    articulated: List[Dict[str, Any]] = []
    rigid: List[Dict[str, Any]] = []

    for src in cfg.get("sources", []):
        name = src["name"]
        if (only and name not in only) or name in skip:
            print(f"Skipping source: {name}")
            continue
        kind = src.get("kind", "articulated")
        src_dir = Path(os.path.expandvars(src["dir"])).expanduser()
        # ids listed in the config's `exclude` file never enter the index
        scale_policy = str(src.get("scale_policy", "none"))
        target_diag = src.get("target_diag")
        weight = float(src.get("weight", 1.0))
        if not src_dir.exists():
            print(f"Warning: source '{name}' dir not found, skipping: {src_dir}")
            continue

        if kind == "rigid":
            exts = src.get("exts", DEFAULT_RIGID_EXTS)
            entries = _scan_rigid_assets(src_dir, exts)
        else:
            layout = src.get("layout", "urdf_tree")
            entries = _scan_flat_urdf(src_dir, compute_scores=False) if layout == "flat" \
                else _scan_urdf_tree(src_dir)

        entries = _cap_assets(entries, int(src.get("max_assets", 0)))
        for entry in entries:
            entry["source"] = name
            if name != "partnet" and not entry["id"].startswith(f"{name}/"):
                entry["id"] = f"{name}/{entry['id']}"
            if kind == "articulated":
                if scale_policy == "fixed":
                    entry["scale"] = float(src.get("scale", 1.0))
                elif scale_policy == "normalize":
                    entry["scale_policy"] = "normalize"
                    if target_diag:
                        entry["target_diag"] = [float(target_diag[0]), float(target_diag[1])]

        if kind == "articulated" and scale_policy == "normalize":
            print(f"Computing native AABB diagonals for '{name}' ({len(entries)} assets)...")
            diags = []
            for entry in entries:
                diag = _urdf_visual_aabb_diag(Path(entry["urdf"]))
                if diag > 1e-6:
                    entry["native_diag"] = diag
                    diags.append(diag)
            if diags:
                print(f"  native_diag min/median/max: {min(diags):.2f} / "
                      f"{sorted(diags)[len(diags)//2]:.2f} / {max(diags):.2f} m "
                      f"({len(diags)}/{len(entries)} measurable)")

        (rigid if kind == "rigid" else articulated).extend(entries)
        sources_meta[name] = {"kind": kind, "weight": weight, "count": len(entries),
                              "scale_policy": scale_policy}
        print(f"  Source '{name}' ({kind}): {len(entries)} assets")

    excluded = set()
    if cfg.get("exclude"):
        ex_path = (config_path.parent / os.path.expandvars(str(cfg["exclude"]))).expanduser()
        if ex_path.exists():
            blob = json.loads(ex_path.read_text())
            excluded = {str(i) for i in (blob.get("excluded_ids", blob)
                                         if isinstance(blob, dict) else blob)}
            before = len(articulated) + len(rigid)
            articulated = [e for e in articulated if str(e.get("id")) not in excluded]
            rigid = [e for e in rigid if str(e.get("id")) not in excluded]
            print(f"Excluded {before - len(articulated) - len(rigid)} assets "
                  f"listed in {ex_path.name}")
        else:
            print(f"WARNING: exclude file not found: {ex_path}")

    hdri_dir = cfg.get("hdri", {}).get("dir")
    data = {
        "meta": {
            "schema_version": 4,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": str(config_path.resolve()),
            "sources": sources_meta,
            "source_mix": cfg.get("source_mix", {}),
            "excluded_ids": sorted(excluded),
        },
        "articulated": articulated,
        "rigid": rigid,
        "hdri": _scan_hdri(Path(os.path.expandvars(hdri_dir)).expanduser(), DEFAULT_HDRI_EXTS) if hdri_dir else [],
    }
    if cfg.get("output"):
        data["meta"]["default_output"] = os.path.expandvars(str(cfg["output"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess asset lists for SAPIEN generation.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config listing the asset sources (assets_config.yaml).")
    parser.add_argument("--only-source", type=str, action="append", default=[],
                        help="With --config: only build these sources (repeatable).")
    parser.add_argument("--skip-source", type=str, action="append", default=[],
                        help="With --config: skip these sources (repeatable).")
    parser.add_argument("--assets-root", type=str, default=str(DEFAULT_ASSETS_ROOT),
                        help="Root directory containing assets (partnet/rigid/hdri).")
    parser.add_argument("--partnet-dir", type=str, default=None,
                        help="PartNet Mobility dataset root (contains model_id/mobility.urdf).")
    parser.add_argument("--rigid-dir", type=str, default=None,
                        help="Root directory of rigid assets (URDF or mesh files).")
    parser.add_argument("--rigid-exts", type=str, default=",".join(DEFAULT_RIGID_EXTS),
                        help="Comma-separated rigid asset extensions to scan.")
    parser.add_argument("--hdri-dir", type=str, default=None,
                        help="Directory of HDRI files (.hdr/.exr) to index.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path.")
    parser.add_argument("--extra-articulated-dir", type=str, action="append", default=[],
                        help="Extra directory of URDF articulated assets (can be specified multiple times).")
    parser.add_argument("--extra-articulated-scale", type=float, action="append", default=[],
                        help="Uniform scale for assets in the matching --extra-articulated-dir.")
    parser.add_argument("--no-scoring", action="store_true",
                        help="Skip movability scoring (faster but no weighted sampling support).")
    args = parser.parse_args()

    assets_root = Path(args.assets_root).expanduser()

    if args.config:
        data = _build_from_config(Path(args.config).expanduser(), args.only_source, args.skip_source)
    else:
        partnet_dir = Path(args.partnet_dir).expanduser() if args.partnet_dir else assets_root / DEFAULT_PARTNET_REL
        rigid_dir = Path(args.rigid_dir).expanduser() if args.rigid_dir else assets_root / "gso_assets"
        hdri_dir = Path(args.hdri_dir).expanduser() if args.hdri_dir else assets_root / "hdri"

        rigid_exts = [e.strip() for e in args.rigid_exts.split(",") if e.strip()]

        data = {
            "meta": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "partnet_dir": str(partnet_dir),
                "rigid_dir": str(rigid_dir) if rigid_dir else None,
                "hdri_dir": str(hdri_dir) if hdri_dir else None,
            },
            "articulated": _scan_partnet(partnet_dir),
            "rigid": _scan_rigid_assets(rigid_dir, rigid_exts) if rigid_dir else [],
            "hdri": _scan_hdri(hdri_dir, DEFAULT_HDRI_EXTS) if hdri_dir else [],
        }

        # Scan extra articulated asset directories
        for i, extra_dir in enumerate(args.extra_articulated_dir):
            extra_path = Path(extra_dir).expanduser()
            if not extra_path.exists():
                print(f"Warning: extra articulated dir not found: {extra_path}")
                continue
            # Try PartNet-style subdirectory layout first, then flat URDF layout
            extra_assets = _scan_partnet(extra_path)
            extra_assets.extend(_scan_flat_urdf(extra_path))
            if args.extra_articulated_scale:
                scale = args.extra_articulated_scale[min(i, len(args.extra_articulated_scale) - 1)]
                if scale != 1.0:
                    for entry in extra_assets:
                        entry["scale"] = scale
            data["articulated"].extend(extra_assets)
            print(f"  Extra articulated from {extra_path}: {len(extra_assets)}")

    # Score articulated assets for weighted sampling
    if not args.no_scoring and data["articulated"]:
        print(f"Scoring {len(data['articulated'])} articulated assets...")
        for entry in data["articulated"]:
            urdf = entry.get("urdf")
            if urdf:
                entry["movability_score"] = _score_urdf(Path(urdf))
            else:
                entry["movability_score"] = 0.1
        scores = [e["movability_score"] for e in data["articulated"]]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}], "
              f"median: {sorted(scores)[len(scores)//2]:.3f}")

    default_output = data.get("meta", {}).pop("default_output", None)
    out_path = (Path(args.output).expanduser() if args.output
                else Path(default_output) if default_output
                else assets_root / "assets_index.json")
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote asset index to {out_path}")
    print(f"  articulated: {len(data['articulated'])}")
    print(f"  rigid: {len(data['rigid'])}")
    print(f"  hdri: {len(data['hdri'])}")


if __name__ == "__main__":
    main()
