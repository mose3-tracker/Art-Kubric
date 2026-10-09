"""Simulate and render one Art-Kubric scene in SAPIEN.

Articulated and rigid objects are dropped into an HDRI-lit dome and simulated.
Outputs RGB, depth, segmentation and per-link poses for the tracking step.
"""

from __future__ import annotations

import argparse
import json
import itertools
import os
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from collections import Counter
import imageio.v2 as imageio
from PIL import ImageColor

from cotracker_export import save_cotracker_generation, numpy_json_default, segmentation_labels
import sapien


# =============================================================================
# Configs
# =============================================================================

@dataclass
class BackgroundConfig:
    hdri_path: Optional[str] = None
    hdri_dir: Optional[str] = None
    ambient_light: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    floor_size: float = 40.0
    ground_tex_size: int = 2048
    dome_radius: float = 40.0
    dome_lat_segments: int = 32
    dome_lon_segments: int = 64
    dome_u_offset: float = 0.0
    dome_use_tonemap: bool = True
    enable_sky_tracking: bool = True
    sky_dome_radius: float = 800.0
    allow_plain_floor: bool = True
    env_light_scale: float = 1.0
    # Opaque HDRI dome around the scene. Disable to light the scene from the env map directly.
    use_hdri_dome: bool = True


@dataclass
class SpawnBox:
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]


@dataclass
class SpawnConfig:
    static_box: SpawnBox = field(default_factory=lambda: SpawnBox(min=(-4.0, -4.0, 0.0), max=(4.0, 4.0, 8.0)))
    dynamic_box: SpawnBox = field(default_factory=lambda: SpawnBox(min=(-3.0, -3.0, 4.0), max=(3.0, 3.0, 8.0)))
    min_gap: float = 1.5
    max_trials: int = 80
    max_yaw: float = np.pi


@dataclass
class ObjectMixConfig:
    min_num_static: int = 10
    max_num_static: int = 20
    min_num_dynamic: int = 1
    max_num_dynamic: int = 3
    articulated_fraction: float = 0.4


@dataclass
class VelocityConfig:
    min_velocity: Tuple[float, float, float] = (-1.0, -1.0, 0.0)
    max_velocity: Tuple[float, float, float] = (1.0, 1.0, 0.0)
    max_speed: float = 4.0
    center_speed_min: float = 2.5
    center_speed_max: float = 4.0
    z_speed_min: float = -0.5
    z_speed_max: float = 1.0


@dataclass
class JointMotionConfig:
    qpos_range_fraction: float = 0.8
    active_drive_prob: float = 0.7
    drive_stiffness: float = 5.0
    drive_damping: float = 0.5
    passive_damping: float = 0.15
    qvel_prob: float = 1.0
    max_qvel: float = 3.0
    salient_stiffness_multiplier: float = 1.0
    salient_friction: float = 0.04
    salient_keywords: Tuple[str, ...] = ("door", "drawer", "lid", "screen", "seat", "blade", "window", "handle")


@dataclass
class CameraConfig:
    width: int = 256
    height: int = 256
    fovy: float = 1.0  # vertical FOV in radians (1.0 rad = 57.3 deg)
    near: float = 0.1
    far: float = 1000.0
    mode: str = "linear"  # linear | fixed_random | linear_movement | linear_movement_linear_lookat | orbit
    look_at: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    max_movement: float = 4.0
    inner_radius: float = 4.0
    outer_radius: float = 6.0
    z_offset: float = 0.1
    lookat_inner_radius: float = 0.5
    lookat_outer_radius: float = 2.5
    fixed_inner_radius: float = 3.5
    fixed_outer_radius: float = 5.5
    orbit_radius: float = 3.5
    orbit_height: float = 2.0
    orbit_rotation: float = np.pi * 0.2


@dataclass
class SimConfig:
    timestep: float = 1 / 240
    settle_steps: int = 300
    steps_per_frame: Optional[int] = None
    num_frames: int = 24
    fps: int = 12
    ray_tracing_samples: int = 64


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    video_name: str = "partnet_drop.mp4"
    annotation_name: str = "rigid_box_drop_annotations.json"


# =============================================================================
# Asset Index
# =============================================================================

@dataclass
class RigidAsset:
    asset_id: str
    path: str
    kind: str = "mesh"  # mesh | urdf
    source: str = "gso"


@dataclass
class ArticulatedAsset:
    asset_id: str
    urdf_path: Optional[str] = None
    uri: Optional[str] = None
    movability_score: Optional[float] = None
    scale: float = 1.0
    source: str = "partnet"
    # scale_policy "normalize": per-spawn loader.scale = U(target_diag) / native_diag
    scale_policy: str = "none"  # none | fixed | normalize
    native_diag: Optional[float] = None
    target_diag: Optional[Tuple[float, float]] = None


@dataclass
class AssetsIndex:
    rigid: List[RigidAsset] = field(default_factory=list)
    articulated: List[ArticulatedAsset] = field(default_factory=list)
    hdri: List[str] = field(default_factory=list)
    # per-source metadata from the asset index
    source_weights: Dict[str, float] = field(default_factory=dict)
    source_mix: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpawnedObject:
    entity: object
    kind: str  # rigid | articulated
    asset_id: str
    is_dynamic: bool
    source: str = ""
    # spawn footprint, so a replacement can reclaim the slot in `occupied`
    spawn_xy: Optional[Tuple[float, float]] = None


def load_assets_index(path: Optional[str]) -> AssetsIndex:
    if not path:
        raise ValueError("assets_index path is required (run preprocess_assets.py first).")
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))

    rigid = []
    for entry in data.get("rigid", []):
        asset_id = entry.get("id") or entry.get("asset_id") or Path(entry.get("path", "unknown")).stem
        path_val = entry.get("path")
        if not path_val:
            continue
        rigid.append(RigidAsset(asset_id=asset_id, path=path_val, kind=entry.get("kind", "mesh"),
                                source=entry.get("source", "gso")))

    articulated = []
    for entry in data.get("articulated", []):
        asset_id = entry.get("id") or entry.get("asset_id") or Path(entry.get("urdf", "unknown")).stem
        target_diag = entry.get("target_diag")
        articulated.append(ArticulatedAsset(
            asset_id=asset_id,
            urdf_path=entry.get("urdf") or entry.get("path"),
            uri=entry.get("uri"),
            movability_score=entry.get("movability_score"),
            scale=float(entry.get("scale", 1.0)),
            source=entry.get("source", "partnet"),
            scale_policy=entry.get("scale_policy",
                                   "fixed" if "scale" in entry else "none"),
            native_diag=entry.get("native_diag"),
            target_diag=tuple(target_diag) if target_diag else None,
        ))

    hdri = [str(p) for p in data.get("hdri", []) if p]

    meta = data.get("meta", {})
    source_weights = {name: float(info.get("weight", 1.0))
                      for name, info in meta.get("sources", {}).items()
                      if info.get("kind", "articulated") == "articulated"}
    return AssetsIndex(rigid=rigid, articulated=articulated, hdri=hdri,
                       source_weights=source_weights,
                       source_mix=meta.get("source_mix", {}))


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


def score_articulated_asset(asset: ArticulatedAsset) -> float:
    """Score an articulated asset by max(child_link_diagonal * joint_range).

    Higher scores mean larger movable parts.
    """
    urdf_path = asset.urdf_path
    if not urdf_path:
        return 0.1
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        return 0.1

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

        # Get joint range
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

        # Down-weight continuous joints (fans, wheels) so doors, drawers and lids dominate sampling.
        if jtype == "continuous":
            joint_range *= 0.4

        # Get child link's mesh diagonal
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


# =============================================================================
# HDRI/Background Utilities (from Kubric)
# =============================================================================


def create_ground_projection_from_hdri(hdri_path: Path, output_size: int = 1024,
                                       u_offset: float = 0.0) -> Path:
    """Create a CIRCULAR ground projection texture from HDRI for floor mapping.

    The UV mapping is aligned with the dome hemisphere so that the horizon
    texels match seamlessly: u = atan2(y,x)/(2*pi) + u_offset, matching
    the dome's u = phi/(2*pi) + u_offset.
    """
    import cv2

    hdri = cv2.imread(str(hdri_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if hdri is None:
        raise RuntimeError(f"Failed to load HDRI file: {hdri_path}")
    hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB).astype(np.float32)

    hdri_h, hdri_w = hdri.shape[:2]

    py, px = np.mgrid[0:output_size, 0:output_size]
    x = (px / (output_size - 1) - 0.5) * 2
    y = (py / (output_size - 1) - 0.5) * 2

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    v = 1.0 - 0.5 * np.clip(r, 0, 1)
    u = (theta / (2 * np.pi) + u_offset) % 1.0

    hx = (u * (hdri_w - 1)).astype(np.int32)
    hy = (v * (hdri_h - 1)).astype(np.int32)
    hx = np.clip(hx, 0, hdri_w - 1)
    hy = np.clip(hy, 0, hdri_h - 1)

    ground_tex = hdri[hy, hx]
    ground_tex = ground_tex / (1.0 + ground_tex)
    ground_tex = np.power(np.clip(ground_tex, 0, 1), 1.0 / 2.2)
    ground_tex_uint8 = np.clip(ground_tex * 255, 0, 255).astype(np.uint8)

    alpha = np.where(r <= 1.0, 255, 0).astype(np.uint8)
    ground_tex_rgba = np.dstack([ground_tex_uint8, alpha])

    output_path = hdri_path.parent / f"{hdri_path.stem}_ground_projection.png"
    success = cv2.imwrite(str(output_path), cv2.cvtColor(ground_tex_rgba, cv2.COLOR_RGBA2BGRA))
    if not success:
        raise RuntimeError(f"Failed to save ground projection texture: {output_path}")
    return output_path


def create_tonemapped_hdri_texture(hdri_path: Path) -> Path:
    import cv2

    output_path = hdri_path.parent / f"{hdri_path.stem}_tonemap.png"
    if output_path.exists():
        return output_path

    hdri = cv2.imread(str(hdri_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if hdri is None:
        raise RuntimeError(f"Failed to load HDRI file: {hdri_path}")
    hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB).astype(np.float32)

    ldr = hdri / (1.0 + hdri)
    ldr = np.power(np.clip(ldr, 0, 1), 1.0 / 2.2)
    ldr_uint8 = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    success = cv2.imwrite(str(output_path), cv2.cvtColor(ldr_uint8, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f"Failed to save tone-mapped HDRI texture: {output_path}")
    return output_path


ENV_LEVELS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)


def create_dimmed_env_map(hdri_path: Path, scale: float) -> Path:
    """Write (and cache) a darkened copy of an HDRI, for use as the IBL environment
    map only.

    The visible dome and floor keep the original full-brightness texture, so this
    lowers the ambient lighting floor that washes shadows out without darkening the
    background the camera sees.
    """
    import cv2

    output_path = hdri_path.parent / f"{hdri_path.stem}_env{scale:.2f}.hdr"
    if output_path.exists():
        return output_path

    hdri = cv2.imread(str(hdri_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if hdri is None:
        raise RuntimeError(f"Failed to load HDRI file: {hdri_path}")

    # Write to a temp file and rename, so concurrent jobs never read a partial file.
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{os.getpid()}.hdr")
    success = cv2.imwrite(str(tmp_path), hdri.astype(np.float32) * float(scale))
    if not success:
        raise RuntimeError(f"Failed to save dimmed environment map: {tmp_path}")
    os.replace(tmp_path, output_path)
    return output_path


def create_disk_obj(output_path: Path, radius: float = 1.0, segments: int = 64):
    """Create an OBJ file for a disk with radial UV mapping."""
    lines = ["# Circular disk mesh", "# Generated for SAPIEN floor"]
    lines.append("v 0 0 0")
    lines.append("vt 0.5 0.5")
    lines.append("vn 0 0 1")
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        lines.append(f"v {x} {y} 0")
        u = 0.5 + 0.5 * np.cos(angle)
        v = 0.5 + 0.5 * np.sin(angle)
        lines.append(f"vt {u} {v}")
        lines.append("vn 0 0 1")
    for i in range(segments):
        v1 = 1
        v2 = i + 2
        v3 = (i + 1) % segments + 2
        lines.append(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")
    output_path.write_text("\n".join(lines))
    return output_path


FLOOR_COLLISION_HALF_Z = 2.0


def _add_floor_collision(builder, half_xy: float):
    """Add a thick downward floor collision whose top surface is at local z=0."""
    builder.add_box_collision(
        pose=sapien.Pose([0.0, 0.0, -FLOOR_COLLISION_HALF_Z]),
        half_size=[half_xy, half_xy, FLOOR_COLLISION_HALF_Z],
    )


def create_circular_floor(scene, hdri_path: Path, floor_radius: float = 40.0,
                           ground_tex_size: int = 2048, segments: int = 64,
                           u_offset: float = 0.0):
    if hdri_path is None or not hdri_path.exists():
        raise RuntimeError(f"HDRI path is required for floor texture: {hdri_path}")

    ground_tex_path = create_ground_projection_from_hdri(hdri_path, output_size=ground_tex_size,
                                                         u_offset=u_offset)
    obj_path = hdri_path.parent / "floor_disk.obj"
    create_disk_obj(obj_path, radius=floor_radius, segments=segments)

    mat = sapien.render.RenderMaterial()
    mat.roughness = 0.9
    mat.metallic = 0.0
    mat.specular = 0.05
    mat.base_color_texture = sapien.render.RenderTexture2D(str(ground_tex_path))

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(obj_path), material=mat)
    _add_floor_collision(builder, floor_radius * 1.1)

    floor = builder.build_static(name="floor")
    floor.set_pose(sapien.Pose(p=[0, 0, -0.01]))
    return floor


def create_dome_obj(output_path: Path, radius: float = 40.0, lat_segments: int = 32, lon_segments: int = 64,
                    u_offset: float = 0.0, overshoot_deg: float = 5.0):
    lines = ["# Hemisphere dome mesh", "# Generated for SAPIEN background dome"]
    verts = []
    uvs = []
    norms = []

    theta_max = np.pi / 2.0 + np.radians(overshoot_deg)
    for i in range(lat_segments + 1):
        theta = (i / lat_segments) * theta_max  # 0 (zenith) -> slightly past horizon
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for j in range(lon_segments + 1):
            phi = (j / lon_segments) * (2.0 * np.pi)
            x = radius * sin_t * np.cos(phi)
            y = radius * sin_t * np.sin(phi)
            z = radius * cos_t
            verts.append((x, y, z))
            u = (phi / (2.0 * np.pi) + u_offset) % 1.0
            v = theta / np.pi  # 0..0.5 for hemisphere
            uvs.append((u, v))
            n = np.array([-x, -y, -z], dtype=np.float32)
            n = n / (np.linalg.norm(n) + 1e-8)
            norms.append((n[0], n[1], n[2]))

    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for vt in uvs:
        lines.append(f"vt {vt[0]} {vt[1]}")
    for vn in norms:
        lines.append(f"vn {vn[0]} {vn[1]} {vn[2]}")

    def vidx(i, j):
        return i * (lon_segments + 1) + j + 1

    for i in range(lat_segments):
        for j in range(lon_segments):
            v0 = vidx(i, j)
            v1 = vidx(i, j + 1)
            v2 = vidx(i + 1, j)
            v3 = vidx(i + 1, j + 1)
            # Reverse winding to face inward
            lines.append(f"f {v0}/{v0}/{v0} {v2}/{v2}/{v2} {v1}/{v1}/{v1}")
            lines.append(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")

    output_path.write_text("\n".join(lines))
    return output_path


def create_hdri_dome(scene, hdri_path: Path, radius: float = 40.0,
                     lat_segments: int = 32, lon_segments: int = 64,
                     u_offset: float = 0.0, use_tonemap: bool = True):
    if hdri_path is None or not hdri_path.exists():
        raise RuntimeError(f"HDRI path is required for dome texture: {hdri_path}")

    obj_path = hdri_path.parent / "hdri_dome.obj"
    create_dome_obj(obj_path, radius=radius, lat_segments=lat_segments, lon_segments=lon_segments,
                    u_offset=u_offset)

    mat = sapien.render.RenderMaterial()
    mat.roughness = 1.0
    mat.metallic = 0.0
    mat.specular = 0.0
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    tex_path = create_tonemapped_hdri_texture(hdri_path) if use_tonemap else hdri_path
    mat.base_color_texture = sapien.render.RenderTexture2D(str(tex_path))

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(obj_path), material=mat)
    dome = builder.build_static(name="hdri_dome")
    dome.set_pose(sapien.Pose(p=[0, 0, 0]))
    return dome


def create_sky_dome_for_tracking(scene, radius: float = 800.0):
    mat = sapien.render.RenderMaterial()
    mat.base_color = [1.0, 1.0, 1.0, 0.005]
    mat.roughness = 1.0
    mat.metallic = 0.0
    mat.specular = 0.0
    mat.transmission = 0.0

    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=mat)
    dome = builder.build_static(name="sky_dome")
    dome.set_pose(sapien.Pose(p=[0, 0, 0]))
    return dome


def gs_to_https(uri: str) -> str:
    if uri.startswith("gs://"):
        return f"https://storage.googleapis.com/{uri[5:]}"
    return uri


def load_json_anywhere(path_or_url: str) -> dict:
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


def fetch_and_extract_tar(asset_path: str, asset_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    asset_path = gs_to_https(asset_path)
    parsed = urlparse(asset_path)
    if parsed.scheme in ("http", "https"):
        tar_dest = cache_dir / (Path(parsed.path).name or f"{asset_id}.tar.gz")
        if not tar_dest.exists():
            print(f"Downloading asset tar {asset_path} -> {tar_dest}")
            try:
                with urlopen(asset_path) as resp, open(tar_dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
            except URLError as e:
                raise RuntimeError(f"Failed to download {asset_path}: {e}") from e
    else:
        tar_dest = Path(asset_path).expanduser()
    if not tar_dest.exists():
        raise FileNotFoundError(f"Asset tar not found: {tar_dest}")

    extract_dir = cache_dir / asset_id
    if not extract_dir.exists():
        with tarfile.open(tar_dest, "r:gz") as tar:
            names = tar.getnames()
            if any(name.startswith(f"{asset_id}/") for name in names):
                tar.extractall(cache_dir)
            else:
                extract_dir.mkdir(parents=True, exist_ok=True)
                tar.extractall(extract_dir)
    return extract_dir


def choose_hdri_path(rng: np.random.Generator, config: BackgroundConfig, assets_index: Optional[AssetsIndex]) -> Optional[Path]:
    if config.hdri_path:
        cand = Path(config.hdri_path).expanduser()
        if cand.exists():
            return cand
        print(f"HDRI path not found: {cand}")

    candidates: List[Path] = []
    if config.hdri_dir:
        hdri_dir = Path(config.hdri_dir).expanduser()
        if hdri_dir.exists():
            # "_env" filter: never pick the dimmed-IBL cache files as sources.
            candidates.extend(sorted(p for p in
                                     list(hdri_dir.glob("**/*.hdr")) + list(hdri_dir.glob("**/*.exr"))
                                     if "_env" not in p.stem))

    if assets_index and assets_index.hdri:
        for p in assets_index.hdri:
            cand = Path(p).expanduser()
            if cand.exists():
                candidates.append(cand)

    if candidates:
        idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]

    return None


def create_background(scene, rng: np.random.Generator, config: BackgroundConfig,
                      assets_index: Optional[AssetsIndex] = None):
    hdri_path = choose_hdri_path(rng, config, assets_index)

    if hdri_path and hdri_path.exists():
        env_map_path = hdri_path
        if config.env_light_scale != 1.0:
            try:
                env_map_path = create_dimmed_env_map(hdri_path, config.env_light_scale)
            except Exception as e:
                print(f"Warning: could not dim environment map ({e}); using original")
        scene.set_environment_map(str(env_map_path))
        scene.set_ambient_light(list(config.ambient_light))
    else:
        raise RuntimeError("HDRI not found. Run preprocess_assets.py with --hdri-dir or provide --hdri-path.")

    hdri_dome = None
    if config.use_hdri_dome:
        try:
            hdri_dome = create_hdri_dome(
                scene,
                hdri_path=hdri_path,
                radius=config.dome_radius,
                lat_segments=config.dome_lat_segments,
                lon_segments=config.dome_lon_segments,
                u_offset=config.dome_u_offset,
                use_tonemap=config.dome_use_tonemap,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create HDRI dome: {e}") from e

    try:
        floor = create_circular_floor(
            scene,
            hdri_path=hdri_path,
            floor_radius=config.floor_size,
            ground_tex_size=config.ground_tex_size,
            u_offset=config.dome_u_offset,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create HDRI floor: {e}") from e

    sky_dome = None
    if config.enable_sky_tracking and config.use_hdri_dome:
        # Skipped with the dome, since it also blocks shadow rays.
        sky_dome = create_sky_dome_for_tracking(scene, radius=config.sky_dome_radius)

    domes = [d for d in [hdri_dome, sky_dome] if d is not None]
    return hdri_path, floor, domes


# =============================================================================
# Utilities
# =============================================================================


def _area_light_pose(dist: float, tilt: float, az: float):
    """Pose for an area light at spherical (dist, tilt, az), aimed at the origin.

    RenderParallelogramLightComponent emits along its local +x axis (the same
    convention sapien's scene wrapper uses for spot lights), so we rotate +x onto
    the direction pointing from the light back at the scene centre.
    """
    pos = np.array([
        dist * np.sin(tilt) * np.cos(az),
        dist * np.sin(tilt) * np.sin(az),
        dist * np.cos(tilt),
    ])
    direction = (-pos / np.linalg.norm(pos)).tolist()
    return sapien.Pose(pos.tolist(), sapien.math.shortest_rotation([1.0, 0.0, 0.0], direction))


def setup_lighting(scene, rng: np.random.Generator,
                   rich_lighting: bool = False, scene_seed: int = 0,
                   area_light_gain: float = 1.0):
    # Draw the same two values as the default path, so --rich-lighting does not shift the main rng stream.
    r0, r1 = rng.random(), rng.random()
    if not rich_lighting:
        main_light_color = [0.9 + r0 * 0.1, 0.9, 0.9 - r1 * 0.1]
        scene.add_directional_light(
            direction=[1, 1, -1],
            color=main_light_color,
            shadow=True,
            shadow_scale=5.0,
            shadow_map_size=2048,
        )
        scene.add_directional_light(
            direction=[-1, -1, -0.5],
            color=[0.4, 0.4, 0.4],
            shadow=False,
        )
        return {"mode": "directional", "main_color": main_light_color}

    # Randomize the key light per scene from an independent rng.
    lr = np.random.default_rng(int(scene_seed) + 1_000_003)

    az = float(lr.uniform(0.0, 2.0 * np.pi))
    tilt = float(lr.uniform(0.55, 1.00))   # 31-57 deg
    dist = float(lr.uniform(7.0, 11.0))
    hw = float(lr.uniform(0.06, 0.35))
    hh = hw * float(lr.uniform(0.8, 1.25))
    # Scale emission by d^2/area so `exposure` sets brightness regardless of light size and distance.
    exposure = float(lr.uniform(1.8, 3.1)) * area_light_gain
    base = exposure * dist * dist / (4.0 * hw * hh)
    warm = float(lr.uniform(0.85, 1.12))
    cool = float(lr.uniform(0.85, 1.12))
    scene.add_area_light_for_ray_tracing(
        _area_light_pose(dist, tilt, az),
        [base * warm, base * 0.97, base * cool],
        hw, hh,
    )

    second_light = lr.random() < 0.25
    if second_light:
        az2 = az + float(lr.uniform(2.0, 4.3))
        tilt2 = float(lr.uniform(0.40, 0.90))
        dist2 = float(lr.uniform(7.0, 11.0))
        hw2 = float(lr.uniform(0.25, 0.75))
        hh2 = hw2 * float(lr.uniform(0.7, 1.3))
        b2 = float(lr.uniform(0.12, 0.28)) * area_light_gain * dist2 * dist2 / (4.0 * hw2 * hh2)
        scene.add_area_light_for_ray_tracing(
            _area_light_pose(dist2, tilt2, az2), [b2, b2, b2], hw2, hh2,
        )

    return {"mode": "rich", "az": az, "tilt": tilt, "dist": dist, "hw": hw, "hh": hh,
            "exposure": exposure, "warm": warm, "cool": cool, "second_light": second_light}


def _parse_kv_floats(spec: str, flag_name: str) -> Dict[str, float]:
    """Parse 'name=0.3,name2=0.7' CLI specs into a dict."""
    out: Dict[str, float] = {}
    for part in spec.split(","):
        name, eq, val = part.partition("=")
        name = name.strip()
        if not eq or not name:
            raise SystemExit(f"{flag_name}: malformed entry '{part}' (want name=value,...)")
        try:
            out[name] = float(val)
        except ValueError:
            raise SystemExit(f"{flag_name}: non-numeric value in '{part}'")
    return out


def random_yaw_quat(rng: np.random.Generator, max_yaw: float):
    yaw = float(rng.uniform(-max_yaw, max_yaw))
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def random_3d_quat(rng: np.random.Generator):
    """Uniformly random 3D rotation (Shoemake's method)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    return [
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ]


def random_upright_quat(rng: np.random.Generator, max_tilt: float = np.pi / 6):
    """Random rotation with limited tilt from upright (yaw free, pitch/roll bounded)."""
    yaw = float(rng.uniform(-np.pi, np.pi))
    pitch = float(rng.uniform(-max_tilt, max_tilt))
    roll = float(rng.uniform(-max_tilt, max_tilt))
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ]


def sample_spawn_pose(rng: np.random.Generator, occupied: List[Tuple[float, float]],
                      spawn_box: SpawnBox, min_gap: float, max_trials: int, max_yaw: float):
    for _ in range(max_trials):
        x = float(rng.uniform(spawn_box.min[0], spawn_box.max[0]))
        y = float(rng.uniform(spawn_box.min[1], spawn_box.max[1]))
        if all(np.hypot(x - ox, y - oy) >= min_gap for ox, oy in occupied):
            z = float(rng.uniform(spawn_box.min[2], spawn_box.max[2]))
            return np.array([x, y, z], dtype=np.float32), random_yaw_quat(rng, max_yaw)
    return None, None


def pose_to_quaternion(pose):
    R = pose[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def get_actor_id(entity):
    for attr in ("get_id", "get_actor_id", "get_per_scene_id"):
        fn = getattr(entity, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
    ent = getattr(entity, "entity", None) or (entity.get_entity() if hasattr(entity, "get_entity") else None)
    if ent:
        for attr in ("get_per_scene_id", "per_scene_id", "get_id", "id"):
            val = getattr(ent, attr, None)
            if callable(val):
                try:
                    return int(val())
                except Exception:
                    continue
            elif val is not None:
                try:
                    return int(val)
                except Exception:
                    continue
    return None


def get_entities(obj):
    if isinstance(obj, (list, tuple)):
        ents = []
        for item in obj:
            ents.extend(get_entities(item))
        return ents
    if hasattr(obj, "get_links") and callable(obj.get_links):
        return obj.get_links()
    return [obj]


def apply_physical_material(obj, material):
    if material is None:
        return
    entities = get_entities(obj)
    for ent in entities:
        shapes = getattr(ent, "get_collision_shapes", None)
        if not callable(shapes):
            continue
        for shape in shapes():
            if hasattr(shape, "set_physical_material"):
                shape.set_physical_material(material)


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_urdf_base(urdf_path: Path) -> Path:
    cur = urdf_path.parent
    for _ in range(4):
        if (cur / "textured_objs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return urdf_path.parent


def _obj_has_enough_unique_vertices(obj_path: Path, min_unique: int = 4) -> bool:
    unique = set()
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    v = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
                unique.add(v)
                if len(unique) >= min_unique:
                    return True
    except OSError:
        return False
    return False


def _validate_articulated_collision_meshes(urdf_path: Path) -> Optional[str]:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as e:
        return f"invalid URDF xml: {e}"
    except OSError as e:
        return f"cannot read URDF: {e}"

    for mesh in root.findall(".//collision//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = (urdf_path.parent / filename).resolve()
        if not mesh_path.exists():
            return f"missing collision mesh: {filename}"
        if mesh_path.suffix.lower() == ".obj" and not _obj_has_enough_unique_vertices(mesh_path):
            return f"degenerate collision mesh (<4 unique vertices): {filename}"
    return None


def set_entity_pose(obj, pos: np.ndarray, quat: List[float]):
    pose = sapien.Pose(p=pos.tolist(), q=quat)
    if isinstance(obj, (list, tuple)):
        root_mat = pose.to_transformation_matrix()
        for ent in obj:
            if hasattr(ent, "get_pose"):
                cur = ent.get_pose().to_transformation_matrix()
                ent.set_pose(sapien.Pose(root_mat @ cur))
            elif hasattr(ent, "set_pose"):
                ent.set_pose(pose)
        return
    if hasattr(obj, "set_pose"):
        obj.set_pose(pose)
    elif hasattr(obj, "set_root_pose"):
        obj.set_root_pose(pose)


def set_entity_velocity(obj, linear: np.ndarray, angular: Optional[np.ndarray] = None):
    if angular is None:
        angular = np.zeros(3, dtype=np.float32)
    if isinstance(obj, (list, tuple)):
        for ent in obj:
            if hasattr(ent, "set_velocity"):
                ent.set_velocity(linear.tolist())
            if hasattr(ent, "set_angular_velocity"):
                ent.set_angular_velocity(angular.tolist())
            if hasattr(ent, "set_root_velocity"):
                ent.set_root_velocity(linear.tolist(), angular.tolist())
        return
    if hasattr(obj, "set_velocity"):
        obj.set_velocity(linear.tolist())
    if hasattr(obj, "set_angular_velocity"):
        obj.set_angular_velocity(angular.tolist())
    if hasattr(obj, "set_root_velocity"):
        obj.set_root_velocity(linear.tolist(), angular.tolist())


def set_entity_damping(obj, linear: float = 0.15, angular: float = 0.3):
    entities = obj if isinstance(obj, (list, tuple)) else [obj]
    for ent in entities:
        if hasattr(ent, "set_damping"):
            try:
                ent.set_damping(float(linear), float(angular))
            except Exception:
                pass
        if hasattr(ent, "set_linear_damping"):
            try:
                ent.set_linear_damping(float(linear))
            except Exception:
                pass
        if hasattr(ent, "set_angular_damping"):
            try:
                ent.set_angular_damping(float(angular))
            except Exception:
                pass


def set_entity_max_depenetration_velocity(obj, max_vel: float):
    """Cap PhysX depenetration ejection speed on all dynamic bodies of obj.

    PhysX defaults this to 1e32 (unbounded); overlapping spawns then eject
    bodies fast enough to tunnel through the floor in one substep.
    """
    for ent in get_entities(obj):
        comp = ent if hasattr(ent, "set_max_depenetration_velocity") else None
        if comp is None:
            finder = getattr(ent, "find_component_by_type", None)
            if callable(finder):
                try:
                    comp = finder(sapien.physx.PhysxRigidDynamicComponent)
                except Exception:
                    comp = None
        if comp is not None and hasattr(comp, "set_max_depenetration_velocity"):
            try:
                comp.set_max_depenetration_velocity(float(max_vel))
            except Exception:
                pass


def clamp_linear_speed(linear: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(linear))
    if speed <= max_speed or speed < 1e-8:
        return linear
    return linear * (max_speed / speed)


def sample_dynamic_initial_velocity(rng: np.random.Generator, pos: np.ndarray, cfg: VelocityConfig) -> np.ndarray:
    vel = rng.uniform(cfg.min_velocity, cfg.max_velocity).astype(np.float32)
    to_center = np.array([-pos[0], -pos[1]], dtype=np.float32)
    dist = float(np.linalg.norm(to_center))
    if dist > 1e-6:
        center_dir = to_center / dist
        center_mag = float(rng.uniform(cfg.center_speed_min, cfg.center_speed_max))
        vel[0] += center_dir[0] * center_mag
        vel[1] += center_dir[1] * center_mag
    vel[2] += float(rng.uniform(cfg.z_speed_min, cfg.z_speed_max))
    return clamp_linear_speed(vel, cfg.max_speed)


def _mesh_aabb_half_size(mesh_path: str) -> Optional[np.ndarray]:
    """Read an OBJ/mesh file and return AABB half-size, or None on failure."""
    path = Path(mesh_path)
    if path.suffix.lower() not in (".obj", ".ply", ".glb", ".gltf"):
        return None
    try:
        import trimesh
        m = trimesh.load(str(path), force="mesh", process=False)
        extents = m.bounding_box.extents
        return np.array(extents, dtype=np.float32) / 2.0
    except Exception:
        pass
    if path.suffix.lower() == ".obj":
        verts = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except OSError:
            return None
        if len(verts) < 3:
            return None
        arr = np.array(verts, dtype=np.float32)
        return (arr.max(axis=0) - arr.min(axis=0)) / 2.0
    return None


RIGID_TARGET_DIAG_MIN = 1.0
RIGID_TARGET_DIAG_MAX = 2.5

MAX_DEPENETRATION_VEL = 5.0

P_PURE_ART = 0.35
P_PURE_RIGID = 0.10
MIX_RIGID_LO = 0.05
MIX_RIGID_HI = 0.60


def _rigid_asset_diag(urdf_path: str) -> Optional[float]:
    """Original AABB diagonal (meters) of a GSO rigid asset.

    Prefers the ``kwargs.bounds`` AABB in the asset's ``data.json`` (Kubric
    FileBasedObject format); falls back to reading ``visual_geometry.obj``.
    """
    d = os.path.dirname(urdf_path)
    try:
        bounds = json.loads(Path(d, "data.json").read_text())["kwargs"]["bounds"]
        return float(np.linalg.norm(np.array(bounds[1], dtype=np.float64)
                                    - np.array(bounds[0], dtype=np.float64)))
    except Exception:
        pass
    diag = _obj_bounding_box_diagonal(Path(d) / "visual_geometry.obj")
    return diag if diag > 1e-6 else None


def build_rigid_from_asset(scene, asset: RigidAsset, rng: np.random.Generator, name: str):
    if asset.kind == "urdf":
        # Build GSO objects with ActorBuilder; the URDF loader leaves their mass at zero.
        gso_dir = Path(asset.path).parent
        visual = gso_dir / "visual_geometry.obj"
        collision = gso_dir / "collision_geometry.obj"
        if visual.exists() and collision.exists():
            scale = 1.0
            cur_diag = _rigid_asset_diag(asset.path)
            if cur_diag and cur_diag > 1e-6:
                target_diag = float(rng.uniform(RIGID_TARGET_DIAG_MIN, RIGID_TARGET_DIAG_MAX))
                scale = target_diag / cur_diag
            sv = [scale, scale, scale]
            with _chdir(gso_dir):
                builder = scene.create_actor_builder()
                builder.add_visual_from_file(str(visual), scale=sv)
                try:
                    builder.add_multiple_convex_collisions_from_file(str(collision), scale=sv, density=1000.0)
                except Exception:
                    builder.add_convex_collision_from_file(str(collision), scale=sv, density=1000.0)
                return builder.build(name=name)
        # Fallback for non-GSO urdf rigid assets: original URDF loader path.
        loader = scene.create_urdf_loader()
        loader.fix_root_link = False
        if hasattr(loader, "scale"):
            cur_diag = _rigid_asset_diag(asset.path)
            if cur_diag and cur_diag > 1e-6:
                target_diag = float(rng.uniform(RIGID_TARGET_DIAG_MIN, RIGID_TARGET_DIAG_MAX))
                loader.scale = target_diag / cur_diag
        if hasattr(loader, "set_density"):
            loader.set_density(1000.0)
        elif hasattr(loader, "density"):
            loader.density = 1000.0
        urdf_path = Path(asset.path)
        base_dir = _resolve_urdf_base(urdf_path)
        try:
            with _chdir(base_dir):
                obj = loader.load(str(urdf_path))
            if hasattr(obj, "set_name"):
                obj.set_name(name)
            return obj
        except Exception as e:
            if "multiple objects" in str(e).lower() and hasattr(loader, "load_multiple"):
                with _chdir(base_dir):
                    objs = loader.load_multiple(str(urdf_path))
                for idx, ent in enumerate(objs):
                    if hasattr(ent, "set_name"):
                        ent.set_name(f"{name}_{idx}")
                return objs
            raise

    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial()
    color = rng.uniform(0.2, 0.95, size=3).tolist() + [1.0]
    mat.base_color = color
    builder.add_visual_from_file(asset.path, material=mat)

    collision_added = False
    if hasattr(builder, "add_multiple_convex_collisions_from_file"):
        try:
            builder.add_multiple_convex_collisions_from_file(asset.path)
            collision_added = True
        except Exception:
            pass
    if not collision_added and hasattr(builder, "add_convex_collision_from_file"):
        try:
            builder.add_convex_collision_from_file(asset.path)
            collision_added = True
        except Exception:
            pass
    if not collision_added:
        half = _mesh_aabb_half_size(asset.path)
        if half is not None:
            half = np.clip(half, 0.02, None)
            builder.add_box_collision(half_size=half.tolist())
        else:
            builder.add_box_collision(half_size=[0.15, 0.15, 0.15])

    return builder.build(name=name)


def _freeze_builder_joints(builder):
    """Set all non-root, non-fixed joints to 'fixed' in an ArticulationBuilder."""
    if not hasattr(builder, "link_builders"):
        return
    for lb in builder.link_builders:
        if lb.parent is not None and lb.joint_record.joint_type not in ("fixed", "undefined"):
            lb.joint_record.joint_type = "fixed"
            lb.joint_record.limits = []


def _articulated_loader_scale(asset: ArticulatedAsset, rng: np.random.Generator) -> Optional[float]:
    """Loader scale for an articulated asset, or None to keep native scale.

    scale_policy "normalize" mirrors the GSO mechanism: a random target AABB
    diagonal per spawn, divided by the asset's preprocess-measured native
    diagonal. "fixed" uses the stored per-asset scale. Anything else -> native.
    """
    if asset.scale_policy == "normalize" and asset.native_diag and asset.native_diag > 1e-6:
        lo, hi = asset.target_diag or (RIGID_TARGET_DIAG_MIN, RIGID_TARGET_DIAG_MAX)
        scale = float(rng.uniform(lo, hi)) / asset.native_diag
        # Guard against bad native_diag estimates (translation-only AABB approx).
        clamped = float(np.clip(scale, 0.2, 12.0))
        if clamped != scale:
            print(f"Warning: clamped scale {scale:.2f} -> {clamped:.2f} for {asset.asset_id}")
        return clamped
    if asset.scale != 1.0:
        return asset.scale
    return None


def build_articulated_from_asset(scene, asset: ArticulatedAsset, rng: np.random.Generator,
                                 name: str, freeze_joints: bool = False):
    if asset.uri:
        try:
            from mani_skill.utils.building import articulations
            builder = articulations.get_articulation_builder(scene, asset.uri)
            if hasattr(builder, "fix_root_link"):
                builder.fix_root_link = False
            builder.initial_pose = sapien.Pose()
            if freeze_joints:
                _freeze_builder_joints(builder)
            art = builder.build(name=name)
            return art
        except Exception:
            pass

    if asset.urdf_path:
        loader = scene.create_urdf_loader()
        loader.fix_root_link = False
        loader_scale = _articulated_loader_scale(asset, rng)
        if loader_scale is not None:
            loader.scale = loader_scale
        if hasattr(loader, "set_density"):
            loader.set_density(1000.0)
        elif hasattr(loader, "density"):
            loader.density = 1000.0
        urdf_path = Path(asset.urdf_path)
        validation_error = _validate_articulated_collision_meshes(urdf_path)
        if validation_error:
            raise RuntimeError(f"Invalid articulated asset {asset.asset_id}: {validation_error}")
        base_dir = _resolve_urdf_base(urdf_path)

        if freeze_joints:
            with _chdir(base_dir):
                builder = loader.load_file_as_articulation_builder(str(urdf_path))
            _freeze_builder_joints(builder)
            art = builder.build()
        else:
            with _chdir(base_dir):
                art = loader.load(str(urdf_path))

        if hasattr(art, "set_name"):
            art.set_name(name)
        return art
    return None


def pick_assets(rng: np.random.Generator, assets: List[Any], count: int,
                weighted: bool = False,
                source_weights: Optional[Dict[str, float]] = None) -> List[Any]:
    if count <= 0 or not assets:
        return []
    replace = len(assets) < count
    weights = np.ones(len(assets))
    if weighted:
        scores = [getattr(a, "movability_score", None) for a in assets]
        if any(s is not None for s in scores):
            weights = np.array([max(s or 0.01, 0.01) for s in scores])
            # Weight by log(1+score), capped at a 5:1 ratio.
            weights = np.log1p(weights)
            weights = np.clip(weights, weights.max() / 5.0, None)
    if source_weights:
        # Source fractions follow the configured weights; movability only affects sampling within a source.
        sources = [getattr(a, "source", "") for a in assets]
        totals: Dict[str, float] = {}
        for s, w in zip(sources, weights):
            totals[s] = totals.get(s, 0.0) + float(w)
        factors = np.array([
            source_weights.get(s, 0.0) / totals[s] if totals.get(s) else 0.0
            for s in sources
        ])
        if factors.sum() > 0:
            weights = weights * factors
        # else: no requested source present in the pool -> keep plain weights
    total = weights.sum()
    p = weights / total if total > 0 else None
    if p is not None and int((p > 0).sum()) < count:
        replace = True  # e.g. single-source scene with a small source pool
    idx = rng.choice(len(assets), size=count, replace=replace, p=p)
    return [assets[i] for i in idx]


def split_counts(total: int, frac: float, has_art: bool, has_rigid: bool) -> Tuple[int, int]:
    if has_art and has_rigid:
        num_art = int(round(total * frac))
        num_art = max(0, min(total, num_art))
        return total - num_art, num_art
    if has_art:
        return 0, total
    if has_rigid:
        return total, 0
    return 0, 0


# Count requested and placed objects, split by why an object was dropped.
SPAWN_TALLY = Counter()


def reset_spawn_tally():
    SPAWN_TALLY.clear()


def spawn_objects(scene,
                  rng: np.random.Generator,
                  rigid_assets: List[RigidAsset],
                  art_assets: List[ArticulatedAsset],
                  num_rigid: int,
                  num_art: int,
                  spawn_box: SpawnBox,
                  spawn_cfg: SpawnConfig,
                  velocity_cfg: VelocityConfig,
                  occupied: List[Tuple[float, float]],
                  is_dynamic: bool,
                  physical_material,
                  joint_cfg: Optional[JointMotionConfig] = None,
                  freeze_joints: bool = False,
                  weighted_articulated: bool = False,
                  art_source_weights: Optional[Dict[str, float]] = None,
                  dynamic_rigid_motion: bool = False,
                  force_xy: Optional[Tuple[float, float]] = None):
    """force_xy pins the spawn footprint instead of rejection-sampling one.

    Used when replacing a floor-penetrating object: its slot was legal, but a
    late scene is packed tight enough (26 objects, min_gap ~1 m, a 4.6 m box)
    that random trials almost never land back in the one disc just vacated.
    """
    spawned: List[SpawnedObject] = []

    def place(box: SpawnBox):
        if force_xy is None:
            return sample_spawn_pose(rng, occupied, box, spawn_cfg.min_gap,
                                     spawn_cfg.max_trials, spawn_cfg.max_yaw)
        z = float(rng.uniform(box.min[2], box.max[2]))
        return (np.array([force_xy[0], force_xy[1], z], dtype=np.float32),
                random_yaw_quat(rng, spawn_cfg.max_yaw))

    rigid_list = pick_assets(rng, rigid_assets, num_rigid)
    art_list = pick_assets(rng, art_assets, num_art,
                           weighted=weighted_articulated,
                           source_weights=art_source_weights)
    SPAWN_TALLY["rigid_requested"] += len(rigid_list)
    SPAWN_TALLY["art_requested"] += len(art_list)

    for i, asset in enumerate(rigid_list):
        pos, quat = place(spawn_box)
        if pos is None:
            SPAWN_TALLY["rigid_no_room"] += 1
            continue
        if is_dynamic:
            quat = random_3d_quat(rng)
        if not os.path.exists(asset.path):
            print(f"Rigid asset missing: {asset.path}")
            SPAWN_TALLY["rigid_load_failed"] += 1
            continue
        try:
            obj = build_rigid_from_asset(scene, asset, rng, name=f"rigid_{asset.asset_id}_{i}")
        except Exception as e:
            print(f"Failed to load rigid asset {asset.asset_id}: {e}")
            SPAWN_TALLY["rigid_load_failed"] += 1
            continue
        set_entity_pose(obj, pos, quat)
        apply_physical_material(obj, physical_material)
        set_entity_max_depenetration_velocity(obj, MAX_DEPENETRATION_VEL)

        if is_dynamic:
            vel = sample_dynamic_initial_velocity(rng, pos, velocity_cfg)
            set_entity_velocity(obj, vel)
            if dynamic_rigid_motion:
                # Lower damping so momentum persists into the later frames.
                set_entity_damping(obj, linear=0.02, angular=0.15)
            else:
                set_entity_damping(obj, linear=0.05, angular=0.25)
        SPAWN_TALLY["rigid_placed"] += 1
        spawned.append(SpawnedObject(entity=obj, kind="rigid", asset_id=asset.asset_id,
                                     is_dynamic=is_dynamic, source=asset.source,
                                     spawn_xy=(float(pos[0]), float(pos[1]))))
        occupied.append((float(pos[0]), float(pos[1])))

    for i, asset in enumerate(art_list):
        # Articulated objects use the same spawn box as rigid objects.
        art_spawn_box = spawn_box
        pos, quat = place(art_spawn_box)
        if pos is None:
            SPAWN_TALLY["art_no_room"] += 1
            continue
        if is_dynamic:
            quat = random_upright_quat(rng)
        try:
            art = build_articulated_from_asset(scene, asset, rng, name=f"art_{asset.asset_id}_{i}", freeze_joints=freeze_joints)
        except Exception as e:
            print(f"Failed to load articulated asset {asset.asset_id}: {e}")
            art = None
        if art is None:
            SPAWN_TALLY["art_load_failed"] += 1
            SPAWN_TALLY[f"art_load_failed_{asset.source}"] += 1
            continue
        set_entity_pose(art, pos, quat)
        apply_physical_material(art, physical_material)
        if joint_cfg is not None:
            configure_articulated_joints(art, rng, joint_cfg, is_dynamic)
        else:
            joint_damping = 5.0 if is_dynamic else 1.0
            joint_friction = 0.5 if is_dynamic else 0.2
            for joint in getattr(art, "get_joints", lambda: [])():
                if getattr(joint, "dof", 0) > 0:
                    try:
                        joint.set_drive_property(stiffness=0.0, damping=joint_damping)
                    except Exception:
                        pass
                    if hasattr(joint, "set_friction"):
                        try:
                            joint.set_friction(joint_friction)
                        except Exception:
                            pass

        # Cap depenetration velocity on every object so set_qpos overlaps do not eject bodies.
        set_entity_max_depenetration_velocity(art, MAX_DEPENETRATION_VEL)
        if is_dynamic:
            vel = sample_dynamic_initial_velocity(rng, pos, velocity_cfg)
            vel[2] = min(vel[2], 1.0)
            set_entity_velocity(art, vel)
            set_entity_damping(art, linear=0.15, angular=1.5)
        SPAWN_TALLY["art_placed"] += 1
        spawned.append(SpawnedObject(entity=art, kind="articulated", asset_id=asset.asset_id,
                                     is_dynamic=is_dynamic, source=asset.source,
                                     spawn_xy=(float(pos[0]), float(pos[1]))))
        occupied.append((float(pos[0]), float(pos[1])))

    return spawned


DEFAULT_PENETRATION_Z_FLOOR = -0.3
DEFAULT_PENETRATION_RETRIES = 2
DEFAULT_PENETRATION_MARGIN = 0.15    # probe-vs-render divergence headroom


def make_velocity_clamp(objects: List["SpawnedObject"]):
    """Per-substep velocity cap for the dynamic bodies of `objects`."""
    max_linear_vel = 15.0
    max_angular_vel = 20.0
    max_angular_vel_articulated = 8.0
    dynamic_links = []
    dynamic_articulated_links = set()
    for obj in objects:
        if obj.is_dynamic:
            ents = get_entities(obj.entity)
            dynamic_links.extend(ents)
            if obj.kind == "articulated":
                dynamic_articulated_links.update(id(e) for e in ents)

    def clamp_velocities():
        for ent in dynamic_links:
            is_art = id(ent) in dynamic_articulated_links
            if hasattr(ent, "get_velocity") and hasattr(ent, "set_velocity"):
                try:
                    v = np.array(ent.get_velocity(), dtype=np.float32)
                    spd = float(np.linalg.norm(v))
                    if spd > max_linear_vel:
                        ent.set_velocity((v * max_linear_vel / spd).tolist())
                except Exception:
                    pass
            if hasattr(ent, "get_angular_velocity") and hasattr(ent, "set_angular_velocity"):
                try:
                    w = np.array(ent.get_angular_velocity(), dtype=np.float32)
                    wspd = float(np.linalg.norm(w))
                    limit = max_angular_vel_articulated if is_art else max_angular_vel
                    if wspd > limit:
                        ent.set_angular_velocity((w * limit / wspd).tolist())
                except Exception:
                    pass

    return clamp_velocities


def _entity_z(ent) -> Optional[float]:
    pose = getattr(ent, "pose", None) or (ent.get_pose() if hasattr(ent, "get_pose") else None)
    if pose is None:
        return None
    try:
        return float(np.asarray(pose.p)[2])
    except Exception:
        return None


def probe_floor_penetration(scene, objects: List["SpawnedObject"], steps: int,
                            arrest_static_spin=None) -> Dict[int, float]:
    """Replay `steps` physics substeps; return {object index: worst link z}.

    Only dynamic objects are considered: static bodies are lock_motion'd before
    this point and cannot sink. Restores the starting state before returning.
    """
    dyn = [(i, get_entities(o.entity)) for i, o in enumerate(objects) if o.is_dynamic]
    if not dyn or steps <= 0:
        return {}
    clamp = make_velocity_clamp(objects)
    worst = {i: float("inf") for i, _ in dyn}

    system = getattr(scene, "physx_system", None)
    blob = None
    if system is not None and hasattr(system, "pack"):
        try:
            blob = system.pack()
        except Exception as e:
            print(f"Penetration probe: cannot snapshot physics ({e}); skipping")
            return {}
    if blob is None:
        return {}

    for _ in range(steps):
        scene.step()
        clamp()
        if arrest_static_spin is not None:
            arrest_static_spin()
        for i, ents in dyn:
            for ent in ents:
                z = _entity_z(ent)
                if z is not None and z < worst[i]:
                    worst[i] = z

    system.unpack(blob)
    return worst


def remove_object(scene, obj: "SpawnedObject") -> bool:
    order = (("remove_articulation", "remove_entity", "remove_actor")
             if obj.kind == "articulated" else
             ("remove_entity", "remove_actor", "remove_articulation"))
    for name in order:
        fn = getattr(scene, name, None)
        if not callable(fn):
            continue
        try:
            fn(obj.entity)
            return True
        except Exception:
            continue
    return False


def _is_continuous_joint(joint) -> bool:
    """Check if a joint is continuous (unlimited rotation, e.g. fan blades, wheels).

    Continuous joints get an initial velocity but no drive target.
    """
    for attr in ("type", "get_type"):
        getter = getattr(joint, attr, None)
        if getter is not None:
            try:
                jtype = getter() if callable(getter) else getter
                if "continuous" in str(jtype).lower():
                    return True
            except Exception:
                pass
    # Fallback: if limits span the full ±2π clamp range, it's likely continuous
    limits = None
    for attr in ("get_limits", "get_limit", "limits", "limit"):
        getter = getattr(joint, attr, None)
        if getter is not None:
            try:
                limits = getter() if callable(getter) else getter
            except Exception:
                limits = None
            if limits is not None:
                break
    if limits is not None:
        try:
            limits = np.asarray(limits).flatten()
            if len(limits) >= 2:
                lo, hi = float(limits[0]), float(limits[-1])
                if lo <= -1e9 or hi >= 1e9 or np.isinf(lo) or np.isinf(hi):
                    return True
        except Exception:
            pass
    return False


def _get_joint_limits(joint) -> Tuple[float, float]:
    """Extract (lo, hi) limits from a SAPIEN joint, handling multiple API formats."""
    limits = None
    for attr in ("get_limits", "get_limit", "limits", "limit"):
        getter = getattr(joint, attr, None)
        if getter is not None:
            try:
                limits = getter() if callable(getter) else getter
            except Exception:
                limits = None
            if limits is not None:
                break

    lo, hi = -np.pi / 2, np.pi / 2  # fallback
    if limits is not None:
        try:
            limits = np.asarray(limits)
            if limits.ndim == 2 and limits.shape[0] >= 1 and limits.shape[1] >= 2:
                lo, hi = float(limits[0, 0]), float(limits[0, 1])
            elif limits.ndim == 1 and limits.shape[0] >= 2:
                lo, hi = float(limits[0]), float(limits[1])
        except Exception:
            lo, hi = -np.pi / 2, np.pi / 2

    lo = max(lo, -2 * np.pi)
    hi = min(hi, 2 * np.pi)
    if hi <= lo:
        lo, hi = -np.pi / 2, np.pi / 2
    return lo, hi


def identify_salient_joints(active_joints: list, joint_cfg: JointMotionConfig) -> set:
    """Identify salient joints: those controlling large, visible parts.

    Returns a set of indices into active_joints that should receive
    aggressive drive parameters for maximum visible motion.
    """
    salient = set()
    joint_ranges = []

    for i, joint in enumerate(active_joints):
        lo, hi = _get_joint_limits(joint)
        joint_range = hi - lo
        joint_ranges.append(joint_range)

        # Check joint name for salient keywords
        name = ""
        for attr in ("get_name", "name"):
            getter = getattr(joint, attr, None)
            if getter is not None:
                try:
                    name = getter() if callable(getter) else getter
                    if name:
                        break
                except Exception:
                    pass
        name = str(name).lower()

        has_keyword = any(kw in name for kw in joint_cfg.salient_keywords)
        has_range = joint_range > 0.3  # > ~17 degrees for revolute, > 0.3m for prismatic
        if has_keyword and has_range:
            salient.add(i)

    # If no keyword matches, fall back to the joints with largest range (top 2)
    if not salient and joint_ranges:
        sorted_indices = sorted(range(len(joint_ranges)), key=lambda i: joint_ranges[i], reverse=True)
        for idx in sorted_indices[:2]:
            if joint_ranges[idx] > 0.3:
                salient.add(idx)

    # Cap at 3 salient joints per asset
    if len(salient) > 3:
        scored = [(i, joint_ranges[i]) for i in salient]
        scored.sort(key=lambda x: x[1], reverse=True)
        salient = {i for i, _ in scored[:3]}

    return salient


def configure_articulated_joints(art, rng: np.random.Generator, joint_cfg: JointMotionConfig, is_dynamic: bool):
    """Configure joint positions, drive targets, and velocities for visible articulated motion."""
    joints = getattr(art, "get_joints", lambda: [])()
    active_joints = [j for j in joints if getattr(j, "dof", 0) > 0]
    if not active_joints:
        return

    salient_indices = identify_salient_joints(active_joints, joint_cfg)
    qpos_values = []
    drive_targets = {}  # index -> target
    continuous_indices = set()  # continuous joints: no drive, only velocity

    for i, joint in enumerate(active_joints):
        lo, hi = _get_joint_limits(joint)
        range_size = hi - lo
        is_continuous = _is_continuous_joint(joint)
        is_salient = i in salient_indices

        if is_continuous:
            continuous_indices.add(i)

        if is_continuous:
            qpos = float(rng.uniform(lo, hi))
            qpos_values.append(qpos)
            try:
                joint.set_drive_property(stiffness=0.0, damping=0.05)
            except Exception:
                pass
            if hasattr(joint, "set_friction"):
                try:
                    joint.set_friction(0.01)
                except Exception:
                    pass
        elif is_salient:
            # Salient joints: place qpos near one end, drive target near the other
            if rng.random() < 0.5:
                qpos = float(rng.uniform(lo + range_size * 0.10, lo + range_size * 0.25))
                target = float(rng.uniform(lo + range_size * 0.70, lo + range_size * 0.95))
            else:
                qpos = float(rng.uniform(lo + range_size * 0.75, lo + range_size * 0.90))
                target = float(rng.uniform(lo + range_size * 0.05, lo + range_size * 0.30))
            qpos_values.append(qpos)
            drive_targets[i] = target

            try:
                joint.set_drive_property(stiffness=joint_cfg.drive_stiffness, damping=joint_cfg.drive_damping)
            except Exception:
                pass
            if hasattr(joint, "set_drive_target"):
                try:
                    joint.set_drive_target(target)
                except Exception:
                    pass
            if hasattr(joint, "set_friction"):
                try:
                    joint.set_friction(joint_cfg.salient_friction)
                except Exception:
                    pass
        else:
            # all other joints
            margin = range_size * (1.0 - joint_cfg.qpos_range_fraction) / 2.0
            qpos = float(rng.uniform(lo + margin, hi - margin))
            qpos_values.append(qpos)

            use_drive = rng.random() < joint_cfg.active_drive_prob
            if use_drive:
                target = float(rng.uniform(lo + margin, hi - margin))
                min_diff = 0.30 * range_size
                for _ in range(20):
                    if abs(target - qpos) >= min_diff:
                        break
                    target = float(rng.uniform(lo + margin, hi - margin))
                drive_targets[i] = target

                try:
                    joint.set_drive_property(stiffness=joint_cfg.drive_stiffness, damping=joint_cfg.drive_damping)
                except Exception:
                    pass
                if hasattr(joint, "set_drive_target"):
                    try:
                        joint.set_drive_target(target)
                    except Exception:
                        pass
            else:
                try:
                    joint.set_drive_property(stiffness=0.0, damping=joint_cfg.passive_damping)
                except Exception:
                    pass

            if hasattr(joint, "set_friction"):
                try:
                    joint.set_friction(0.03)
                except Exception:
                    pass

    # Set all joint positions at once
    if hasattr(art, "set_qpos") and qpos_values:
        try:
            art.set_qpos(np.array(qpos_values, dtype=np.float32))
        except Exception:
            pass

    # Set initial joint velocities
    if hasattr(art, "set_qvel") and active_joints:
        try:
            qvel = np.zeros(len(active_joints), dtype=np.float32)
            has_any = False
            for i in range(len(active_joints)):
                if i in continuous_indices:
                    # Continuous joints: always get initial spin
                    qvel[i] = float(rng.uniform(-joint_cfg.max_qvel, joint_cfg.max_qvel))
                    has_any = True
                elif i in salient_indices and i in drive_targets:
                    # Salient joints: always get velocity toward drive target
                    direction = 1.0 if drive_targets[i] > qpos_values[i] else -1.0
                    qvel[i] = direction * float(rng.uniform(1.0, joint_cfg.max_qvel))
                    has_any = True
                elif rng.random() < joint_cfg.qvel_prob:
                    # All other joints: probabilistic velocity
                    qvel[i] = float(rng.uniform(-joint_cfg.max_qvel, joint_cfg.max_qvel))
                    has_any = True
            if has_any:
                art.set_qvel(qvel)
        except Exception:
            pass


# =============================================================================
# Camera helpers
# =============================================================================


DEFAULT_FOVY = 1.0


def resolve_camera_fov(args, seed) -> float:
    """Return this scene's vertical FOV in radians from the --fov* flags.

    A randomised FOV is drawn from its own RNG stream derived from the scene
    seed, not from the main ``rng``, so enabling FOV jitter leaves the object,
    lighting and trajectory sampling sequence untouched: the same seed gives the
    same scene, viewed through a different lens.
    """
    lo, hi = args.fov_min, args.fov_max
    if (lo is None) != (hi is None):
        raise SystemExit("--fov-min and --fov-max must be given together")

    if lo is not None:
        if not 0.0 < lo <= hi < 180.0:
            raise SystemExit(f"--fov-min/--fov-max must satisfy 0 < min <= max < 180 (got {lo}, {hi})")
        # SeedSequence spawn keeps this stream independent of the main rng.
        fov_rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), 0xF0F]))
        fov_deg = float(fov_rng.uniform(lo, hi))
    elif args.fov is not None:
        fov_deg = float(args.fov)
        if not 0.0 < fov_deg < 180.0:
            raise SystemExit(f"--fov must be in (0, 180) degrees (got {fov_deg})")
    else:
        return DEFAULT_FOVY

    return float(np.deg2rad(fov_deg))


DEFAULT_FRAMING_ALPHA_MIN = 0.75


def resolve_framing_alpha(args, seed) -> float:
    """Framing compensation exponent for this scene: 0 = none, 1 = framing fully held.

    With ``--fov-framing-alpha random`` it is drawn per scene from [MIN, 1].
    """
    spec = getattr(args, "fov_framing_alpha", None)
    if spec is None:
        return 1.0 if getattr(args, "fov_keep_framing", False) else 0.0
    spec = str(spec).lower()
    if spec.startswith("random"):
        lo = DEFAULT_FRAMING_ALPHA_MIN
        if ":" in spec:
            lo = float(spec.split(":", 1)[1])
            if not 0.0 <= lo <= 1.0:
                raise SystemExit(f"--fov-framing-alpha random:LO needs LO in [0, 1] (got {lo})")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), 0xA1FA]))
        return float(rng.uniform(lo, 1.0))
    alpha = float(spec)
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"--fov-framing-alpha must be in [0, 1], 'random' or 'random:LO' (got {alpha})")
    return alpha


def apply_camera_fov(cam_cfg: CameraConfig, fovy: float, alpha: float) -> float:
    """Set ``cam_cfg.fovy`` and scale the camera radii by the framing compensation.

    The scale is clamped so the camera stays outside the object area and inside the dome.
    """
    cam_cfg.fovy = fovy
    if alpha <= 0.0:
        return 1.0
    ratio = np.tan(DEFAULT_FOVY / 2) / np.tan(fovy / 2)
    scale = float(np.clip(ratio ** alpha, 0.6, 3.5))
    for attr in ("inner_radius", "outer_radius", "fixed_inner_radius",
                 "fixed_outer_radius", "orbit_radius", "orbit_height", "max_movement"):
        setattr(cam_cfg, attr, getattr(cam_cfg, attr) * scale)
    return scale


DEFAULT_SPAWN_MIN_GAP = 1.5
DEFAULT_SPAWN_XY_SCALE = 1.0


def _resolve_ranged(spec, default, seed, salt, name, lo_limit, hi_limit):
    """Parse a `value` / `random:LO:HI` spec into a per-scene number.

    Each parameter draws from its own SeedSequence salt, so parameters resolved
    this way are statistically independent of one another and of the main rng --
    varying two of them together does not introduce a correlation between them.
    """
    if spec is None:
        return float(default)
    spec = str(spec).lower()
    if spec.startswith("random"):
        parts = spec.split(":")
        lo, hi = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (lo_limit, hi_limit)
        if not lo_limit <= lo <= hi <= hi_limit:
            raise SystemExit(f"--{name} random:LO:HI needs {lo_limit} <= LO <= HI <= {hi_limit} "
                             f"(got {lo}, {hi})")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), salt]))
        return float(rng.uniform(lo, hi))
    val = float(spec)
    if not lo_limit <= val <= hi_limit:
        raise SystemExit(f"--{name} must be within [{lo_limit}, {hi_limit}] (got {val})")
    return val


def resolve_spawn_shape(args, seed):
    """Per-scene (min_gap, xy_scale): minimum spawn clearance and footprint scale."""
    gap = _resolve_ranged(getattr(args, "spawn_min_gap", None), DEFAULT_SPAWN_MIN_GAP,
                          seed, 0x5A17, "spawn-min-gap", 0.0, 3.0)
    xy = _resolve_ranged(getattr(args, "spawn_xy_scale", None), DEFAULT_SPAWN_XY_SCALE,
                         seed, 0x5B18, "spawn-xy-scale", 0.2, 2.0)
    return gap, xy


def sample_point_in_half_sphere_shell(rng: np.random.Generator, inner_radius: float, outer_radius: float, z_offset: float):
    """Uniformly sample a point in a half-sphere shell (volume-uniform)."""
    vec = rng.normal(size=3)
    if vec[2] < 0:
        vec[2] *= -1.0
    vec /= np.linalg.norm(vec) + 1e-8
    u = rng.random()
    r = (u * (outer_radius ** 3 - inner_radius ** 3) + inner_radius ** 3) ** (1.0 / 3.0)
    point = vec * r
    point[2] += z_offset
    return point


def sample_linear_camera_motion_start_end(rng: np.random.Generator, cfg: CameraConfig):
    while True:
        start = sample_point_in_half_sphere_shell(rng, cfg.inner_radius, cfg.outer_radius, cfg.z_offset)
        direction = rng.random(3) - 0.5
        direction /= np.linalg.norm(direction) + 1e-8
        movement = direction * rng.uniform(0.0, cfg.max_movement)
        end = start + movement
        if cfg.inner_radius <= np.linalg.norm(end) <= cfg.outer_radius and end[2] > cfg.z_offset:
            return start, end


def sample_linear_lookat_motion_start_end(rng: np.random.Generator, cfg: CameraConfig):
    while True:
        lookat_through = sample_point_in_half_sphere_shell(rng, 0.0, cfg.lookat_inner_radius, 0.0)
        while True:
            lookat_start = sample_point_in_half_sphere_shell(rng, 0.0, cfg.lookat_outer_radius, 0.0)
            if lookat_start[-1] < cfg.lookat_inner_radius:
                break
        continuation = rng.random() * 1.0
        lookat_end = lookat_through + continuation * (lookat_through - lookat_start)
        if rng.random() < 0.5:
            lookat_start, lookat_end = lookat_end, lookat_start
        return lookat_start, lookat_end


def sample_kubric_linear_trajectory(rng: np.random.Generator, num_frames: int, cfg: CameraConfig, panning: bool):
    start, end = sample_linear_camera_motion_start_end(rng, cfg)
    lookat_start, lookat_end = (None, None)
    if panning:
        lookat_start, lookat_end = sample_linear_lookat_motion_start_end(rng, cfg)

    positions, lookats = [], []
    for frame in range(num_frames):
        interp = (frame + 1) / (num_frames + 2)
        pos = interp * start + (1 - interp) * end
        if panning and lookat_start is not None:
            lookat = interp * lookat_start + (1 - interp) * lookat_end
        else:
            lookat = cfg.look_at
        positions.append(pos)
        lookats.append(lookat)
    return positions, lookats


def sample_orbit_camera_path(num_frames: int, cfg: CameraConfig):
    center = cfg.look_at
    angles = np.linspace(0.0, cfg.orbit_rotation, num_frames)
    positions = []
    for ang in angles:
        pos = np.array([
            np.cos(ang) * cfg.orbit_radius,
            np.sin(ang) * cfg.orbit_radius,
            cfg.orbit_height,
        ], dtype=np.float32)
        positions.append(pos + center)
    return positions


def set_camera_pose(camera, position: np.ndarray, look_at: np.ndarray):
    fwd = look_at - position
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    left = np.cross([0, 0, 1], fwd)
    left = left / (np.linalg.norm(left) + 1e-8)
    up = np.cross(fwd, left)

    mat44 = np.eye(4, dtype=np.float32)
    mat44[:3, :3] = np.stack([fwd, left, up], axis=1)
    mat44[:3, 3] = position
    camera.entity.set_pose(sapien.Pose(mat44))


# =============================================================================
# Image/Annotation helpers
# =============================================================================


def depth_from_position(position):
    depth = (-position[..., 2]).astype(np.float32)
    valid = position[..., 3] < 1.0 if position.shape[-1] > 3 else np.isfinite(depth)
    return np.where(valid, depth, np.nan)


def mask_dome_depth(depth: np.ndarray, actor_labels: np.ndarray,
                    dome_actor_ids: list) -> np.ndarray:
    """Set depth to -1 for dome/sky background pixels identified by segmentation."""
    masked = depth.copy()
    bg_ids = set(dome_actor_ids)
    bg_ids.add(0)
    for aid in bg_ids:
        masked[actor_labels == aid] = -1.0
    masked[np.isnan(masked)] = -1.0
    return masked


def depth_to_png(depth: np.ndarray) -> np.ndarray:
    """Convert float depth to 16-bit PNG-friendly array.

    Foreground pixels are linearly mapped to [1, 65535] (0 is reserved for
    invalid / background where depth == -1).
    """
    valid = depth > 0
    out = np.zeros(depth.shape, dtype=np.uint16)
    if not np.any(valid):
        return out
    dmin, dmax = depth[valid].min(), depth[valid].max()
    span = dmax - dmin if dmax > dmin else 1.0
    out[valid] = ((depth[valid] - dmin) / span * 65534 + 1).clip(1, 65535).astype(np.uint16)
    return out


def process_segmentation(seg):
    mesh = np.where(seg[..., 0] < 0, 0, seg[..., 0]).astype(np.int32)
    actor = np.where(seg[..., 1] < 0, 0, seg[..., 1]).astype(np.int32)
    return mesh, actor


def _rotation_angle_between(R_a: np.ndarray, R_b: np.ndarray) -> float:
    rel = R_a.T @ R_b
    cos_val = float((np.trace(rel) - 1.0) * 0.5)
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.arccos(cos_val))


def _relative_pose_constant(poses_a: List[np.ndarray], poses_b: List[np.ndarray],
                            translation_tol: float, rotation_tol: float) -> bool:
    if not poses_a or not poses_b or len(poses_a) != len(poses_b):
        return False
    rel_ref = np.linalg.inv(poses_a[0]) @ poses_b[0]
    rel_ref_t = rel_ref[:3, 3]
    rel_ref_R = rel_ref[:3, :3]

    for Ta, Tb in zip(poses_a[1:], poses_b[1:]):
        rel = np.linalg.inv(Ta) @ Tb
        if float(np.linalg.norm(rel[:3, 3] - rel_ref_t)) > translation_tol:
            return False
        if _rotation_angle_between(rel_ref_R, rel[:3, :3]) > rotation_tol:
            return False
    return True


def build_relative_pose_segmentation(actor_pose_history: Dict[int, List[np.ndarray]],
                                     translation_tol: float = 1e-3,
                                     rotation_tol: float = 1e-3) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    actor_ids = sorted(int(a) for a in actor_pose_history.keys())
    if not actor_ids:
        return {}, []

    parent = {aid: aid for aid in actor_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i, aid_i in enumerate(actor_ids):
        poses_i = actor_pose_history[aid_i]
        for aid_j in actor_ids[i + 1:]:
            if _relative_pose_constant(poses_i, actor_pose_history[aid_j], translation_tol, rotation_tol):
                union(aid_i, aid_j)

    groups: Dict[int, List[int]] = {}
    for aid in actor_ids:
        root = find(aid)
        groups.setdefault(root, []).append(aid)

    sorted_roots = sorted(groups.keys(), key=lambda r: min(groups[r]))
    root_to_seg = {root: idx + 1 for idx, root in enumerate(sorted_roots)}

    actor_to_seg = {}
    segments = []
    for root in sorted_roots:
        seg_id = int(root_to_seg[root])
        members = sorted(int(a) for a in groups[root])
        segments.append({"seg_id": seg_id, "actor_ids": members})
        for aid in members:
            actor_to_seg[aid] = seg_id
    return actor_to_seg, segments


def remap_actor_labels_to_seg_ids(actor_labels: np.ndarray, actor_to_seg: Dict[int, int]) -> np.ndarray:
    out = np.zeros(actor_labels.shape, dtype=np.int32)
    for aid in np.unique(actor_labels):
        aid_int = int(aid)
        if aid_int == 0:
            continue
        seg_id = actor_to_seg.get(aid_int)
        if seg_id is not None:
            out[actor_labels == aid_int] = seg_id
    return out


def colorize_segmentation(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    color = palette[labels % len(palette)].copy()
    color[labels == 0] = np.array([0, 0, 0], dtype=np.uint8)
    return color


def compute_object_coordinates(position, actor_labels, objects, cam_model_mat):
    H, W = actor_labels.shape
    obj_coords = np.zeros((H, W, 3), dtype=np.uint16)

    id_to_entity = {}
    for obj in objects:
        for ent in get_entities(obj):
            aid = get_actor_id(ent)
            if aid is not None:
                id_to_entity[aid] = ent

    cam_pts = position[..., :3]
    valid = position[..., 3] < 1.0 if position.shape[-1] > 3 else np.ones((H, W), dtype=bool)
    R, t = cam_model_mat[:3, :3], cam_model_mat[:3, 3]
    world_pts = np.einsum("ij,hwj->hwi", R, cam_pts) + t

    entity_data = {}
    for aid in np.unique(actor_labels):
        if aid == 0 or aid not in id_to_entity:
            continue
        ent = id_to_entity[aid]
        mask = (actor_labels == aid) & valid
        if not np.any(mask):
            continue

        T = ent.get_pose().to_transformation_matrix()
        R_inv = T[:3, :3].T
        t_inv = -R_inv @ T[:3, 3]
        local = (R_inv @ world_pts[mask].T).T + t_inv

        if ent not in entity_data:
            entity_data[ent] = []
        entity_data[ent].append((mask, local, aid))

    bbox_info = {}
    for ent, data_list in entity_data.items():
        all_pts = np.concatenate([d[1] for d in data_list])
        bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
        bbox_size = np.where(bbox_max - bbox_min < 1e-6, 1.0, bbox_max - bbox_min)

        for mask, local, aid in data_list:
            bbox_info[aid] = (bbox_min.copy(), bbox_max.copy(), bbox_size.copy())
            norm = (local - bbox_min) / bbox_size
            obj_coords[mask] = (norm * 65535).clip(0, 65535).astype(np.uint16)

    return obj_coords, bbox_info


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Kubric-like SAPIEN generator")
    default_assets_root = os.environ.get("SAPIEN_ASSETS_ROOT")
    default_assets_index = (
        str(Path(default_assets_root) / "assets_index.json")
        if default_assets_root else "assets_index.json"
    )
    parser.add_argument("--assets-index", type=str, default=default_assets_index,
                        help="JSON index from preprocess_assets.py (required).")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-static", type=int, default=5)
    parser.add_argument("--max-static", type=int, default=10)
    parser.add_argument("--min-dynamic", type=int, default=10)
    parser.add_argument("--max-dynamic", type=int, default=20)
    parser.add_argument("--num-static", type=int, default=None, help="Fixed number of static objects (overrides min/max).")
    parser.add_argument("--num-dynamic", type=int, default=None, help="Fixed number of dynamic objects (overrides min/max).")
    parser.add_argument("--num-static-rigid", type=int, default=None, help="Fixed rigid count for static objects.")
    parser.add_argument("--num-static-articulated", type=int, default=None, help="Fixed articulated count for static objects.")
    parser.add_argument("--num-dynamic-rigid", type=int, default=None, help="Fixed rigid count for dynamic objects.")
    parser.add_argument("--num-dynamic-articulated", type=int, default=None, help="Fixed articulated count for dynamic objects.")
    parser.add_argument("--articulated-fraction", type=float, default=1)
    parser.add_argument("--scene-mix-mode", type=str, default="fixed",
                        choices=["fixed", "varied"],
                        help="fixed: every scene uses --articulated-fraction. varied: per-scene mix of all-articulated, all-rigid and mixed scenes.")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12)")
    parser.add_argument("--width", type=int, default=256, help="Render width (default: 256)")
    parser.add_argument("--height", type=int, default=256, help="Render height (default: 256)")
    parser.add_argument(
        "--camera-mode",
        choices=["linear", "orbit", "fixed_random", "linear_movement", "linear_movement_linear_lookat"],
        default="linear",
    )
    parser.add_argument("--spawn-min-gap", type=str, default=None, metavar="G",
                        help="Minimum centre-to-centre spawn clearance in metres, or 'random:LO:HI' per scene. Default 1.5.")
    parser.add_argument("--spawn-xy-scale", type=str, default=None, metavar="S",
                        help="Scale of the spawn footprint, or 'random:LO:HI' per scene. Default 1.0.")
    parser.add_argument("--fov", type=float, default=None,
                        help="Vertical field of view in degrees (default: 57.3).")
    parser.add_argument("--fov-min", type=float, default=None,
                        help="Lower bound of the per-scene vertical FOV in degrees. Use with --fov-max.")
    parser.add_argument("--fov-max", type=float, default=None,
                        help="Upper bound for per-scene random vertical FOV in degrees; use with --fov-min.")
    parser.add_argument("--fov-keep-framing", action="store_true",
                        help="Hold framing across FOVs by scaling camera distance (equivalent to "
                             "--fov-framing-alpha 1).")
    parser.add_argument("--fov-framing-alpha", type=str, default=None, metavar="A",
                        help="Framing compensation exponent in [0, 1], or 'random' / 'random:LO' to draw it per scene from [LO, 1].")
    parser.add_argument("--max-camera-movement", type=float, default=3.0)
    parser.add_argument("--camera-inner-radius", type=float, default=4.0)
    parser.add_argument("--camera-outer-radius", type=float, default=6.0)
    parser.add_argument("--camera-z-offset", type=float, default=0.1)
    parser.add_argument("--lookat-inner-radius", type=float, default=0.5)
    parser.add_argument("--lookat-outer-radius", type=float, default=2.5)
    parser.add_argument("--hdri-path", type=str, default=None)
    parser.add_argument("--hdri-dir", type=str, default=None)
    # Joint motion parameters
    parser.add_argument("--drive-stiffness", type=float, default=5.0, help="PD controller stiffness for active joints")
    parser.add_argument("--drive-damping", type=float, default=0.5, help="PD controller damping for active joints")
    parser.add_argument("--passive-damping", type=float, default=0.15, help="Damping for passive joints")
    parser.add_argument("--active-drive-prob", type=float, default=0.7, help="Probability of setting drive target per joint")
    parser.add_argument("--qpos-range-fraction", type=float, default=0.8, help="Fraction of joint limits to use")
    parser.add_argument("--qvel-prob", type=float, default=1.0, help="Probability of initial joint velocity")
    parser.add_argument("--max-qvel", type=float, default=3.0, help="Max initial joint velocity")
    parser.add_argument("--salient-stiffness-multiplier", type=float, default=1.0,
                        help="Stiffness multiplier for salient joints")
    parser.add_argument("--salient-friction", type=float, default=0.04,
                        help="Friction for salient joints")
    parser.add_argument("--no-weighted-sampling", action="store_true",
                        help="Disable weighted sampling (use uniform random instead)")
    parser.add_argument("--source-mix-mode", choices=["index", "fixed", "varied"], default="index",
                        help="Articulated source mixing: 'index' (use the asset index), 'fixed' or 'varied' (per-scene jitter).")
    parser.add_argument("--source-weights", type=str, default=None,
                        help="Override articulated source weights, e.g. "
                             "'partnet=0.5,infinigen=0.3,articraft=0.2'")
    parser.add_argument("--output-format", choices=["full", "cotracker"], default="full",
                        help="Output format: full (default) or cotracker (CoTracker Kubric format)")
    parser.add_argument("--penetration-retries", type=int,
                        default=DEFAULT_PENETRATION_RETRIES, metavar="N",
                        help="Replace dynamic objects that sink below --penetration-z-floor, up to N times. 0 drops them, -1 disables.")
    parser.add_argument("--penetration-z-floor", type=float,
                        default=DEFAULT_PENETRATION_Z_FLOOR, metavar="Z",
                        help="Sink depth (m) that counts as floor penetration.")
    parser.add_argument("--penetration-probe-margin", type=float,
                        default=DEFAULT_PENETRATION_MARGIN, metavar="M",
                        help="Margin in metres above --penetration-z-floor at which objects are replaced. Default 0.15.")
    parser.add_argument("--dynamic-rigid-motion", action="store_true",
                        help="Stagger drop heights and lower friction so rigid objects keep moving.")
    parser.add_argument("--rich-lighting", action="store_true",
                        help="Randomize lighting per scene (angle, size, intensity, color temperature).")
    parser.add_argument("--no-hdri-dome", action="store_true",
                        help="Skip the HDRI dome meshes and light the scene from the environment map directly.")
    parser.add_argument("--env-light-scale", type=float, default=1.0,
                        help="Scale the environment map used for lighting. The visible background is unchanged.")
    parser.add_argument("--composition-target", type=str, default=None,
                        help="Dataset-level share per source, e.g. 'partnet=0.30,infinigen=0.20,articraft=0.20,gso=0.30'. Must sum to 1.")
    parser.add_argument("--source-keep-rates", type=str,
                        default="partnet=0.74,infinigen=0.82,articraft=0.68,gso=1.0",
                        help="Per-source fraction of requested objects that end up placed, used with --composition-target.")
    parser.add_argument("--single-source-prob", type=float, default=0.30,
                        help="With --composition-target: probability that a scene uses a single source.")
    parser.add_argument("--composition-alpha", type=float, default=8.0,
                        help="With --composition-target: Dirichlet concentration of the per-scene jitter.")
    parser.add_argument("--light-level-min", type=float, default=None,
                        help="Lower bound of the per-scene lighting level in [0, 1], from sharp shadows (0) to flat light (1).")
    parser.add_argument("--light-level-max", type=float, default=None,
                        help="Per-scene brightness high end in [0,1]; see --light-level-min. "
                             "min==max pins the level (debugging).")
    parser.add_argument("--area-light-gain", type=float, default=1.0,
                        help="Multiplier on the key light emission with --rich-lighting.")
    parser.add_argument("--rt-path-depth", type=int, default=0,
                        help="Ray-tracing path depth. 0 keeps the SAPIEN default.")
    parser.add_argument("--freeze-joints", action="store_true",
                        help="Lock all articulated joints at default position (objects behave as rigid)")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    assets_index_path = Path(args.assets_index).expanduser()
    if not assets_index_path.exists():
        raise SystemExit(f"assets_index not found: {assets_index_path}. Run preprocess_assets.py first.")
    assets_index = load_assets_index(str(assets_index_path))
    rigid_assets = assets_index.rigid
    art_assets = assets_index.articulated

    # Use pre-computed scores from assets_index.json; fall back to runtime scoring
    if not getattr(args, "no_weighted_sampling", False) and art_assets:
        has_scores = any(a.movability_score is not None for a in art_assets)
        if not has_scores:
            print(f"No pre-computed scores in assets_index. Scoring {len(art_assets)} assets at runtime...")
            print("  (Tip: re-run preprocess_assets.py to pre-compute scores and skip this step)")
            for asset in art_assets:
                asset.movability_score = score_articulated_asset(asset)
        scores = [a.movability_score for a in art_assets if a.movability_score is not None]
        if scores:
            print(f"  Movability scores: [{min(scores):.3f}, {max(scores):.3f}], "
                  f"median: {sorted(scores)[len(scores)//2]:.3f}")

    mix_cfg = ObjectMixConfig(
        min_num_static=args.min_static,
        max_num_static=args.max_static,
        min_num_dynamic=args.min_dynamic,
        max_num_dynamic=args.max_dynamic,
        articulated_fraction=args.articulated_fraction,
    )
    reset_spawn_tally()
    spawn_cfg = SpawnConfig()
    spawn_gap, spawn_xy = resolve_spawn_shape(args, args.seed)
    spawn_cfg.min_gap = spawn_gap
    if args.dynamic_rigid_motion:
        # Stagger drop heights so objects keep colliding in later frames.
        spawn_cfg.dynamic_box = SpawnBox(min=(-3.0, -3.0, 4.0), max=(3.0, 3.0, 12.0))
    if spawn_xy != 1.0:
        # Scale the footprint in XY only.
        for attr in ("static_box", "dynamic_box"):
            box = getattr(spawn_cfg, attr)
            setattr(spawn_cfg, attr, SpawnBox(
                min=(box.min[0] * spawn_xy, box.min[1] * spawn_xy, box.min[2]),
                max=(box.max[0] * spawn_xy, box.max[1] * spawn_xy, box.max[2])))
    print(f"Spawn: min_gap {spawn_gap:.2f} m, footprint x{spawn_xy:.2f} "
          f"({spawn_cfg.dynamic_box.min[0]:.2f} to {spawn_cfg.dynamic_box.max[0]:.2f} m)")
    vel_cfg = VelocityConfig()
    cam_cfg = CameraConfig(
        width=args.width,
        height=args.height,
        mode=args.camera_mode,
        max_movement=args.max_camera_movement,
        inner_radius=args.camera_inner_radius,
        outer_radius=args.camera_outer_radius,
        z_offset=args.camera_z_offset,
        lookat_inner_radius=args.lookat_inner_radius,
        lookat_outer_radius=args.lookat_outer_radius,
    )
    scene_fovy = resolve_camera_fov(args, args.seed)
    framing_alpha = resolve_framing_alpha(args, args.seed)
    framing_scale = apply_camera_fov(cam_cfg, scene_fovy, framing_alpha)
    print(f"Camera FOV: {np.rad2deg(scene_fovy):.1f} deg vertical "
          f"(focal {0.5 / np.tan(scene_fovy / 2):.4f}, framing alpha {framing_alpha:.2f}, "
          f"scale {framing_scale:.2f})")
    sim_cfg = SimConfig(num_frames=args.num_frames, fps=args.fps)
    if sim_cfg.steps_per_frame is None:
        sim_cfg.steps_per_frame = max(1, int(round((1.0 / sim_cfg.fps) / sim_cfg.timestep)))
    freeze_joints = getattr(args, "freeze_joints", False)
    if freeze_joints:
        joint_cfg = None
    else:
        joint_cfg = JointMotionConfig(
            drive_stiffness=args.drive_stiffness,
            drive_damping=args.drive_damping,
            passive_damping=args.passive_damping,
            active_drive_prob=args.active_drive_prob,
            qpos_range_fraction=args.qpos_range_fraction,
            qvel_prob=args.qvel_prob,
            max_qvel=args.max_qvel,
            salient_stiffness_multiplier=args.salient_stiffness_multiplier,
            salient_friction=args.salient_friction,
        )
    out_cfg = OutputConfig(output_dir=args.output_dir)

    if (args.light_level_min is None) != (args.light_level_max is None):
        raise SystemExit("--light-level-min and --light-level-max must be given together")
    light_level = None
    env_scale_eff = args.env_light_scale
    key_mult = 1.0
    if args.light_level_min is not None:
        if not (0.0 <= args.light_level_min <= args.light_level_max <= 1.0):
            raise SystemExit("need 0 <= --light-level-min <= --light-level-max <= 1")
        assert len({f"{l:.2f}" for l in ENV_LEVELS}) == len(ENV_LEVELS)
        level_rng = np.random.default_rng(int(args.seed or 0) + 2_000_003)
        light_level = float(level_rng.uniform(args.light_level_min, args.light_level_max))
        env_scale_eff = ENV_LEVELS[min(int(light_level * len(ENV_LEVELS)), len(ENV_LEVELS) - 1)]
        key_mult = 1.0 - light_level

    bg_cfg = BackgroundConfig(hdri_path=args.hdri_path, hdri_dir=args.hdri_dir,
                              env_light_scale=env_scale_eff,
                              use_hdri_dome=not args.no_hdri_dome)

    cotracker_mode = getattr(args, "output_format", "full") == "cotracker"
    os.makedirs(out_cfg.output_dir, exist_ok=True)
    if cotracker_mode:
        for sub in ["frames", "depths"]:
            os.makedirs(os.path.join(out_cfg.output_dir, sub), exist_ok=True)
    else:
        for sub in ["rgb", "depth", "segmentation", "normal", "object_coordinates", "seg_rp"]:
            os.makedirs(os.path.join(out_cfg.output_dir, sub), exist_ok=True)

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(sim_cfg.ray_tracing_samples)
    sapien.render.set_ray_tracing_denoiser("optix")
    if args.rt_path_depth > 0:
        # Fewer bounces = less indirect fill bouncing off the HDRI dome = crisper shadows.
        sapien.render.set_ray_tracing_path_depth(args.rt_path_depth)

    scene = sapien.Scene()
    scene.set_timestep(sim_cfg.timestep)
    if args.dynamic_rigid_motion:
        # Lower friction and higher restitution so rigid objects slide and bounce.
        static_mat = scene.create_physical_material(0.9, 0.9, 0.3)
        dynamic_mat = scene.create_physical_material(0.5, 0.4, 0.4)
    else:
        static_mat = scene.create_physical_material(0.9, 0.9, 0.1)
        dynamic_mat = scene.create_physical_material(0.9, 0.9, 0.2)

    hdri_path, floor, domes = create_background(scene, rng, bg_cfg, assets_index)
    if floor is not None:
        apply_physical_material(floor, static_mat)

    floor_actor_id = get_actor_id(floor) if floor is not None else None
    dome_actor_ids = []
    for dome in domes:
        aid = get_actor_id(dome)
        if aid is not None:
            dome_actor_ids.append(aid)
    light_params = setup_lighting(scene, rng, rich_lighting=args.rich_lighting, scene_seed=(args.seed or 0),
                                  area_light_gain=args.area_light_gain * key_mult)

    if args.num_static is not None:
        mix_cfg.min_num_static = args.num_static
        mix_cfg.max_num_static = args.num_static
    if args.num_dynamic is not None:
        mix_cfg.min_num_dynamic = args.num_dynamic
        mix_cfg.max_num_dynamic = args.num_dynamic

    num_static = rng.integers(mix_cfg.min_num_static, mix_cfg.max_num_static + 1)
    num_dynamic = rng.integers(mix_cfg.min_num_dynamic, mix_cfg.max_num_dynamic + 1)

    has_rigid = bool(rigid_assets)
    has_art = bool(art_assets)
    def resolve_counts(default_total: int, override_rigid: Optional[int], override_art: Optional[int]):
        if override_rigid is None and override_art is None:
            return None
        rigid = override_rigid if override_rigid is not None else 0
        art = override_art if override_art is not None else 0
        if override_rigid is None or override_art is None:
            target = default_total
            if override_rigid is None:
                rigid = max(0, target - art)
            if override_art is None:
                art = max(0, target - rigid)
        return rigid, art

    scene_rigid_frac = None
    scene_art_fraction = mix_cfg.articulated_fraction
    composition_active = bool(getattr(args, "composition_target", None))
    if (not composition_active and getattr(args, "scene_mix_mode", "fixed") == "varied"
            and has_rigid and has_art):
        u = rng.random()
        if u < P_PURE_ART:
            scene_rigid_frac = 0.0
        elif u < P_PURE_ART + P_PURE_RIGID:
            scene_rigid_frac = 1.0
        else:
            scene_rigid_frac = float(rng.uniform(MIX_RIGID_LO, MIX_RIGID_HI))
        scene_art_fraction = 1.0 - scene_rigid_frac

    # Per-scene articulated-source mix, from meta.sources or --source-weights.
    weighted_articulated = not getattr(args, "no_weighted_sampling", False)
    scene_source_weights = None
    scene_single_source = None
    source_mix_mode = getattr(args, "source_mix_mode", "index")
    if source_mix_mode == "index":
        source_mix_mode = (assets_index.source_mix or {}).get("mode", "fixed")
    active_sources = sorted({a.source for a in art_assets})
    if not composition_active and len(active_sources) >= 2:
        cfg_weights = dict(assets_index.source_weights)
        # Sources missing from the index default to 1.0; with --source-weights, unlisted sources get 0.
        default_w = 1.0
        if getattr(args, "source_weights", None):
            cfg_weights = {}
            default_w = 0.0
            for part in args.source_weights.split(","):
                name, _, val = part.partition("=")
                cfg_weights[name.strip()] = float(val)
        w = np.array([max(cfg_weights.get(s, default_w), 0.0) for s in active_sources], dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones(len(active_sources))
        w = w / w.sum()
        mix = assets_index.source_mix or {}
        u_single = rng.random()
        dirichlet = rng.dirichlet(float(mix.get("dirichlet_concentration", 8.0)) * w + 1e-3)
        single_pick = int(rng.choice(len(active_sources), p=w))
        if source_mix_mode == "varied":
            if u_single < float(mix.get("p_single_source", 0.15)):
                w = np.zeros_like(w)
                w[single_pick] = 1.0
                scene_single_source = active_sources[single_pick]
            else:
                w = dirichlet
        scene_source_weights = {s: float(x) for s, x in zip(active_sources, w)}

    # --composition-target: per-scene composition sampled around a dataset-level share per source.
    composition_meta = None
    if composition_active:
        rigid_sources = sorted({a.source for a in rigid_assets})
        all_sources = sorted(set(active_sources) | set(rigid_sources))
        target = _parse_kv_floats(args.composition_target, "--composition-target")
        if set(target) != set(all_sources):
            raise SystemExit(f"--composition-target must cover exactly the active sources "
                             f"{all_sources}, got {sorted(target)}")
        tot = sum(target.values())
        if abs(tot - 1.0) > 1e-6:
            raise SystemExit(f"--composition-target must sum to 1 (got {tot:.6f})")
        keep = _parse_kv_floats(args.source_keep_rates, "--source-keep-rates")
        missing = [s for s in all_sources if keep.get(s, 0.0) <= 0.0]
        if missing:
            raise SystemExit(f"--source-keep-rates needs a positive rate for {missing}")
        if not (0.0 <= args.single_source_prob <= 1.0):
            raise SystemExit("--single-source-prob must be in [0,1]")
        # Requested weight is proportional to target share / keep rate.
        w_all = np.array([target[s] / keep[s] for s in all_sources], dtype=np.float64)
        w_all = w_all / w_all.sum()
        # Fixed draw pattern: all three draws consumed on every scene.
        u_comp = rng.random()
        dirichlet_all = rng.dirichlet(float(args.composition_alpha) * w_all + 1e-3)
        single_pick = int(rng.choice(len(all_sources), p=w_all))
        if u_comp < float(args.single_source_prob):
            pi = np.zeros_like(w_all)
            pi[single_pick] = 1.0
            scene_single_source = all_sources[single_pick]
        else:
            pi = dirichlet_all
            scene_single_source = None
        pi_map = {s: float(x) for s, x in zip(all_sources, pi)}
        scene_rigid_frac = float(sum(pi_map[s] for s in rigid_sources))
        scene_art_fraction = 1.0 - scene_rigid_frac
        art_w = np.array([pi_map[s] for s in active_sources], dtype=np.float64)
        art_w = art_w / art_w.sum() if art_w.sum() > 0 else np.full(max(len(active_sources), 1),
                                                                    1.0 / max(len(active_sources), 1))
        scene_source_weights = {s: float(x) for s, x in zip(active_sources, art_w)}
        source_mix_mode = "composition"
        print(f"Composition target active: rigid_frac={scene_rigid_frac:.2f} "
              f"art_weights={scene_source_weights} "
              f"(--scene-mix-mode/--source-mix-mode/--articulated-fraction ignored)")
        composition_meta = {
            "composition_target": target,
            "source_keep_rates": {s: float(keep[s]) for s in all_sources},
            "compensated_weights": {s: float(x) for s, x in zip(all_sources, w_all)},
            "single_source_prob": float(args.single_source_prob),
            "composition_alpha": float(args.composition_alpha),
            "scene_source_weights": pi_map,
        }

    override_static = resolve_counts(num_static, args.num_static_rigid, args.num_static_articulated)
    if override_static is None:
        rigid_static, art_static = split_counts(num_static, scene_art_fraction, has_art, has_rigid)
    else:
        rigid_static, art_static = override_static
        num_static = rigid_static + art_static

    override_dynamic = resolve_counts(num_dynamic, args.num_dynamic_rigid, args.num_dynamic_articulated)
    if override_dynamic is None:
        rigid_dyn, art_dyn = split_counts(num_dynamic, scene_art_fraction, has_art, has_rigid)
    else:
        rigid_dyn, art_dyn = override_dynamic
        num_dynamic = rigid_dyn + art_dyn

    if not has_rigid and rigid_static + rigid_dyn > 0:
        raise RuntimeError("Rigid assets required by counts, but assets_index has none.")
    if not has_art and art_static + art_dyn > 0:
        raise RuntimeError("Articulated assets required by counts, but assets_index has none.")

    occupied: List[Tuple[float, float]] = []
    objects: List[SpawnedObject] = []

    # Spawn static objects
    if rigid_static > 0:
        objects.extend(spawn_objects(
            scene, rng, rigid_assets, art_assets, rigid_static, 0,
            spawn_cfg.static_box, spawn_cfg, vel_cfg, occupied, False, static_mat,
        ))

    objects.extend(spawn_objects(
        scene, rng, rigid_assets, art_assets, 0, art_static,
        spawn_cfg.static_box, spawn_cfg, vel_cfg, occupied, False, static_mat,
        joint_cfg=joint_cfg, freeze_joints=freeze_joints,
        weighted_articulated=weighted_articulated, art_source_weights=scene_source_weights,
    ))

    for _ in range(sim_cfg.settle_steps):
        scene.step()
    scene.update_render()

    # Set joint positions before locking roots; set_qpos is ignored on locked articulations.
    if joint_cfg is not None:
        for obj in objects:
            if obj.kind == "articulated":
                configure_articulated_joints(obj.entity, rng, joint_cfg, is_dynamic=False)

    for obj in objects:
        if obj.kind == "articulated":
            set_entity_velocity(obj.entity, np.zeros(3, dtype=np.float32))
            entities = get_entities(obj.entity)
            # Lock root link to keep static articulated objects in place
            if entities:
                root = entities[0]
                if hasattr(root, "set_angular_velocity"):
                    try:
                        root.set_angular_velocity([0.0, 0.0, 0.0])
                    except Exception:
                        pass
                if hasattr(root, "lock_motion"):
                    try:
                        root.lock_motion(True, True, True, True, True, True)
                    except Exception:
                        pass
        else:
            # Rigid objects: zero velocity and lock all motion
            set_entity_velocity(obj.entity, np.zeros(3, dtype=np.float32))
            for ent in get_entities(obj.entity):
                if hasattr(ent, "set_angular_velocity"):
                    try:
                        ent.set_angular_velocity([0.0, 0.0, 0.0])
                    except Exception:
                        pass
                if hasattr(ent, "lock_motion"):
                    try:
                        ent.lock_motion(True, True, True, True, True, True)
                    except Exception:
                        pass

    # Cap root angular velocity of static articulations each substep to stop in-place spinning.
    STATIC_MAX_ROOT_ANGVEL = 1.5  # rad/s -- allows tipping, forbids fast spin
    static_arts = [obj.entity for obj in objects
                   if obj.kind == "articulated"
                   and hasattr(obj.entity, "get_root_angular_velocity")
                   and hasattr(obj.entity, "set_root_angular_velocity")]

    def arrest_static_spin():
        for art in static_arts:
            try:
                w = np.asarray(art.get_root_angular_velocity(), dtype=np.float32)
                speed = float(np.linalg.norm(w))
                if speed > STATIC_MAX_ROOT_ANGVEL:
                    art.set_root_angular_velocity((w * (STATIC_MAX_ROOT_ANGVEL / speed)).tolist())
            except Exception:
                pass

    # Spawn dynamic objects
    if rigid_dyn > 0:
        objects.extend(spawn_objects(
            scene, rng, rigid_assets, art_assets, rigid_dyn, 0,
            spawn_cfg.dynamic_box, spawn_cfg, vel_cfg, occupied, True, dynamic_mat,
            dynamic_rigid_motion=args.dynamic_rigid_motion,
        ))

    objects.extend(spawn_objects(
        scene, rng, rigid_assets, art_assets, 0, art_dyn,
        spawn_cfg.dynamic_box, spawn_cfg, vel_cfg, occupied, True, dynamic_mat,
        joint_cfg=joint_cfg, freeze_joints=freeze_joints,
        weighted_articulated=weighted_articulated, art_source_weights=scene_source_weights,
    ))

    # Replace dynamic objects that sink through the floor with a different asset.
    penetration_meta = {
        "retries_allowed": int(args.penetration_retries),
        "z_floor": float(args.penetration_z_floor),
        "probe_margin": float(args.penetration_probe_margin),
        "attempts": [],
        "blocked_assets": [],
        "replaced": 0,
        "dropped": 0,
    }
    if args.penetration_retries >= 0 and sim_cfg.steps_per_frame:
        probe_steps = sim_cfg.num_frames * sim_cfg.steps_per_frame
        probe_threshold = float(args.penetration_z_floor) + float(args.penetration_probe_margin)
        blocked = set()
        for attempt in range(int(args.penetration_retries) + 1):
            worst = probe_floor_penetration(scene, objects, probe_steps, arrest_static_spin)
            bad = {i: z for i, z in worst.items() if z < probe_threshold}
            # Prediction for the recorded window; se3/link_poses.npy must match.
            penetration_meta["probe_min_dynamic_z"] = (
                round(min(worst.values()), 4) if worst else None)
            penetration_meta["attempts"].append(
                {"attempt": attempt,
                 "offenders": [{"object": i, "asset_id": objects[i].asset_id,
                                "source": objects[i].source, "min_z": round(z, 3)}
                               for i, z in sorted(bad.items())]})
            if not bad:
                break
            last = attempt == int(args.penetration_retries)
            for i in sorted(bad, reverse=True):
                obj = objects[i]
                blocked.add(obj.asset_id)
                if not remove_object(scene, obj):
                    continue
                objects.pop(i)
                SPAWN_TALLY["penetration_removed"] += 1
                # Free its footprint for the replacement.
                if obj.spawn_xy is not None and obj.spawn_xy in occupied:
                    occupied.remove(obj.spawn_xy)
                if last:
                    penetration_meta["dropped"] += 1
                    continue
                pool = [a for a in (rigid_assets if obj.kind == "rigid" else art_assets)
                        if a.asset_id not in blocked]
                if not pool:
                    penetration_meta["dropped"] += 1
                    continue
                repl = spawn_objects(
                    scene, rng,
                    pool if obj.kind == "rigid" else [],
                    [] if obj.kind == "rigid" else pool,
                    1 if obj.kind == "rigid" else 0,
                    0 if obj.kind == "rigid" else 1,
                    spawn_cfg.dynamic_box, spawn_cfg, vel_cfg, occupied, True, dynamic_mat,
                    joint_cfg=None if obj.kind == "rigid" else joint_cfg,
                    freeze_joints=freeze_joints,
                    weighted_articulated=weighted_articulated,
                    art_source_weights=scene_source_weights,
                    dynamic_rigid_motion=args.dynamic_rigid_motion,
                    force_xy=obj.spawn_xy,
                )
                objects.extend(repl)
                if repl:
                    penetration_meta["replaced"] += 1
                    SPAWN_TALLY["penetration_replaced"] += 1
                else:
                    penetration_meta["dropped"] += 1
        penetration_meta["blocked_assets"] = sorted(blocked)
        if blocked:
            print(f"Floor penetration: {len(blocked)} asset(s) blocked, "
                  f"{penetration_meta['replaced']} replaced, "
                  f"{penetration_meta['dropped']} dropped "
                  f"over {len(penetration_meta['attempts'])} probe(s)")

    all_links = []
    for obj in objects:
        all_links.extend(get_entities(obj.entity))

    camera = scene.add_camera("cam", cam_cfg.width, cam_cfg.height, cam_cfg.fovy, cam_cfg.near, cam_cfg.far)

    if cam_cfg.mode in ("linear", "linear_movement"):
        cam_positions, cam_lookats = sample_kubric_linear_trajectory(rng, sim_cfg.num_frames, cam_cfg, panning=False)
    elif cam_cfg.mode == "linear_movement_linear_lookat":
        cam_positions, cam_lookats = sample_kubric_linear_trajectory(rng, sim_cfg.num_frames, cam_cfg, panning=True)
    elif cam_cfg.mode == "fixed_random":
        fixed_pos = sample_point_in_half_sphere_shell(
            rng, cam_cfg.fixed_inner_radius, cam_cfg.fixed_outer_radius, cam_cfg.z_offset
        )
        cam_positions = [fixed_pos for _ in range(sim_cfg.num_frames)]
        cam_lookats = [cam_cfg.look_at for _ in range(sim_cfg.num_frames)]
    else:
        cam_positions = sample_orbit_camera_path(sim_cfg.num_frames, cam_cfg)
        cam_lookats = [cam_cfg.look_at for _ in range(sim_cfg.num_frames)]

    def update_camera(frame_idx: int):
        set_camera_pose(
            camera,
            np.array(cam_positions[frame_idx], dtype=np.float32),
            np.array(cam_lookats[frame_idx], dtype=np.float32),
        )

    update_camera(0)

    all_actor_ids = {get_actor_id(e) for e in all_links if get_actor_id(e) is not None}
    bbox_info = {}

    warmup_system = getattr(scene, "physx_system", None)
    warmup_blob = None
    if warmup_system is not None and hasattr(warmup_system, "pack"):
        try:
            warmup_blob = warmup_system.pack()
        except Exception as e:
            print(f"bbox warm-up: cannot snapshot physics ({e}); recording will start post-warm-up")

    for attempt in range(min(10, sim_cfg.num_frames)):
        camera.take_picture()
        pos = camera.get_picture("Position")
        seg = camera.get_picture("Segmentation")
        _, actor_labels = process_segmentation(seg)
        _, frame_bbox = compute_object_coordinates(pos, actor_labels, [o.entity for o in objects], camera.get_model_matrix())
        for aid, data in frame_bbox.items():
            if aid not in bbox_info:
                bbox_info[aid] = data
        if set(bbox_info.keys()) == all_actor_ids:
            break
        for _ in range(sim_cfg.steps_per_frame):
            scene.step()
        scene.update_render()

    if warmup_blob is not None:
        warmup_system.unpack(warmup_blob)
        scene.update_render()

    for aid in all_actor_ids - set(bbox_info.keys()):
        d = np.array([-0.05, -0.05, -0.05], dtype=np.float32)
        bbox_info[aid] = (d.copy(), -d.copy(), np.array([0.1, 0.1, 0.1]))

    num_frames = sim_cfg.num_frames
    all_rgb, all_depth, all_seg, all_obj_coords, all_normal = [], [], [], [], []
    num_links = len(all_links)
    all_bbox3d = [[] for _ in range(num_links)]
    all_quat = [[] for _ in range(num_links)]
    all_link_poses = [[] for _ in range(num_links)]
    all_cam_pos, all_cam_quat = [], []
    depth_min, depth_max = float("inf"), float("-inf")

    focal_length = 0.5 / np.tan(cam_cfg.fovy / 2)
    seg_palette = np.array([ImageColor.getrgb(c) for c in sorted(set(ImageColor.colormap.values()))], dtype=np.uint8)
    link_to_idx = {get_actor_id(link): i for i, link in enumerate(all_links)}
    rp_actor_entities = {get_actor_id(link): link for link in all_links if get_actor_id(link) is not None}
    if floor is not None and floor_actor_id is not None:
        rp_actor_entities[floor_actor_id] = floor
    rp_actor_ids = sorted(int(aid) for aid in rp_actor_entities.keys() if aid is not None)
    rp_pose_history = {aid: [] for aid in rp_actor_ids}

    link_asset_ids = []
    link_object_ids = []
    link_object_types = []
    link_is_dynamic = []
    link_sources = []
    object_asset_ids = []
    object_types = []
    object_is_dynamic = []
    object_sources = []

    for obj_idx, obj in enumerate(objects):
        object_asset_ids.append(obj.asset_id)
        object_types.append(obj.kind)
        object_is_dynamic.append(obj.is_dynamic)
        object_sources.append(obj.source)
        for link in get_entities(obj.entity):
            link_asset_ids.append(obj.asset_id)
            link_object_ids.append(obj_idx)
            link_object_types.append(obj.kind)
            link_is_dynamic.append(obj.is_dynamic)
            link_sources.append(obj.source)

    realized_source_counts: Dict[str, int] = {}
    for src in object_sources:
        realized_source_counts[src] = realized_source_counts.get(src, 0) + 1

    clamp_velocities = make_velocity_clamp(objects)

    for frame in range(num_frames):
        update_camera(frame)
        for step in range(sim_cfg.steps_per_frame):
            scene.step()
            # Clamp every substep.
            clamp_velocities()
            arrest_static_spin()
        scene.update_render()
        camera.take_picture()

        rgba = camera.get_picture("Color")
        position = camera.get_picture("Position")
        segmentation = camera.get_picture("Segmentation")
        normal = camera.get_picture("Normal")
        cam_mat = camera.get_model_matrix()

        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        all_rgb.append(rgb)

        _, actor_labels = process_segmentation(segmentation)
        all_seg.append(actor_labels)

        depth_raw = depth_from_position(position)
        depth = mask_dome_depth(depth_raw, actor_labels, dome_actor_ids)
        fg_depth = depth[depth > 0]
        if fg_depth.size > 0:
            depth_min = min(depth_min, float(fg_depth.min()))
            depth_max = max(depth_max, float(fg_depth.max()))
        all_depth.append(depth)

        if cotracker_mode:
            imageio.imwrite(os.path.join(out_cfg.output_dir, "frames", f"{frame:03d}.png"), rgb)
        else:
            imageio.imwrite(os.path.join(out_cfg.output_dir, "rgb", f"frame_rgb_{frame:03d}.png"), rgb)
            np.save(os.path.join(out_cfg.output_dir, "segmentation", f"frame_seg_{frame:03d}_actor.npy"), actor_labels)
            imageio.imwrite(
                os.path.join(out_cfg.output_dir, "segmentation", f"frame_seg_{frame:03d}_color.png"),
                colorize_segmentation(actor_labels, seg_palette),
            )
            np.save(os.path.join(out_cfg.output_dir, "depth", f"frame_depth_{frame:03d}.npy"), depth)
            depth_png = depth_to_png(depth)
            imageio.imwrite(os.path.join(out_cfg.output_dir, "depth", f"frame_depth_{frame:03d}.png"), depth_png)

            norm_u16 = ((normal[..., :3] * 0.5 + 0.5) * 65535).clip(0, 65535).astype(np.uint16)
            norm_u8 = ((normal[..., :3] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_cfg.output_dir, "normal", f"frame_normal_{frame:03d}.png"), norm_u8)
            all_normal.append(norm_u16)

            obj_coords, _ = compute_object_coordinates(position, actor_labels, [o.entity for o in objects], cam_mat)
            np.save(os.path.join(out_cfg.output_dir, "object_coordinates", f"frame_coords_{frame:03d}.npy"), obj_coords)
            all_obj_coords.append(obj_coords)

        for link in all_links:
            link_pose = link.get_pose().to_transformation_matrix()
            quat = pose_to_quaternion(link_pose)
            aid = get_actor_id(link)
            link_idx = link_to_idx.get(aid, 0)
            if aid in bbox_info:
                bmin, bmax, _ = bbox_info[aid]
                corners = np.array(list(itertools.product([bmin[0], bmax[0]], [bmin[1], bmax[1]], [bmin[2], bmax[2]])))
                corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
                corners_w = (link_pose @ corners_h.T).T[:, :3]
            else:
                corners_w = np.zeros((8, 3))
            all_bbox3d[link_idx].append(corners_w.tolist())
            all_quat[link_idx].append(quat)
            all_link_poses[link_idx].append(link_pose.tolist())

        for aid in rp_actor_ids:
            pose = rp_actor_entities[aid].get_pose().to_transformation_matrix()
            rp_pose_history[aid].append(np.array(pose, dtype=np.float64))

        all_cam_pos.append(cam_mat[:3, 3].tolist())
        all_cam_quat.append(pose_to_quaternion(cam_mat))

        if (frame + 1) % 6 == 0:
            print(f"  {frame + 1}/{num_frames}")

    actor_ids = [get_actor_id(link) for link in all_links]

    # Build annotation dict (needed by both formats for the tracking step)
    rp_translation_tol = 1e-3
    rp_rotation_tol = 1e-3
    actor_to_rp_seg, rp_segments = build_relative_pose_segmentation(
        rp_pose_history, translation_tol=rp_translation_tol, rotation_tol=rp_rotation_tol
    )

    annotation = {
        "kubric_format": {
            "metadata": {
                "depth_range": [float(depth_min), float(depth_max)],
                "seed": args.seed,
                "dome_radius": bg_cfg.dome_radius,
                "floor_actor_id": floor_actor_id,
                "dome_actor_ids": dome_actor_ids,
                "scene_mix_mode": getattr(args, "scene_mix_mode", "fixed"),
                "spawn_min_gap": float(spawn_gap),
                "spawn_xy_scale": float(spawn_xy),
                # requested vs placed, split by why an object was lost
                "spawn_tally": dict(SPAWN_TALLY),
                "penetration_retry": penetration_meta,
                "requested_rigid_frac": scene_rigid_frac,
                "source_mix_mode": source_mix_mode,
                "requested_source_weights": scene_source_weights,
                "single_source": scene_single_source,
                "light_level": light_level,
                "env_light_scale": env_scale_eff,
                "key_gain_mult": key_mult,
                "area_light_gain": args.area_light_gain,
                "rich_lighting": args.rich_lighting,
                "light_params": light_params,
                **(composition_meta or {}),
            },
            "camera": {
                "focal_length": float(focal_length),
                "sensor_width": 1.0,
                # Vertical FOV, also derivable from focal_length.
                "field_of_view": float(cam_cfg.fovy),
                "fov_deg": float(np.rad2deg(cam_cfg.fovy)),
                "fov_framing_scale": float(framing_scale),
                "fov_framing_alpha": float(framing_alpha),
                "positions": all_cam_pos,
                "quaternions": all_cam_quat,
            },
            "instances": {
                "bboxes_3d": all_bbox3d,
                "quaternions": all_quat,
            },
            "frame_count": num_frames,
            "height": cam_cfg.height,
            "width": cam_cfg.width,
            "num_links": num_links,
            "link_names": [link.get_name() for link in all_links],
            "actor_ids": actor_ids,
            "relative_pose_segmentation": {
                "folder": "seg_rp",
                "translation_tol": rp_translation_tol,
                "rotation_tol": rp_rotation_tol,
                "ground_actor_id": floor_actor_id,
                "num_segments": len(rp_segments),
                "actor_to_seg_id": [
                    {"actor_id": int(aid), "seg_id": int(seg_id)}
                    for aid, seg_id in sorted(actor_to_rp_seg.items())
                ],
                "segments": rp_segments,
            },
            "assets": {
                "objects": {
                    "asset_ids": object_asset_ids,
                    "types": object_types,
                    "is_dynamic": object_is_dynamic,
                    "sources": object_sources,
                },
                "links": {
                    "asset_ids": link_asset_ids,
                    "object_ids": link_object_ids,
                    "types": link_object_types,
                    "is_dynamic": link_is_dynamic,
                    "sources": link_sources,
                },
                "realized_source_counts": realized_source_counts,
            },
        }
    }

    if cotracker_mode:
        # CoTracker format: save intermediates for the tracking step.

        link_poses_arr = np.array(all_link_poses, dtype=np.float64)  # [L, F, 4, 4]
        link_actor_ids_arr = np.array(actor_ids, dtype=np.int32)     # [L]
        link_rp_seg_ids_arr = np.array(
            [int(actor_to_rp_seg.get(int(aid), 0)) for aid in actor_ids],
            dtype=np.int32,
        )                                                             # [L]
        # Preserve the original IDs, including scenes with more than 255 links/parts.
        rigid_map_rp_arr = segmentation_labels(np.stack([
            remap_actor_labels_to_seg_ids(labels, actor_to_rp_seg)
            for labels in all_seg
        ], axis=0))                                                   # [F, H, W]
        rigid_map_actor_arr = segmentation_labels(np.stack(all_seg, axis=0))

        se3_data = {
            "link_poses":      link_poses_arr,
            "link_actor_ids":  link_actor_ids_arr,
            "link_rp_seg_ids": link_rp_seg_ids_arr,
            "rigid_map_rp":    rigid_map_rp_arr,
            "rigid_map_actor": rigid_map_actor_arr,
        }

        seq_id = os.path.basename(out_cfg.output_dir)
        save_cotracker_generation(
            out_dir=out_cfg.output_dir,
            seq_id=seq_id,
            all_rgb=all_rgb,
            all_depth=all_depth,
            all_seg=all_seg,
            annotation=annotation,
            depth_min=depth_min,
            depth_max=depth_max,
            link_poses=all_link_poses,
            se3_data=se3_data,
            bboxes_3d=np.array(all_bbox3d, dtype=np.float32),
        )
    else:
        # full format: save all per-frame and stacked data
        np.save(os.path.join(out_cfg.output_dir, "video.npy"), np.stack(all_rgb))
        np.save(os.path.join(out_cfg.output_dir, "depth.npy"), np.stack(all_depth))
        np.save(os.path.join(out_cfg.output_dir, "segmentations.npy"), np.stack(all_seg))
        all_seg_rp = []
        for frame, actor_labels in enumerate(all_seg):
            seg_rp = remap_actor_labels_to_seg_ids(actor_labels, actor_to_rp_seg)
            np.save(os.path.join(out_cfg.output_dir, "seg_rp", f"frame_seg_rp_{frame:03d}.npy"), seg_rp)
            imageio.imwrite(
                os.path.join(out_cfg.output_dir, "seg_rp", f"frame_seg_rp_{frame:03d}_color.png"),
                colorize_segmentation(seg_rp, seg_palette),
            )
            all_seg_rp.append(seg_rp)
        np.save(os.path.join(out_cfg.output_dir, "segmentations_rp.npy"), np.stack(all_seg_rp))
        np.save(os.path.join(out_cfg.output_dir, "object_coordinates.npy"), np.stack(all_obj_coords))
        np.save(os.path.join(out_cfg.output_dir, "normal.npy"), np.stack(all_normal))
        np.save(os.path.join(out_cfg.output_dir, "bboxes_3d.npy"), np.array(all_bbox3d))
        np.save(os.path.join(out_cfg.output_dir, "quaternions.npy"), np.array(all_quat))
        np.save(os.path.join(out_cfg.output_dir, "link_poses.npy"), np.array(all_link_poses))

        imageio.mimsave(os.path.join(out_cfg.output_dir, out_cfg.video_name), all_rgb, fps=sim_cfg.fps)

        with open(os.path.join(out_cfg.output_dir, out_cfg.annotation_name), "w") as f:
            json.dump(annotation, f, indent=2, default=numpy_json_default)

    print(f"\nDone! Output: {out_cfg.output_dir}")
    print(f"Depth range: [{depth_min:.3f}, {depth_max:.3f}]")
    print(f"Saved: video {len(all_rgb)}x{cam_cfg.height}x{cam_cfg.width}")
    print(f"Links tracked: {num_links}")
    if not cotracker_mode:
        print(f"Relative-pose seg ids: {len(rp_segments)} (saved to seg_rp)")
        print(f"bboxes_3d shape: [{num_links}, {num_frames}, 8, 3]")


if __name__ == "__main__":
    main()
