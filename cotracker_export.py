"""CoTracker Kubric-format export for SAPIEN-generated data.

Converts full-format output into the format used by CoTracker training:
  {seq_id}.npy            pickle dict with coords, visibility, depth, segmentations, metadata, camera
  {seq_id}_trajs_2d.npy   (N, F, 2) 2D trajectories
  {seq_id}_visibility.npy (N, F) bool visibility
  {seq_id}_with_rank.npz  extrinsics, intrinsics, ranking
  frames/{f:03d}.png      RGB frames
  depths/{f:03d}.npy      (H, W, 1) uint16 depth
"""

import json
import os

import imageio.v2 as imageio
import numpy as np


def segmentation_labels(labels):
    """Preserve nonnegative segmentation IDs as int32."""
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("segmentation labels must be integers")
    if labels.size and (labels.min() < 0 or labels.max() > np.iinfo(np.int32).max):
        raise ValueError("segmentation labels must fit nonnegative int32")
    return labels.astype(np.int32, copy=False)


def numpy_json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Helpers
# =============================================================================

def quat_to_rot_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def depth_float_to_uint16(depth, depth_min, depth_max):
    """Scale float32 depth to uint16.  0 = invalid / no depth."""
    valid = depth > 0
    normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    uint16 = (normalized * 65534 + 1).clip(0, 65535).astype(np.uint16)
    uint16[~valid] = 0
    return uint16


def compute_intrinsics(focal_length, sensor_width, H, W):
    """Pixel-space camera intrinsic matrix (3x3)."""
    f_px = focal_length / sensor_width * W
    f_py = focal_length / sensor_width * H
    return np.array([
        [f_px, 0.0, W / 2.0],
        [0.0, f_py, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def compute_extrinsics(positions, quaternions):
    """World-to-OpenGL-camera matrices, shape (N, 3, 4).

    Recorded [w, x, y, z] quaternions rotate camera axes into world axes.
    Camera axes are +X right, +Y up, -Z forward. For projection with the
    positive-focal-length pixel intrinsics, convert these matrices using
    opengl_to_opencv_extrinsics first.
    """
    N = len(positions)
    extrinsics = np.zeros((N, 3, 4), dtype=np.float64)
    for i in range(N):
        R_cam = quat_to_rot_matrix(quaternions[i])    # camera-to-world rotation
        R_inv = R_cam.T                                # world-to-camera rotation
        t = -R_inv @ np.array(positions[i], dtype=np.float64)
        extrinsics[i, :3, :3] = R_inv
        extrinsics[i, :3, 3] = t
    return extrinsics


def opengl_to_opencv_extrinsics(extrinsics):
    """Convert (..., 3, 4) world-to-camera matrices to +Y down, +Z forward.

    Left-multiply by diag(1, -1, -1), flipping both rotation and translation.
    The input is preserved; the result is float64.
    """
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim < 2 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("extrinsics must have shape (..., 3, 4)")
    return extrinsics * np.array([1., -1., -1.])[:, None]


def compute_ranking(positions):
    """Frame-pair ranking by camera baseline distance, shape (N, N)."""
    pos = np.array(positions, dtype=np.float64)
    # Pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    dists = np.linalg.norm(diff, axis=-1)              # (N, N)
    ranking = np.argsort(dists, axis=1).astype(np.int64)
    return ranking


# =============================================================================
# Generation-phase export
# =============================================================================

# Bulky per-frame arrays that belong in the .npy payload, not in metadata.json.
_METADATA_EXCLUDE = frozenset({"forward_flow_range", "backward_flow_range"})


def save_cotracker_generation(out_dir, seq_id, all_rgb, all_depth, all_seg,
                              annotation, depth_min, depth_max, link_poses,
                              se3_data=None, bboxes_3d=None):
    """Save CoTracker-format output from the generation step.

    Writes frames/, depths/, depth_f32.npy, bboxes_3d.npy, se3/ and the intermediate
    files needed by the tracking step.
    """
    frames_dir = os.path.join(out_dir, "frames")
    depths_dir = os.path.join(out_dir, "depths")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    for f, (rgb, depth) in enumerate(zip(all_rgb, all_depth)):
        imageio.imwrite(os.path.join(frames_dir, f"{f:03d}.png"), rgb)
        d_uint16 = depth_float_to_uint16(depth, depth_min, depth_max)
        np.save(os.path.join(depths_dir, f"{f:03d}.npy"),
                d_uint16[:, :, np.newaxis])

    # Intermediate files for the tracking step
    np.save(os.path.join(out_dir, "video.npy"), np.stack(all_rgb))
    # depth_f32.npy is kept after cleanup.
    np.save(os.path.join(out_dir, "depth_f32.npy"),
            np.stack(all_depth).astype(np.float32))
    np.save(os.path.join(out_dir, "segmentations.npy"), np.stack(all_seg))
    np.save(os.path.join(out_dir, "link_poses.npy"), np.array(link_poses))
    with open(os.path.join(out_dir, "rigid_box_drop_annotations.json"), "w") as fp:
        json.dump(annotation, fp, default=numpy_json_default)

    if bboxes_3d is not None:
        np.save(os.path.join(out_dir, "bboxes_3d.npy"),
                np.asarray(bboxes_3d, dtype=np.float32))

    # SE(3) ground-truth files consumed by downstream training pipelines.
    if se3_data is not None:
        se3_dir = os.path.join(out_dir, "se3")
        os.makedirs(se3_dir, exist_ok=True)
        np.save(os.path.join(se3_dir, "link_poses.npy"), se3_data["link_poses"])
        np.save(os.path.join(se3_dir, "link_actor_ids.npy"), se3_data["link_actor_ids"])
        np.save(os.path.join(se3_dir, "link_rp_seg_ids.npy"), se3_data["link_rp_seg_ids"])
        np.save(os.path.join(se3_dir, "rigid_map_rp.npy"),
                segmentation_labels(se3_data["rigid_map_rp"]))
        np.save(os.path.join(se3_dir, "rigid_map_actor.npy"),
                segmentation_labels(se3_data["rigid_map_actor"]))


# =============================================================================
# Tracking-phase export
# =============================================================================

def save_cotracker_tracking(out_dir, seq_id, target_points, occluded,
                            rel_depth, annotation, target_3d=None,
                            abs_depth=None):
    """Assemble and save CoTracker-format tracking files.

    Loads depth.npy and segmentations.npy written during generation. If given,
    ``target_3d`` (N,F,3) and ``abs_depth`` (N,F) are also saved as float32.
    """
    kubric = annotation["kubric_format"]
    H = kubric["height"]
    W = kubric["width"]
    F = kubric["frame_count"]
    depth_min, depth_max = kubric["metadata"]["depth_range"]
    positions = np.array(kubric["camera"]["positions"], dtype=np.float32)
    quaternions = np.array(kubric["camera"]["quaternions"], dtype=np.float32)
    focal_length = kubric["camera"]["focal_length"]
    sensor_width = kubric["camera"]["sensor_width"]

    # Load intermediates (depth_f32 is kept permanently, not cleaned up).
    depth_stacked = np.load(os.path.join(out_dir, "depth_f32.npy"))   # (F, H, W) float32
    seg_stacked = np.load(os.path.join(out_dir, "segmentations.npy"))  # (F, H, W)

    # Convert depth to uint16 with trailing dim
    depth_uint16 = depth_float_to_uint16(depth_stacked, depth_min, depth_max)
    depth_ct = depth_uint16[:, :, :, np.newaxis]    # (F, H, W, 1)

    # Actor IDs can exceed 255 in scenes with many articulated links.
    seg_ct = segmentation_labels(seg_stacked)[:, :, :, np.newaxis]  # (F, H, W, 1)

    num_instances = int(seg_stacked.max())

    # Compute field of view from focal length
    fovy = 2.0 * np.arctan(0.5 / focal_length)

    # Tracking data conversions
    coords = target_points.astype(np.float32)         # (N, F, 2)
    visibility = (~occluded).astype(bool)              # (N, F)
    coords_depth = rel_depth.astype(np.float32)        # (N, F)

    # --- Main pickle dict ---
    data = {
        "coords": coords,
        "coords_depth": coords_depth,
        "visibility": visibility,
        "depth": depth_ct,
        "segmentations": seg_ct,
    }
    if abs_depth is not None:
        data["coords_depth_abs"] = abs_depth.astype(np.float32)   # (N, F) metres
    if target_3d is not None:
        data["coords_3d"] = target_3d.astype(np.float32)          # (N, F, 3) world
    data.update({
        "metadata": {
            "video_name": seq_id.encode() if isinstance(seq_id, str) else seq_id,
            "width": np.int32(W),
            "height": np.int32(H),
            "num_frames": np.int32(F),
            "num_instances": np.uint16(num_instances),
            "forward_flow_range": np.array([0.0, 0.0], dtype=np.float32),
            "backward_flow_range": np.array([0.0, 0.0], dtype=np.float32),
            "depth_range": np.array([depth_min, depth_max], dtype=np.float32),
        },
        "camera": {
            "positions": positions,
            "quaternions": quaternions,
            "field_of_view": np.float32(fovy),
            "focal_length": np.float32(focal_length),
            "sensor_width": np.float32(sensor_width),
        },
    })
    np.save(os.path.join(out_dir, f"{seq_id}.npy"), data)

    # --- Convenience files ---
    np.save(os.path.join(out_dir, f"{seq_id}_trajs_2d.npy"), coords)
    np.save(os.path.join(out_dir, f"{seq_id}_visibility.npy"), visibility)
    if target_3d is not None:
        np.save(os.path.join(out_dir, f"{seq_id}_trajs_3d.npy"),
                target_3d.astype(np.float32))
    if abs_depth is not None:
        np.save(os.path.join(out_dir, f"{seq_id}_coords_depth_abs.npy"),
                abs_depth.astype(np.float32))

    # --- Camera matrices npz ---
    extrinsics = compute_extrinsics(positions, quaternions)
    intrinsics = compute_intrinsics(focal_length, sensor_width, H, W)
    ranking = compute_ranking(positions)
    frame_names = np.array([f"{i:03d}" for i in range(F)])

    np.savez(os.path.join(out_dir, f"{seq_id}_with_rank.npz"),
             names=frame_names,
             extrinsics=extrinsics,
             extrinsics_convention="opengl",
             shared_intrinsics=intrinsics,
             ranking=ranking)

    # Asset and link metadata, kept after cleanup_intermediates.
    meta_out = {
        "num_links": kubric.get("num_links"),
        "link_names": kubric.get("link_names"),
        "actor_ids": kubric.get("actor_ids"),
        "assets": kubric.get("assets"),
        "relative_pose_segmentation": kubric.get("relative_pose_segmentation"),
        "metadata": {k: v for k, v in (kubric.get("metadata") or {}).items()
                     if k not in _METADATA_EXCLUDE},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fp:
        json.dump(meta_out, fp, default=numpy_json_default)

    print(f"CoTracker export: {seq_id}.npy  coords {coords.shape}  "
          f"visibility {visibility.shape}  depth {depth_ct.shape}")


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_intermediates(out_dir):
    """Remove full-format intermediate files after CoTracker export.

    The ``se3/`` subdirectory (SE(3) GT) and ``depth_f32.npy``,
    ``bboxes_3d.npy``, ``metadata.json`` (asset / link metadata written by
    ``save_cotracker_tracking``) are intentionally preserved.
    """
    for name in ["video.npy", "segmentations.npy",
                 "link_poses.npy",
                 "rigid_box_drop_annotations.json"]:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
