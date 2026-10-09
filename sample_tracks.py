"""Point tracking for SAPIEN scenes.

Uses depth unprojection + per-link poses to track 2D points across video frames.
Outputs Kubric-compatible query_points, target_points, occluded, relative_depth.
"""

import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

from cotracker_export import save_cotracker_tracking, cleanup_intermediates


# =============================================================================
# Data Loading
# =============================================================================

def load_sapien_data(data_dir):
    """Load SAPIEN data including per-link poses for tracking."""
    with open(os.path.join(data_dir, "rigid_box_drop_annotations.json"), 'r') as f:
        annotation = json.load(f)

    kubric_data = annotation['kubric_format']
    depth_path = os.path.join(data_dir, "depth_f32.npy")
    if not os.path.isfile(depth_path):
        depth_path = os.path.join(data_dir, "depth.npy")

    result = {
        'annotation': annotation,
        'video': np.load(os.path.join(data_dir, "video.npy")),
        'depth': np.load(depth_path),
        'segmentations': np.load(os.path.join(data_dir, "segmentations.npy")),
        'camera': {
            'focal_length': kubric_data['camera']['focal_length'],
            'sensor_width': kubric_data['camera']['sensor_width'],
            'positions': np.array(kubric_data['camera']['positions']),
            'quaternions': np.array(kubric_data['camera']['quaternions']),
        },
        'metadata': {
            'depth_range': np.array(kubric_data['metadata']['depth_range']),
            'num_links': kubric_data.get('num_links', 1),
            'link_names': kubric_data.get('link_names', []),
            'actor_ids': kubric_data.get('actor_ids', []),
            'dome_radius': kubric_data['metadata'].get('dome_radius'),
            'floor_actor_id': kubric_data['metadata'].get('floor_actor_id'),
            'dome_actor_ids': kubric_data['metadata'].get('dome_actor_ids', []),
        },
        'assets': kubric_data.get('assets', {}),
    }

    link_poses_path = os.path.join(data_dir, "link_poses.npy")
    if os.path.exists(link_poses_path):
        result['link_poses'] = np.load(link_poses_path)
    else:
        raise FileNotFoundError(
            f"link_poses.npy not found in {data_dir}. "
            "Re-run render_scene.py to produce it.")

    return result


# =============================================================================
# Camera and Projection
# =============================================================================

def quat_to_rot_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def get_camera_matrices(focal_length, positions, quaternions, sensor_width, input_size, num_frames):
    """Build camera intrinsic and world matrices."""
    intrinsics, matrix_world = [], []
    f_x = focal_length / sensor_width
    f_y = f_x * input_size[0] / input_size[1]

    for i in range(num_frames):
        intrinsics.append(np.array([[f_x, 0., -0.5], [0., -f_y, -0.5], [0., 0., -1.]], dtype=np.float32))
        R = quat_to_rot_matrix(quaternions[i])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3], T[:3, 3] = R, positions[i]
        matrix_world.append(T)

    return np.stack(intrinsics), np.stack(matrix_world)


def project_point(cam, point3d, num_frames):
    """Project 3D world points to normalised image coordinates [0,1]."""
    homo_transform = np.linalg.inv(cam['matrix_world'])
    homo_intrinsics = np.concatenate([cam['intrinsics'], np.zeros((num_frames, 3, 1))], axis=2)

    point4d = np.concatenate([point3d, np.ones((num_frames, point3d.shape[1], 1))], axis=2)
    projected = np.einsum('fpi,fij->fpj', point4d, homo_transform.transpose(0, 2, 1))
    projected = np.einsum('fpi,fij->fpj', projected, homo_intrinsics.transpose(0, 2, 1))

    image_coords = projected / projected[:, :, 2:3]
    return np.concatenate([image_coords[:, :, :2], np.sign(projected[:, :, 2:])], axis=2)


def unproject(coord, cam, depth):
    """Unproject 2D pixel coordinates to 3D world coordinates."""
    H, W = depth.shape
    coord_xy = coord[:, ::-1].astype(np.float32)
    projected_pt = np.concatenate([(coord_xy + 0.5) / np.array([W, H]), np.ones((len(coord_xy), 1))], axis=1)

    y_idx = np.clip(coord[:, 0].astype(int), 0, H - 1)
    x_idx = np.clip(coord[:, 1].astype(int), 0, W - 1)
    depths = depth[y_idx, x_idx]

    camera_plane = projected_pt @ np.linalg.inv(cam['intrinsics'].T)
    camera_ball = camera_plane * (-depths[:, None] / camera_plane[:, 2:3])
    camera_ball_4d = np.concatenate([camera_ball, np.ones((len(camera_ball), 1))], axis=1)

    points_3d = camera_ball_4d @ cam['matrix_world'].T
    return points_3d[:, :3] / points_3d[:, 3:4]


# =============================================================================
# Occlusion
# =============================================================================

def estimate_occlusion(depth_map, segments, x, y, num_frames, expected_depth, expected_seg):
    """Estimate occlusion by depth and segmentation mismatch."""
    x, y = x - 0.5, y - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    H, W = depth_map.shape[1:3]
    x0, x1 = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)
    y0, y1 = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)
    fr = np.arange(num_frames)[:, None]

    d = [depth_map[fr, y0, x0], depth_map[fr, y1, x0],
         depth_map[fr, y0, x1], depth_map[fr, y1, x1]]
    d = [np.where(np.isnan(di), np.inf, di) for di in d]
    depth = np.where(np.isinf(np.maximum.reduce(d)), np.nan, np.maximum.reduce(d))

    s = [segments[fr, y0, x0], segments[fr, y1, x0],
         segments[fr, y0, x1], segments[fr, y1, x1]]
    seg_id = expected_seg[None, :] if expected_seg.ndim == 1 else expected_seg
    seg_occluded = (s[0] != seg_id) & (s[1] != seg_id) & (s[2] != seg_id) & (s[3] != seg_id)

    thresh = expected_depth[None, :] * 0.95 if expected_depth.ndim == 1 else expected_depth * 0.95
    depth_occluded = np.where(np.isnan(depth), True, np.less(depth, thresh))

    return (depth_occluded | seg_occluded).T


# =============================================================================
# Point Sampling
# =============================================================================

def farthest_point_sample(coords_2d, n):
    """Greedily select *n* points from *coords_2d* (N,2) with maximal spread."""
    N = len(coords_2d)
    if N <= n:
        return np.arange(N)
    selected = [np.random.randint(N)]
    dists = np.full(N, np.inf)
    for _ in range(n - 1):
        d = np.linalg.norm(coords_2d - coords_2d[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return np.array(selected)


def get_num_to_sample(counts, max_seg_id, total, object_seg_ids,
                      min_per_object=10, bg_frac=0.1, max_seg_share=3.0):
    """Allocate tracking points across segments.

    Each visible object segment gets at least *min_per_object* points.
    Background gets at most *bg_frac* of the total.  No single segment may
    exceed *max_seg_share* × fair-share to prevent one large part from
    monopolising the budget.
    """
    result = np.zeros(max_seg_id, dtype=int)

    obj_segs = [sid for sid in object_seg_ids if sid < max_seg_id and counts[sid] > 0]
    bg_segs = [sid for sid in range(max_seg_id)
               if sid not in object_seg_ids and counts[sid] > 0]

    bg_budget = int(total * bg_frac)
    min_obj_points = min(len(obj_segs), total) if obj_segs else 0
    if total - bg_budget < min_obj_points:
        bg_budget = max(0, total - min_obj_points)
    obj_budget = total - bg_budget

    # --- background ---
    if bg_segs and bg_budget > 0:
        bg_counts = np.array([counts[sid] for sid in bg_segs])
        bg_total = bg_counts.sum()
        if bg_total > 0:
            for i, sid in enumerate(bg_segs):
                result[sid] = max(1, int(bg_budget * bg_counts[i] / bg_total))
            bg_allocated = sum(result[sid] for sid in bg_segs)
            if bg_allocated > bg_budget:
                scale = bg_budget / bg_allocated
                for sid in bg_segs:
                    result[sid] = max(1, int(result[sid] * scale))

    # --- objects ---
    if obj_segs and obj_budget > 0:
        n_segs = len(obj_segs)
        obj_counts = np.array([counts[sid] for sid in obj_segs])

        fair_share = max(1, obj_budget // max(n_segs, 1))
        max_per_seg = max(min_per_object + 1, int(fair_share * max_seg_share))

        alloc = np.minimum(obj_counts, min_per_object).astype(int)
        alloc_sum = int(alloc.sum())

        if alloc_sum > obj_budget:
            scale = obj_budget / max(alloc_sum, 1)
            alloc = np.floor(alloc * scale).astype(int)
            if obj_budget >= n_segs:
                alloc = np.maximum(alloc, 1)
            while alloc.sum() > obj_budget:
                idx = int(np.argmax(alloc))
                if alloc[idx] > 0:
                    alloc[idx] -= 1
                else:
                    break
        else:
            remaining = obj_budget - alloc_sum
            if remaining > 0:
                headroom = np.maximum(np.minimum(obj_counts, max_per_seg) - alloc, 0)
                total_headroom = headroom.sum()
                if total_headroom > 0:
                    extra = np.floor(remaining * headroom / total_headroom).astype(int)
                    alloc += np.minimum(extra, headroom)

        while alloc.sum() < obj_budget:
            headroom = np.maximum(np.minimum(obj_counts, max_per_seg) - alloc, 0)
            if headroom.sum() == 0:
                headroom = np.maximum(obj_counts - alloc, 0)
                if headroom.sum() == 0:
                    break
            idx = int(np.argmax(headroom))
            alloc[idx] += 1

        for i, sid in enumerate(obj_segs):
            result[sid] = int(alloc[i])

    return result


# =============================================================================
# Main Tracking (depth + per-link poses)
# =============================================================================

def track_points(depth, depth_range, segmentations, link_poses, actor_ids,
                 cam_focal_length, cam_positions, cam_quaternions, cam_sensor_width,
                 window, tracks_to_sample=256, sampling_stride=2,
                 sample_frame="max_visibility", min_per_object=5, bg_frac=0.1,
                 dome_actor_ids=None, floor_actor_id=None):
    """Track points across frames via depth unprojection + per-link poses.

    For *object* pixels the pipeline is:
      1. Unproject pixel → 3D world coord (using depth at sampling frame).
      2. Convert to link-local coord:  p_local = T_sample⁻¹ · p_world.
      3. For every frame f:  p_world_f = T_f · p_local  → project to 2D.

    For *background* (floor) pixels the point is static in world space and
    simply re-projected through each frame's camera.
    """

    # -- depth ----------------------------------------------------------------
    depth_min, depth_max = depth_range
    if depth.dtype in (np.float32, np.float64):
        depth_map = depth.astype(np.float32)
        depth_map[depth_map <= 0] = np.nan
    else:
        invalid_mask = depth == 0
        depth_map = depth_min + depth.astype(np.float32) * (depth_max - depth_min) / 65535
        depth_map[invalid_mask] = np.nan

    num_frames, H, W = depth_map.shape
    input_size = (H, W)

    # -- cameras --------------------------------------------------------------
    intrinsics, matrix_world = get_camera_matrices(
        cam_focal_length, cam_positions, cam_quaternions,
        cam_sensor_width, input_size, num_frames)
    camera = {'intrinsics': intrinsics, 'matrix_world': matrix_world}
    world_to_cam = np.linalg.inv(matrix_world)            # (F, 4, 4)

    # -- segment mappings -----------------------------------------------------
    valid_actor_ids = [int(aid) for aid in actor_ids if aid is not None]
    seg_to_link = ({int(aid): i for i, aid in enumerate(actor_ids) if aid is not None}
                   if valid_actor_ids else {})
    object_seg_ids = set(seg_to_link.keys())
    dome_seg_ids = set(dome_actor_ids or [])

    max_seg_id = max(
        int(segmentations.max()) + 1,
        (max(valid_actor_ids) + 1) if valid_actor_ids else 1,
    )

    # -- select sampling frame(s) -----------------------------------------------
    def _select_complementary_frames(segs, obj_ids, n_frames):
        """Select complementary frames starting from frame 0.

        Frame 0 is always first (users typically inspect the first frame),
        then greedily add frames that cover the most new segments until
        all visible segments are covered.
        """
        per_frame = []
        for f in range(n_frames):
            visible = set(int(s) for s in np.unique(segs[f]) if int(s) in obj_ids)
            per_frame.append(visible)
        all_segs = set().union(*per_frame) if per_frame else set()
        selected = [0]
        covered = per_frame[0].copy()
        while covered < all_segs:
            best_f, best_new = -1, -1
            for f in range(n_frames):
                if f in selected:
                    continue
                new_count = len(per_frame[f] - covered)
                if new_count > best_new:
                    best_new, best_f = new_count, f
            if best_new <= 0:
                break
            selected.append(best_f)
            covered |= per_frame[best_f]
        return selected

    use_all_frames = sample_frame == "all"

    if use_all_frames:
        frame_indices = list(range(num_frames))
    elif sample_frame == "multi" and valid_actor_ids:
        frame_indices = _select_complementary_frames(
            segmentations, object_seg_ids, num_frames)
    elif sample_frame == "max_visibility" and valid_actor_ids:
        best_f, best_count = 0, -1
        for f in range(num_frames):
            seg_f = segmentations[f]
            count = sum(int((seg_f == sid).sum()) for sid in object_seg_ids)
            if count > best_count:
                best_count, best_f = count, f
        frame_indices = [best_f]
    else:
        fi = 0 if sample_frame == "first" else np.random.randint(0, num_frames)
        frame_indices = [fi]

    n_rounds = len(frame_indices)
    print(f"  Sampling frames: {frame_indices} (strategy={sample_frame})")

    # -- auto-adjust stride so the grid has enough candidates ----------------
    win_h = window[2] - window[0]
    win_w = window[3] - window[1]
    per_round_tracks = max(1, tracks_to_sample // max(n_rounds, 1))
    min_candidates = per_round_tracks * 4
    max_grid = win_h * win_w
    if max_grid // (sampling_stride ** 2) < min_candidates and sampling_stride > 1:
        old_stride = sampling_stride
        sampling_stride = max(1, int(np.sqrt(max_grid / min_candidates)))
        print(f"  Auto-adjusted stride {old_stride} → {sampling_stride} "
              f"(need ≥{min_candidates} grid cells for {per_round_tracks} tracks/round)")

    # ======================================================================
    # Sampling loop
    # ======================================================================
    if not use_all_frames:
        _covered = set()
        _exclusive_per_round = []
        for fi in frame_indices:
            visible = set(int(s) for s in np.unique(segmentations[fi])
                          if int(s) in object_seg_ids and int(s) not in dome_seg_ids)
            exclusive = visible - _covered
            _exclusive_per_round.append(len(exclusive))
            _covered |= visible
        reserved_per_round = [0] * n_rounds
        for ri in range(1, n_rounds):
            reserved_per_round[ri] = min(
                _exclusive_per_round[ri] * min_per_object * 5,
                tracks_to_sample // (n_rounds * 2))

    already_sampled_segs = set()
    chosen_list, reproj_list, occ_list, depth_list, wc_list = [], [], [], [], []
    seg_stats = []
    remaining_budget = tracks_to_sample

    for round_idx, frame_idx in enumerate(frame_indices):
        if remaining_budget <= 0:
            break

        if use_all_frames:
            frame_budget = tracks_to_sample // n_rounds
            if round_idx == n_rounds - 1:
                frame_budget = remaining_budget
        elif round_idx < n_rounds - 1:
            future_reserved = sum(reserved_per_round[round_idx + 1:])
            frame_budget = max(min_per_object, remaining_budget - future_reserved)
        else:
            frame_budget = remaining_budget

        frame_budget = min(frame_budget, remaining_budget)
        if not use_all_frames or round_idx < 2 or round_idx == n_rounds - 1:
            print(f"\n  --- Round {round_idx+1}/{n_rounds}: "
                  f"frame {frame_idx}, budget {frame_budget} ---")

        # -- sampling grid for this frame ------------------------------------
        start_y = window[0] + np.random.randint(0, min(sampling_stride, win_h))
        start_x = window[1] + np.random.randint(0, min(sampling_stride, win_w))
        seg_box = segmentations[frame_idx,
                                start_y:window[2]:sampling_stride,
                                start_x:window[3]:sampling_stride]

        yy, xx = np.meshgrid(np.arange(start_y, window[2], sampling_stride),
                             np.arange(start_x, window[3], sampling_stride),
                             indexing='ij')
        zz = np.full_like(yy, frame_idx)
        pix_coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

        grid_y = np.clip(pix_coords[:, 1].astype(int), 0, H - 1)
        grid_x = np.clip(pix_coords[:, 2].astype(int), 0, W - 1)
        grid_depth = depth_map[frame_idx, grid_y, grid_x]

        # -- per-segment pixel counts (skip dome; skip already-sampled in non-all modes)
        seg_flat = seg_box.ravel()
        unique_segs = np.unique(seg_flat)
        counts_arr = np.zeros(max_seg_id, dtype=int)
        for sid in unique_segs:
            sid = int(sid)
            if sid in dome_seg_ids or not (0 <= sid < max_seg_id):
                continue
            if not use_all_frames and sid in already_sampled_segs:
                continue
            seg_mask = seg_flat == sid
            counts_arr[sid] = int(np.isfinite(grid_depth[seg_mask]).sum())

        # -- allocate budget (with redistribution) ---------------------------
        num_to_sample = get_num_to_sample(
            counts_arr, max_seg_id, frame_budget,
            object_seg_ids, min_per_object=min_per_object, bg_frac=bg_frac)

        for _pass in range(3):
            freed = 0
            for sid in range(max_seg_id):
                if num_to_sample[sid] > counts_arr[sid]:
                    freed += num_to_sample[sid] - counts_arr[sid]
                    num_to_sample[sid] = counts_arr[sid]
            if freed == 0:
                break
            eligible = [(sid, counts_arr[sid] - num_to_sample[sid])
                        for sid in range(max_seg_id)
                        if counts_arr[sid] > num_to_sample[sid]]
            if not eligible:
                break
            total_headroom = sum(h for _, h in eligible)
            for sid, headroom in eligible:
                give = min(headroom, max(1, int(freed * headroom / total_headroom)))
                num_to_sample[sid] += give
                freed -= give
                if freed <= 0:
                    break

        total_candidates = int(counts_arr.sum())
        total_alloc = int(num_to_sample.sum())
        if not use_all_frames or round_idx < 2 or round_idx == n_rounds - 1:
            print(f"    Grid: {len(seg_flat)} cells, {total_candidates} candidates "
                  f"(stride={sampling_stride}), alloc {total_alloc}/{frame_budget}")

        # -- camera at this sampling frame -----------------------------------
        cam_f = {'intrinsics': intrinsics[frame_idx],
                 'matrix_world': matrix_world[frame_idx]}

        for seg_id in range(max_seg_id):
            if num_to_sample[seg_id] == 0:
                continue

            is_object = seg_id in seg_to_link
            mask = seg_flat == seg_id
            if not np.any(mask):
                continue

            valid_depth_mask = mask & np.isfinite(grid_depth)
            if not np.any(valid_depth_mask):
                continue

            pt_coords = pix_coords[valid_depth_mask]
            n_candidates = len(pt_coords)

            n = min(n_candidates, num_to_sample[seg_id])
            idx = farthest_point_sample(pt_coords[:, 1:].astype(np.float32), n)
            pt_coords = pt_coords[idx]

            # -- unproject to 3D world coords --------------------------------
            pt_world = unproject(pt_coords[:, 1:], cam_f, depth_map[frame_idx])
            finite = np.all(np.isfinite(pt_world), axis=1)
            pt_world, pt_coords = pt_world[finite], pt_coords[finite]
            N_pts = len(pt_world)
            if N_pts == 0:
                seg_stats.append((seg_id, "obj" if is_object else "bg",
                                  n_candidates, num_to_sample[seg_id], n, 0,
                                  frame_idx))
                continue

            # -- build per-frame world coords --------------------------------
            if is_object:
                link_idx = seg_to_link[seg_id]
                T_sample = link_poses[link_idx, frame_idx]
                T_sample_inv = np.linalg.inv(T_sample)

                pt_h = np.concatenate([pt_world, np.ones((N_pts, 1))], axis=1)
                pt_local = (pt_h @ T_sample_inv.T)[:, :3]

                world_coords = np.zeros((num_frames, N_pts, 3), dtype=np.float32)
                pt_local_h = np.concatenate([pt_local, np.ones((N_pts, 1))], axis=1)
                for f in range(num_frames):
                    T_f = link_poses[link_idx, f]
                    world_coords[f] = (pt_local_h @ T_f.T)[:, :3]
            else:
                world_coords = np.tile(pt_world[None], [num_frames, 1, 1])

            # -- project to 2D ----------------------------------------------
            projections = project_point(camera, world_coords, num_frames)
            reproj = projections.transpose(1, 0, 2)     # (N, F, 3)

            # -- camera-space Z-depth for occlusion --------------------------
            pt_4d = np.concatenate(
                [world_coords, np.ones((*world_coords.shape[:2], 1))], axis=2)
            pt_cam = np.einsum('fpi,fji->fpj', pt_4d, world_to_cam)
            z_depth = -pt_cam[:, :, 2]                   # (F, N)

            # -- occlusion ---------------------------------------------------
            occluded = reproj[:, :, 2] < 0               # behind camera
            reproj_px = reproj[:, :, :2].copy()
            reproj_px[:, :, 0] *= W
            reproj_px[:, :, 1] *= H

            pt_seg = np.full(N_pts, seg_id, dtype=int)
            for f in range(num_frames):
                vis = ~occluded[:, f]
                if not np.any(vis):
                    continue
                occ = estimate_occlusion(
                    depth_map[None, f], segmentations[None, f],
                    reproj_px[vis, f, 0][None], reproj_px[vis, f, 1][None],
                    1,
                    z_depth[f, vis][None],
                    pt_seg[vis],
                )
                occluded[vis, f] = occ[:, 0]

            # window bounds
            y_min, x_min, y_max, x_max = window
            out = ((reproj_px[:, :, 0] < x_min) | (reproj_px[:, :, 0] >= x_max) |
                   (reproj_px[:, :, 1] < y_min) | (reproj_px[:, :, 1] >= y_max))
            occluded |= out

            if is_object:
                wc = world_coords.transpose(1, 0, 2)    # (N, F, 3)
                for f in range(num_frames):
                    occluded[:, f] |= ~np.all(np.isfinite(wc[:, f]), axis=1)
                    occluded[:, f] |= (np.linalg.norm(wc[:, f], axis=1) > 50.0)
                    if f > 0:
                        jump = np.linalg.norm(wc[:, f] - wc[:, f - 1], axis=1)
                        occluded[:, f] |= (jump > 5.0)

            seg_stats.append((seg_id, "obj" if is_object else "bg",
                              n_candidates, num_to_sample[seg_id], n, N_pts,
                              frame_idx))
            already_sampled_segs.add(seg_id)
            remaining_budget -= N_pts
            chosen_list.append(pt_coords)
            reproj_list.append(reproj_px)
            occ_list.append(occluded)
            depth_list.append(z_depth.T)                 # (N, F)
            wc_list.append(world_coords.transpose(1, 0, 2))  # (N, F, 3)

        round_sampled = sum(s[5] for s in seg_stats
                            if s[6] == frame_idx)
        if not use_all_frames or round_idx < 2 or round_idx == n_rounds - 1:
            print(f"    Round {round_idx+1} sampled: {round_sampled}, "
                  f"remaining budget: {remaining_budget}")

    # -- per-segment diagnostics -----------------------------------------------
    if seg_stats and not use_all_frames:
        print("\n  Per-segment sampling:")
        for sid, stype, cand, alloc, fps, final, sf in seg_stats:
            label = f"seg {sid:>3d} ({stype})"
            print(f"    {label}: {cand:>5d} cand, {alloc:>5d} alloc, "
                  f"{fps:>5d} FPS, {final:>5d} unproj [frame {sf}]")
    elif seg_stats:
        # Summarise for "all" mode
        per_seg_total = {}
        for sid, stype, _, _, _, final, _ in seg_stats:
            per_seg_total.setdefault(sid, [stype, 0])
            per_seg_total[sid][1] += final
        print(f"\n  Segment totals ({len(per_seg_total)} segments):")
        for sid in sorted(per_seg_total):
            stype, total = per_seg_total[sid]
            print(f"    seg {sid:>3d} ({stype}): {total:>5d} points")

    # -- assemble -------------------------------------------------------------
    if not reproj_list:
        z = np.zeros
        return (z((tracks_to_sample, 3)),
                z((tracks_to_sample, num_frames, 2)),
                z((tracks_to_sample, num_frames), dtype=bool),
                z((tracks_to_sample, num_frames)),
                z((tracks_to_sample, num_frames, 3)),
                z((tracks_to_sample, num_frames)))

    all_reproj = np.concatenate(reproj_list)
    all_occ = np.concatenate(occ_list)
    all_depth = np.concatenate(depth_list)
    all_wc = np.concatenate(wc_list)       # (total_N, F, 3)
    chosen = np.concatenate(chosen_list)

    # drop always-occluded
    not_always_occ = ~all_occ.all(axis=1)
    n_before = len(all_reproj)
    n_survive = int(not_always_occ.sum())
    print(f"  Points: {n_before} sampled → {n_survive} survive "
          f"(dropped {n_before - n_survive} always-occluded)")
    if n_survive > 0:
        all_reproj = all_reproj[not_always_occ]
        all_occ = all_occ[not_always_occ]
        all_depth = all_depth[not_always_occ]
        all_wc = all_wc[not_always_occ]
        chosen = chosen[not_always_occ]

    # normalise query coords
    wd = np.array([0., window[0], window[1],
                   num_frames, window[2], window[3]], dtype=np.float32)
    mult = np.array([num_frames, H, W], dtype=np.float32)
    chosen_norm = ((chosen.astype(np.float32) + np.array([0., 0.5, 0.5]) - wd[:3])
                   / (wd[3:] - wd[:3]) * mult)

    # normalise reproj to window
    y_min, x_min, y_max, x_max = window
    all_reproj[:, :, 0] = (all_reproj[:, :, 0] - x_min) / (x_max - x_min) * W
    all_reproj[:, :, 1] = (all_reproj[:, :, 1] - y_min) / (y_max - y_min) * H

    # relative depth
    query_depth = np.array([
        depth_map[int(p[0]),
                  np.clip(int(p[1]), 0, H - 1),
                  np.clip(int(p[2]), 0, W - 1)]
        for p in chosen])
    rel_depth = all_depth / (query_depth[:, None] + 1e-8)

    # pad / trim
    N = len(chosen_norm)
    if N < tracks_to_sample:
        pad = tracks_to_sample - N
        print(f"  Warning: only {N} unique points, padding {pad} "
              f"to reach {tracks_to_sample}")
        chosen_norm = np.concatenate([chosen_norm, np.tile(chosen_norm[-1:], (pad, 1))])
        all_reproj = np.concatenate([all_reproj, np.tile(all_reproj[-1:], (pad, 1, 1))])
        all_occ = np.concatenate([all_occ, np.tile(all_occ[-1:], (pad, 1))])
        rel_depth = np.concatenate([rel_depth, np.tile(rel_depth[-1:], (pad, 1))])
        all_wc = np.concatenate([all_wc, np.tile(all_wc[-1:], (pad, 1, 1))])
        all_depth = np.concatenate([all_depth, np.tile(all_depth[-1:], (pad, 1))])

    return (chosen_norm[:tracks_to_sample],
            all_reproj[:tracks_to_sample],
            all_occ[:tracks_to_sample],
            rel_depth[:tracks_to_sample],
            all_wc[:tracks_to_sample],
            all_depth[:tracks_to_sample].astype(np.float32))


# =============================================================================
# Visualisation
# =============================================================================

def plot_tracks(rgb, points, occluded, show_occluded=True):
    """Render tracking visualisation video."""
    num_frames, H, W = rgb.shape[:3]
    colors = plt.cm.hsv(np.linspace(0, 1, len(points)))
    np.random.shuffle(colors)

    in_bounds = (
        (points[:, :, 0] >= 0) & (points[:, :, 0] < W) &
        (points[:, :, 1] >= 0) & (points[:, :, 1] < H)
    )

    frames = []
    for i in range(num_frames):
        fig = plt.figure(figsize=(W / 64, H / 64), dpi=64, frameon=False)
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(np.clip(rgb[i], 0, 255).astype(np.uint8))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

        visible = ~occluded[:, i] & in_bounds[:, i]
        occ_in_bounds = occluded[:, i] & in_bounds[:, i]

        if np.any(visible):
            col = colors[visible, :3]
            col = np.concatenate([col, np.ones((col.shape[0], 1))], axis=1)
            ax.scatter(points[visible, i, 0], points[visible, i, 1],
                       s=6, c=col, clip_on=True)

        if show_occluded and np.any(occ_in_bounds):
            col2 = colors[occ_in_bounds, :3]
            col2 = np.concatenate([col2, np.full((col2.shape[0], 1), 0.4)], axis=1)
            ax.scatter(points[occ_in_bounds, i, 0], points[occ_in_bounds, i, 1],
                       s=20, facecolors='none', edgecolors=col2,
                       linewidths=0.6, clip_on=True)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    return np.stack(frames)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAPIEN point tracking")
    parser.add_argument("--data-dir", type=str, default="outputs",
                        help="Directory with generated SAPIEN outputs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Tracking seed (default: recorded scene seed, or 0 if absent).")
    parser.add_argument("--tracks", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sample-frame",
                        choices=["first", "random", "max_visibility", "multi", "all"],
                        default="all")
    parser.add_argument("--min-per-object", type=int, default=5)
    parser.add_argument("--bg-frac", type=float, default=0.1)
    parser.add_argument("--show-occluded", action="store_true", default=True,
                        help="Show occluded point trajectories (default: on)")
    parser.add_argument("--no-show-occluded", dest="show_occluded",
                        action="store_false",
                        help="Hide occluded point trajectories")
    parser.add_argument("--output-format", choices=["full", "cotracker"], default="full",
                        help="Output format: full (default) or cotracker (CoTracker Kubric format)")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Loading SAPIEN data from {data_dir}...")
    data = load_sapien_data(data_dir)
    annotation = data['annotation']
    scene_metadata = annotation['kubric_format']['metadata']
    tracking_seed = args.seed if args.seed is not None else scene_metadata.get('seed')
    if tracking_seed is None:
        tracking_seed = 0
    np.random.seed(tracking_seed)
    scene_metadata['tracking_seed'] = tracking_seed
    print(f"Tracking seed: {tracking_seed}")

    print(f"Loaded: video {data['video'].shape}, depth {data['depth'].shape}, "
          f"seg {data['segmentations'].shape}")
    print(f"Links: {data['metadata'].get('num_links', '?')}, "
          f"link_poses: {data['link_poses'].shape}")

    query, target, occluded, rel_depth, target_3d, abs_depth = track_points(
        data['depth'], data['metadata']['depth_range'],
        data['segmentations'], data['link_poses'],
        data['metadata'].get('actor_ids', []),
        data['camera']['focal_length'], data['camera']['positions'],
        data['camera']['quaternions'], data['camera']['sensor_width'],
        window=[0, 0, data['video'].shape[1], data['video'].shape[2]],
        tracks_to_sample=args.tracks,
        sampling_stride=args.stride,
        sample_frame=args.sample_frame,
        min_per_object=args.min_per_object,
        bg_frac=args.bg_frac,
        dome_actor_ids=data['metadata'].get('dome_actor_ids', []),
        floor_actor_id=data['metadata'].get('floor_actor_id'),
    )

    print(f"\nTracking: {query.shape[0]} points, {target.shape[1]} frames, "
          f"occlusion rate: {np.mean(occluded):.2%}")

    rgb_float = data['video'].astype(np.float32) / 255.0
    rgb_bright = np.power(np.clip(rgb_float, 0, 1), 0.85) * 255.0
    disp = plot_tracks(rgb_bright, target, occluded,
                       show_occluded=args.show_occluded)

    output_path = os.path.join(data_dir, "tracked_points.mp4")
    imageio.mimsave(output_path, disp, fps=12)
    print(f"Saved: {output_path}")

    if args.output_format == "cotracker":
        seq_id = os.path.basename(data_dir)
        save_cotracker_tracking(data_dir, seq_id, target, occluded, rel_depth,
                                annotation, target_3d=target_3d,
                                abs_depth=abs_depth)
        cleanup_intermediates(data_dir)
        print(f"Saved CoTracker format: {seq_id}.npy")
    else:
        np.save(os.path.join(data_dir, "query_points.npy"), query)
        np.save(os.path.join(data_dir, "target_points.npy"), target)
        np.save(os.path.join(data_dir, "target_points_3d.npy"), target_3d)
        np.save(os.path.join(data_dir, "occluded.npy"), occluded)
        np.save(os.path.join(data_dir, "relative_depth.npy"), rel_depth)
        with open(os.path.join(data_dir, "rigid_box_drop_annotations.json"), 'w') as f:
            json.dump(annotation, f)
        print("Saved trajectory data")


if __name__ == '__main__':
    main()
