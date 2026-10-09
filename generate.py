#!/usr/bin/env python3
"""Generate Art-Kubric scenes.

For each scene index, renders the scene (render_scene.py) and samples its point
tracks (sample_tracks.py). Output goes to <data-dir>/0001/, <data-dir>/0002/, ...
The scene index is also the seed, so any range can be generated independently.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # sibling scripts, so cwd does not matter


def files_present(directory, names):
    return all((directory / name).is_file() and (directory / name).stat().st_size > 0
               for name in names)


def generation_complete(directory, output_format):
    depth_name = "depth_f32.npy"
    if output_format == "full" and not (directory / depth_name).exists():
        depth_name = "depth.npy"
    names = ["video.npy", depth_name, "segmentations.npy", "link_poses.npy",
             "rigid_box_drop_annotations.json"]
    return files_present(directory, names)


def tracking_complete(directory, output_format):
    seq = directory.name
    if output_format == "cotracker":
        names = [f"{seq}.npy", f"{seq}_trajs_2d.npy", f"{seq}_trajs_3d.npy",
                 f"{seq}_coords_depth_abs.npy", f"{seq}_visibility.npy",
                 f"{seq}_with_rank.npz", "bboxes_3d.npy", "depth_f32.npy", "metadata.json"]
        names += ["se3/" + name for name in ("link_poses.npy", "link_actor_ids.npy",
                  "link_rp_seg_ids.npy", "rigid_map_actor.npy", "rigid_map_rp.npy")]
    else:
        if not generation_complete(directory, output_format):
            return False
        names = ["query_points.npy", "target_points.npy", "target_points_3d.npy",
                 "occluded.npy", "relative_depth.npy"]
    return files_present(directory, names)


def run_command(cmd, cwd=None, timeout=None):
    """Run command and return success status.

    If ``timeout`` (seconds) is exceeded, the whole process group is killed and
    the call returns False.
    """
    print(f"\nRunning: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    proc = None
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, text=True, start_new_session=True)
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            print(f"Error: command failed with return code {proc.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Error: command exceeded timeout of {timeout}s, killing process "
              f"group and skipping scene (likely a physics/render hang)")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            if proc is not None:
                proc.kill()
        try:
            if proc is not None:
                proc.wait(timeout=30)
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"Error: {e}")
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate and track SAPIEN data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (one scene; see the README for the full recipe):
  python generate.py --num-scenes 1 --start-idx 1 \\
      --data-dir out/ --assets-index $SAPIEN_ASSETS_ROOT/assets_index.json \\
      --output-format cotracker
"""
    )
    
    # Basic arguments
    parser.add_argument("--num-scenes", type=int, required=True,
                        help="Number of scenes to generate")
    parser.add_argument("--start-idx", type=int, default=1,
                        help="Starting scene index (default: 1)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Base directory for output (default: data)")
    
    # Generation arguments
    parser.add_argument("--assets-index", type=str, default=None,
                        help="Path to assets index file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Legacy compatibility option; scene seeds use the index "
                             "or --seed-override regardless of this value.")
    parser.add_argument("--min-static", type=int, default=5,
                        help="Min number of static objects")
    parser.add_argument("--max-static", type=int, default=10,
                        help="Max number of static objects")
    parser.add_argument("--min-dynamic", type=int, default=10,
                        help="Min number of dynamic objects")
    parser.add_argument("--max-dynamic", type=int, default=20,
                        help="Max number of dynamic objects")
    parser.add_argument("--num-frames", type=int, default=24,
                        help="Number of frames per scene")
    parser.add_argument("--fps", type=int, default=12,
                        help="Frames per second (default: 12)")
    parser.add_argument("--width", type=int, default=256,
                        help="Render width (default: 256)")
    parser.add_argument("--height", type=int, default=256,
                        help="Render height (default: 256)")
    parser.add_argument("--camera-mode", type=str, default="linear",
                        choices=["linear", "orbit", "fixed_random", "linear_movement", "linear_movement_linear_lookat"],
                        help="Camera mode")
    parser.add_argument("--spawn-min-gap", type=str, default=None, metavar="G",
                        help="Minimum centre-to-centre spawn clearance in metres, or 'random:LO:HI' per scene. Default 1.5.")
    parser.add_argument("--spawn-xy-scale", type=str, default=None, metavar="S",
                        help="Scale of the spawn footprint, or 'random:LO:HI' per scene. Default 1.0.")
    parser.add_argument("--penetration-retries", type=int, default=None, metavar="N",
                        help="Replace dynamic objects that sink below the floor, up to N times. Default 2, -1 disables.")
    parser.add_argument("--penetration-z-floor", type=float, default=None, metavar="Z",
                        help="Sink depth (m) counted as floor penetration (default -0.3)")
    parser.add_argument("--penetration-probe-margin", type=float, default=None, metavar="M",
                        help="Headroom (m) above the z-floor at which the probe acts (default 0.15)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Vertical FOV in degrees (default: 57.3)")
    parser.add_argument("--fov-min", type=float, default=None,
                        help="Lower bound for per-scene random vertical FOV in degrees (use with --fov-max)")
    parser.add_argument("--fov-max", type=float, default=None,
                        help="Upper bound for per-scene random vertical FOV in degrees (use with --fov-min)")
    parser.add_argument("--fov-keep-framing", action="store_true",
                        help="Hold framing across FOVs (equivalent to --fov-framing-alpha 1)")
    parser.add_argument("--fov-framing-alpha", type=str, default=None, metavar="A",
                        help="Framing compensation exponent in [0, 1], or 'random' / 'random:LO'.")
    parser.add_argument("--camera-inner-radius", type=float, default=None,
                        help="Inner radius of the camera sampling shell in metres (default 4.0)")
    parser.add_argument("--camera-outer-radius", type=float, default=None,
                        help="Outer radius of the camera sampling shell in metres. Default 6.0.")
    parser.add_argument("--hdri-path", type=str, default=None,
                        help="HDRI path")
    parser.add_argument("--hdri-dir", type=str, default=None,
                        help="HDRI directory")
    
    # Tracking arguments
    parser.add_argument("--tracks", type=int, default=4000,
                        help="Number of tracking points")
    parser.add_argument("--stride", type=int, default=4,
                        help="Sampling stride")
    parser.add_argument("--sample-frame", type=str, default="all",
                        choices=["first", "random", "max_visibility", "multi", "all"],
                        help="Frame sampling strategy (all=sample equally from every frame for uniform coverage)")
    parser.add_argument("--min-per-object", type=int, default=2,
                        help="Min points per object")
    parser.add_argument("--bg-frac", type=float, default=0.15,
                        help="Background sampling fraction")
    
    # Visualization options
    parser.add_argument("--show-occluded", action="store_true", default=True,
                        help="Show occluded point trajectories in video (default: on)")
    parser.add_argument("--no-show-occluded", dest="show_occluded", action="store_false",
                        help="Hide occluded point trajectories in video")

    # Joint motion arguments
    parser.add_argument("--drive-stiffness", type=float, default=5.0,
                        help="PD controller stiffness for active joints")
    parser.add_argument("--drive-damping", type=float, default=0.5,
                        help="PD controller damping for active joints")
    parser.add_argument("--passive-damping", type=float, default=0.15,
                        help="Damping for passive joints")
    parser.add_argument("--active-drive-prob", type=float, default=0.7,
                        help="Probability of setting drive target per joint")
    parser.add_argument("--qpos-range-fraction", type=float, default=0.8,
                        help="Fraction of joint limits to use")
    parser.add_argument("--qvel-prob", type=float, default=1.0,
                        help="Probability of initial joint velocity")
    parser.add_argument("--max-qvel", type=float, default=3.0,
                        help="Max initial joint velocity")
    parser.add_argument("--salient-stiffness-multiplier", type=float, default=1.0,
                        help="Stiffness multiplier for salient joints")
    parser.add_argument("--salient-friction", type=float, default=0.04,
                        help="Friction for salient joints")
    parser.add_argument("--no-weighted-sampling", action="store_true",
                        help="Disable weighted sampling of articulated assets")
    parser.add_argument("--articulated-fraction", type=float, default=None,
                        help="Forward to render_scene.py: fraction of objects that are articulated (rest are GSO rigid)")
    parser.add_argument("--scene-mix-mode", type=str, default=None,
                        choices=["fixed", "varied"],
                        help="fixed (per --articulated-fraction) or varied (per-scene mix).")
    parser.add_argument("--source-mix-mode", type=str, default=None,
                        choices=["index", "fixed", "varied"],
                        help="Forward to render_scene.py: articulated source mixing mode")
    parser.add_argument("--light-level-min", type=float, default=None,
                        help="Forward to render_scene.py: per-scene brightness low end (0=darkest/sharpest shadows)")
    parser.add_argument("--light-level-max", type=float, default=None,
                        help="Forward to render_scene.py: per-scene brightness high end (1=bright/no shadows)")
    parser.add_argument("--composition-target", type=str, default=None,
                        help="Forward to render_scene.py: dataset-level per-source realized share targets")
    parser.add_argument("--source-keep-rates", type=str, default=None,
                        help="Forward to render_scene.py: per-source keep rates for composition compensation")
    parser.add_argument("--single-source-prob", type=float, default=None,
                        help="Forward to render_scene.py: pure single-source scene probability")
    parser.add_argument("--composition-alpha", type=float, default=None,
                        help="Forward to render_scene.py: Dirichlet concentration for per-scene composition jitter")
    parser.add_argument("--dynamic-rigid-motion", action="store_true",
                        help="Passed to render_scene.py: keep rigid scenes dynamic in later frames")
    parser.add_argument("--rich-lighting", action="store_true",
                        help="Passed to render_scene.py: randomize lighting per scene")
    parser.add_argument("--gen-timeout", type=int, default=600,
                        help="Per-scene generation timeout in seconds. 0 disables it.")
    parser.add_argument("--track-timeout", type=int, default=600,
                        help="Per-scene tracking timeout in seconds (default: 600). "
                             "0 disables the timeout.")
    parser.add_argument("--seed-override", type=int, default=None,
                        help="Use this seed for every scene instead of the scene index.")
    parser.add_argument("--rt-path-depth", type=int, default=0,
                        help="Passed to render_scene.py: ray-tracing path depth "
                             "override; lower (2-4) = crisper shadows. 0 = SAPIEN default")
    parser.add_argument("--area-light-gain", type=float, default=1.0,
                        help="Passed to render_scene.py: multiplier on area-light "
                             "emission (raise if shadows are washed out)")
    parser.add_argument("--env-light-scale", type=float, default=1.0,
                        help="Passed to render_scene.py: scale the IBL environment "
                             "map; below 1 lowers the ambient floor so shadows show")
    parser.add_argument("--no-hdri-dome", action="store_true",
                        help="Passed to render_scene.py: skip the opaque dome meshes "
                             "that block all shadow rays")
    parser.add_argument("--source-weights", type=str, default=None,
                        help="Override articulated source weights, e.g. 'partnet=0.5,infinigen=0.3,articraft=0.2'.")

    # Output format
    parser.add_argument("--output-format", type=str, default="full",
                        choices=["full", "cotracker"],
                        help="Output format: full (default) or cotracker (CoTracker Kubric format)")

    # Joint freeze mode
    parser.add_argument("--freeze-joints", action="store_true",
                        help="Lock all articulated joints (objects behave as rigid)")

    # Other options
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip tracking, only generate scenes")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue processing after errors; exit non-zero if any scene fails")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenes that already have complete output")
    
    args = parser.parse_args()
    
    # Find assets_index path
    if args.assets_index is None:
        assets_root = os.environ.get("SAPIEN_ASSETS_ROOT")
        if assets_root:
            default_index = str(Path(assets_root) / "assets_index.json")
            if os.path.exists(default_index):
                args.assets_index = default_index
            else:
                default_index = "assets_index.json"
                if os.path.exists(default_index):
                    args.assets_index = default_index
        else:
            default_index = "assets_index.json"
            if os.path.exists(default_index):
                args.assets_index = default_index
    
    if args.assets_index is None:
        print("Error: assets_index file not found. Use --assets-index or ensure assets_index.json exists.")
        sys.exit(1)
    
    if not os.path.exists(args.assets_index):
        print(f"Error: assets_index file does not exist: {args.assets_index}")
        sys.exit(1)
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_scenes} scenes")
    print(f"Start index: {args.start_idx}")
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Assets index: {args.assets_index}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    total_start = time.time()
    scene_times = []
    
    for i in range(args.num_scenes):
        scene_idx = args.start_idx + i
        if args.output_format == "cotracker":
            scene_name = f"{scene_idx:04d}"
        else:
            scene_name = f"scene{scene_idx:04d}"
        scene_dir = data_dir / scene_name
        scene_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing scene {i+1}/{args.num_scenes}: {scene_name}")
        print(f"{'='*60}")

        # Resume logic: two-level check
        skip_generation = False
        if args.resume:
            complete = (generation_complete(scene_dir, args.output_format) if args.skip_tracking
                        else tracking_complete(scene_dir, args.output_format))
            if complete:
                print(f"  Skipping {scene_name} (output complete)")
                success_count += 1
                continue
            # If generation output exists but tracking didn't finish, skip generation only
            if generation_complete(scene_dir, args.output_format):
                print(f"  Resuming {scene_name} (generation exists, re-running tracking)")
                skip_generation = True

        # Create scene directory
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Build generation command
        gen_cmd = [
            sys.executable, str(_HERE / "render_scene.py"),
            "--assets-index", args.assets_index,
            "--output-dir", str(scene_dir),
            "--min-static", str(args.min_static),
            "--max-static", str(args.max_static),
            "--min-dynamic", str(args.min_dynamic),
            "--max-dynamic", str(args.max_dynamic),
            "--num-frames", str(args.num_frames),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--camera-mode", args.camera_mode,
            "--output-format", args.output_format,
        ]
        
        scene_seed = scene_idx if args.seed_override is None else args.seed_override
        gen_cmd.extend(["--seed", str(scene_seed)])
        
        if args.spawn_min_gap is not None:
            gen_cmd.extend(["--spawn-min-gap", str(args.spawn_min_gap)])
        if args.spawn_xy_scale is not None:
            gen_cmd.extend(["--spawn-xy-scale", str(args.spawn_xy_scale)])
        if args.penetration_retries is not None:
            gen_cmd.extend(["--penetration-retries", str(args.penetration_retries)])
        if args.penetration_z_floor is not None:
            gen_cmd.extend(["--penetration-z-floor", str(args.penetration_z_floor)])
        if args.penetration_probe_margin is not None:
            gen_cmd.extend(["--penetration-probe-margin", str(args.penetration_probe_margin)])

        # Per-scene FOV is sampled in render_scene.py from the scene seed.
        if args.fov is not None:
            gen_cmd.extend(["--fov", str(args.fov)])
        if args.fov_min is not None:
            gen_cmd.extend(["--fov-min", str(args.fov_min)])
        if args.fov_max is not None:
            gen_cmd.extend(["--fov-max", str(args.fov_max)])
        if args.fov_framing_alpha is not None:
            gen_cmd.extend(["--fov-framing-alpha", str(args.fov_framing_alpha)])
        elif args.fov_keep_framing:
            gen_cmd.append("--fov-keep-framing")

        if args.camera_inner_radius is not None:
            gen_cmd.extend(["--camera-inner-radius", str(args.camera_inner_radius)])
        if args.camera_outer_radius is not None:
            gen_cmd.extend(["--camera-outer-radius", str(args.camera_outer_radius)])

        if args.hdri_path:
            gen_cmd.extend(["--hdri-path", args.hdri_path])
        if args.hdri_dir:
            gen_cmd.extend(["--hdri-dir", args.hdri_dir])

        # Joint motion parameters
        gen_cmd.extend([
            "--drive-stiffness", str(args.drive_stiffness),
            "--drive-damping", str(args.drive_damping),
            "--passive-damping", str(args.passive_damping),
            "--active-drive-prob", str(args.active_drive_prob),
            "--qpos-range-fraction", str(args.qpos_range_fraction),
            "--qvel-prob", str(args.qvel_prob),
            "--max-qvel", str(args.max_qvel),
            "--salient-stiffness-multiplier", str(args.salient_stiffness_multiplier),
            "--salient-friction", str(args.salient_friction),
        ])
        if args.no_weighted_sampling:
            gen_cmd.append("--no-weighted-sampling")
        if args.freeze_joints:
            gen_cmd.append("--freeze-joints")
        if args.articulated_fraction is not None:
            gen_cmd.extend(["--articulated-fraction", str(args.articulated_fraction)])
        if args.scene_mix_mode is not None:
            gen_cmd.extend(["--scene-mix-mode", args.scene_mix_mode])
        if args.source_mix_mode is not None:
            gen_cmd.extend(["--source-mix-mode", args.source_mix_mode])
        if args.source_weights is not None:
            gen_cmd.extend(["--source-weights", args.source_weights])
        if args.dynamic_rigid_motion:
            gen_cmd.append("--dynamic-rigid-motion")
        if args.rich_lighting:
            gen_cmd.append("--rich-lighting")
        if args.rt_path_depth > 0:
            gen_cmd.extend(["--rt-path-depth", str(args.rt_path_depth)])
        if args.area_light_gain != 1.0:
            gen_cmd.extend(["--area-light-gain", str(args.area_light_gain)])
        if args.env_light_scale != 1.0:
            gen_cmd.extend(["--env-light-scale", str(args.env_light_scale)])
        if args.no_hdri_dome:
            gen_cmd.append("--no-hdri-dome")
        if args.light_level_min is not None:
            gen_cmd.extend(["--light-level-min", str(args.light_level_min)])
        if args.light_level_max is not None:
            gen_cmd.extend(["--light-level-max", str(args.light_level_max)])
        if args.composition_target is not None:
            gen_cmd.extend(["--composition-target", args.composition_target])
        if args.source_keep_rates is not None:
            gen_cmd.extend(["--source-keep-rates", args.source_keep_rates])
        if args.single_source_prob is not None:
            gen_cmd.extend(["--single-source-prob", str(args.single_source_prob)])
        if args.composition_alpha is not None:
            gen_cmd.extend(["--composition-alpha", str(args.composition_alpha)])
        # Run generation (skip if resuming with existing data)
        if skip_generation:
            print("\n[1/2] Generation skipped (data exists)")
        else:
            print("\n[1/2] Generating scene data...")
            gen_t0 = time.time()
            gen_success = run_command(gen_cmd, cwd=os.getcwd(),
                                      timeout=args.gen_timeout or None)
            gen_elapsed = time.time() - gen_t0
            print(f"  Generation took {gen_elapsed:.1f}s")

            if not gen_success:
                print(f"Error: Scene {scene_name} generation failed")
                fail_count += 1
                if not args.continue_on_error:
                    print("Stopping (use --continue-on-error to continue on error)")
                    sys.exit(1)
                continue

            # Check required files
            if not generation_complete(scene_dir, args.output_format):
                print(f"Error: Scene {scene_name} has incomplete generation output")
                fail_count += 1
                if not args.continue_on_error:
                    sys.exit(1)
                continue
        
        # Run tracking
        if not args.skip_tracking:
            print("\n[2/2] Running point tracking...")
            track_t0 = time.time()
            track_cmd = [
                sys.executable, str(_HERE / "sample_tracks.py"),
                "--data-dir", str(scene_dir),
                "--tracks", str(args.tracks),
                "--stride", str(args.stride),
                "--sample-frame", args.sample_frame,
                "--min-per-object", str(args.min_per_object),
                "--bg-frac", str(args.bg_frac),
                "--output-format", args.output_format,
            ]
            if not args.show_occluded:
                track_cmd.append("--no-show-occluded")
            
            track_success = run_command(track_cmd, cwd=os.getcwd(),
                                        timeout=args.track_timeout or None)
            track_elapsed = time.time() - track_t0
            print(f"  Tracking took {track_elapsed:.1f}s")
            
            if not track_success or not tracking_complete(scene_dir, args.output_format):
                print(f"Error: Scene {scene_name} tracking failed or output is incomplete")
                fail_count += 1
                if not args.continue_on_error:
                    sys.exit(1)
                continue
        else:
            print("  Tracking skipped")
        
        scene_elapsed = time.time() - scene_start
        scene_times.append(scene_elapsed)
        avg_time = sum(scene_times) / len(scene_times)
        remaining = avg_time * (args.num_scenes - i - 1)
        print(f"  Scene {scene_name} total: {scene_elapsed:.1f}s | "
              f"Avg: {avg_time:.1f}s/scene | "
              f"ETA: {remaining/60:.1f}min ({args.num_scenes - i - 1} remaining)")
        success_count += 1
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("Done!")
    print(f"Success: {success_count}/{args.num_scenes}")
    print(f"Failed: {fail_count}/{args.num_scenes}")
    print(f"Total time: {total_elapsed/60:.1f}min ({total_elapsed:.0f}s)")
    if scene_times:
        print(f"Avg per scene: {sum(scene_times)/len(scene_times):.1f}s")
    print(f"Data saved to: {data_dir.absolute()}")
    print(f"{'='*60}")

    if fail_count or (success_count == 0 and args.num_scenes > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
