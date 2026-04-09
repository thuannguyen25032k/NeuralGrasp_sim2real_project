"""
convert_replay_to_zarr.py
=========================
One-shot converter: reads all RLBench NeRF demos from disk (the same format
used by the existing fill_replay pipeline) and writes a compact, chunked,
LZ4-compressed Zarr store that can be read by ManiFlowZarrDataset.

Usage
-----
    python scripts/convert_replay_to_zarr.py \
        --demo_path  data/train_data \
        --out_path   data/train_zarr \
        --tasks      close_jar open_drawer ... \
        --num_demos  100 \
        --split      train

The script is intentionally standalone — it imports only stdlib + the
existing project helpers, so it can be run inside the Docker container
without installing extra packages beyond what the project already needs.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# ---- project imports -------------------------------------------------------
# Assumes the script is run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rlbench.utils as rlbench_utils
from rlbench.observation_config import ObservationConfig
from helpers import demo_loading_utils, utils
from helpers.language_model import create_language_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK = 64          # zarr chunk size along the leading (sample) axis
COMPRESSOR = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)


def _create_or_append(zarr_file: zarr.Group, field: str,
                      data: np.ndarray, shape_per_sample: tuple,
                      dtype):
    """
    Create a zarr dataset if it doesn't exist, then append *data* along axis 0.
    """
    if field not in zarr_file:
        zarr_file.create_dataset(
            field,
            shape=(0,) + shape_per_sample,
            chunks=(CHUNK,) + shape_per_sample,
            compressor=COMPRESSOR,
            dtype=dtype,
        )
    zarr_file[field].append(data)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize and canonicalize (qw >= 0) a quaternion array (..., 4)."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    mask = q[..., -1] < 0
    q[mask] = -q[mask]
    return q


# ---------------------------------------------------------------------------
# Per-demo processing
# ---------------------------------------------------------------------------

def process_demo(demo, episode_keypoints, cameras, language_model, description,
                 zarr_file, obs_config, cfg_episode_length=25):
    """
    Walk through keypoints of a single demo and write each transition to zarr.
    """
    prev_action = None

    for k, keypoint in enumerate(episode_keypoints):
        obs      = demo[max(0, keypoint - 1)]   # current obs (before action)
        obs_tp1  = demo[keypoint]               # next obs (target keypoint)

        # ---- Action -------------------------------------------------------
        quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
        if quat[-1] < 0:
            quat = -quat
        grip = float(obs_tp1.gripper_open)
        action = np.concatenate([obs_tp1.gripper_pose[:3], quat, [grip]]).astype(np.float32)

        # ---- Low-dim state ------------------------------------------------
        robot_state = np.array([
            obs.gripper_open,
            *np.clip(obs.gripper_joint_positions, 0., 0.04)
        ], dtype=np.float32)
        time = (1. - (k / float(cfg_episode_length - 1))) * 2. - 1.
        low_dim_state = np.append(robot_state, time).astype(np.float32)

        # ---- Language embeddings ------------------------------------------
        sentence_emb, token_embs = language_model.extract(description)
        lang_goal_emb   = sentence_emb[0].float().detach().cpu().numpy()   # (1024,)
        lang_token_embs = token_embs[0].float().detach().cpu().numpy()     # (77, D)

        # ---- Camera observations ------------------------------------------
        ncam = len(cameras)
        H, W = 128, 128
        rgb_arr  = np.zeros((ncam, 3, H, W), dtype=np.uint8)
        dep_arr  = np.zeros((ncam, 1, H, W), dtype=np.float16)
        pcd_arr  = np.zeros((ncam, 3, H, W), dtype=np.float16)
        ext_arr  = np.zeros((ncam, 4, 4),    dtype=np.float32)
        int_arr  = np.zeros((ncam, 3, 3),    dtype=np.float32)

        obs_dict = utils.extract_obs(
            obs, t=k, prev_action=prev_action,
            cameras=cameras,
            episode_length=cfg_episode_length,
            next_obs=obs_tp1,
        )
        for ci, cname in enumerate(cameras):
            # RGB: (3, H, W) float32 [0,1] → uint8 [0,255]
            rgb_f32 = obs_dict.get(f"{cname}_rgb", np.zeros((3, H, W), np.float32))
            rgb_arr[ci] = (np.clip(rgb_f32, 0., 1.) * 255).astype(np.uint8)
            # Depth
            dep = obs_dict.get(f"{cname}_depth",
                               np.zeros((1, H, W), np.float32))
            dep_arr[ci] = dep.astype(np.float16)
            # Point cloud
            pcd = obs_dict.get(f"{cname}_point_cloud",
                               np.zeros((3, H, W), np.float32))
            pcd_arr[ci] = pcd.astype(np.float16)
            # Extrinsics / intrinsics
            ext_arr[ci] = obs.misc.get(f"{cname}_camera_extrinsics",
                                        np.eye(4)).astype(np.float32)
            int_arr[ci] = obs.misc.get(f"{cname}_camera_intrinsics",
                                        np.eye(3)).astype(np.float32)

        # ---- NeRF paths ---------------------------------------------------
        nv = len(obs.nerf_multi_view_rgb) if obs.nerf_multi_view_rgb is not None else 0
        nerf_rgb_p  = np.array(obs.nerf_multi_view_rgb  or [], dtype=object)
        nerf_dep_p  = np.array(obs.nerf_multi_view_depth or [], dtype=object)
        nerf_cam_p  = np.array(obs.nerf_multi_view_camera or [], dtype=object)
        nn_rgb_p    = np.array(obs_tp1.nerf_multi_view_rgb  or [], dtype=object)
        nn_dep_p    = np.array(obs_tp1.nerf_multi_view_depth or [], dtype=object)
        nn_cam_p    = np.array(obs_tp1.nerf_multi_view_camera or [], dtype=object)

        nview = max(len(nerf_rgb_p), 1)

        prev_action = action.copy()

        # ---- Write to zarr ------------------------------------------------
        i = np.newaxis   # batch dim
        _create_or_append(zarr_file, "rgb",           rgb_arr[i],  (ncam, 3, H, W), "uint8")
        _create_or_append(zarr_file, "depth",         dep_arr[i],  (ncam, 1, H, W), "float16")
        _create_or_append(zarr_file, "point_cloud",   pcd_arr[i],  (ncam, 3, H, W), "float16")
        _create_or_append(zarr_file, "extrinsics",    ext_arr[i],  (ncam, 4, 4),    "float32")
        _create_or_append(zarr_file, "intrinsics",    int_arr[i],  (ncam, 3, 3),    "float32")
        _create_or_append(zarr_file, "low_dim_state", low_dim_state[i], (low_dim_state.shape[0],), "float32")
        _create_or_append(zarr_file, "action",        action[i],   (8,),            "float32")
        _create_or_append(zarr_file, "gripper_pose",  obs_tp1.gripper_pose[i], (7,), "float32")
        _create_or_append(zarr_file, "lang_goal_emb",   lang_goal_emb[i],   (1024,),                  "float32")
        _create_or_append(zarr_file, "lang_token_embs", lang_token_embs[i], (lang_token_embs.shape[0],
                                                                              lang_token_embs.shape[1]), "float32")
        _create_or_append(zarr_file, "task",             np.array([[description]], dtype=object), (1,), object)
        _create_or_append(zarr_file, "lang_goal",        np.array([[description]], dtype=object), (1,), object)
        _create_or_append(zarr_file, "nerf_rgb_path",     nerf_rgb_p[i], (nview,), object)
        _create_or_append(zarr_file, "nerf_dep_path",     nerf_dep_p[i], (nview,), object)
        _create_or_append(zarr_file, "nerf_cam_path",     nerf_cam_p[i], (nview,), object)
        _create_or_append(zarr_file, "nerf_next_rgb_path", nn_rgb_p[i],  (nview,), object)
        _create_or_append(zarr_file, "nerf_next_dep_path", nn_dep_p[i],  (nview,), object)
        _create_or_append(zarr_file, "nerf_next_cam_path", nn_cam_p[i],  (nview,), object)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert RLBench demos → Zarr.")
    p.add_argument("--demo_path",   required=True,  help="Root of stored RLBench demos")
    p.add_argument("--out_path",    required=True,  help="Output Zarr directory")
    p.add_argument("--tasks",       required=True,  nargs="+", help="Task names")
    p.add_argument("--num_demos",   type=int, default=100)
    p.add_argument("--cameras",     nargs="+",
                   default=["front", "left_shoulder", "right_shoulder", "wrist"])
    p.add_argument("--language_model", default="clip",
                   help="Language model name (default: clip)")
    p.add_argument("--episode_length", type=int, default=25)
    p.add_argument("--keypoint_method", default="heuristic")
    p.add_argument("--split",       default="train")
    p.add_argument("--device",      default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.split}.zarr")

    print(f"[convert_replay_to_zarr] Output → {out_file}")
    print(f"  Tasks    : {args.tasks}")
    print(f"  Demos    : {args.num_demos} per task")
    print(f"  Cameras  : {args.cameras}")

    # Build a minimal ObservationConfig
    from helpers.utils import create_obs_config
    obs_config = create_obs_config(
        camera_names=args.cameras,
        camera_resolution=[128, 128],
        method_name="convert",
        use_nerf_multi_view=True,
        use_depth=True,
    )

    language_model = create_language_model(name=args.language_model,
                                           device=args.device)

    with zarr.open_group(out_file, mode="w") as zarr_file:
        for task in args.tasks:
            print(f"\n  Task: {task}")
            for d_idx in tqdm(range(args.num_demos), desc=task):
                try:
                    demo = rlbench_utils.get_stored_demos(
                        amount=1, image_paths=False,
                        dataset_root=args.demo_path,
                        variation_number=-1, task_name=task,
                        obs_config=obs_config,
                        random_selection=False,
                        from_episode_number=d_idx,
                    )[0]
                except Exception as e:
                    print(f"    [WARN] skipping demo {d_idx}: {e}")
                    continue

                descs = demo._observations[0].misc.get("descriptions", [task])
                desc  = descs[0] if descs else task
                episode_keypoints = demo_loading_utils.keypoint_discovery(
                    demo, method=args.keypoint_method
                )
                if not episode_keypoints:
                    continue

                process_demo(
                    demo=demo,
                    episode_keypoints=episode_keypoints,
                    cameras=args.cameras,
                    language_model=language_model,
                    description=desc,
                    zarr_file=zarr_file,
                    obs_config=obs_config,
                    cfg_episode_length=args.episode_length,
                )

    # Final sanity check
    with zarr.open_group(out_file, mode="r") as zf:
        n = len(zf["action"])
        print(f"\n[convert_replay_to_zarr] Done. {n} transitions written to {out_file}")
        for k in zf.keys():
            assert len(zf[k]) == n, f"Length mismatch: {k}"
        print("  All arrays consistent ✓")


if __name__ == "__main__":
    main()
