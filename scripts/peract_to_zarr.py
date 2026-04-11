"""
convert_replay_to_zarr.py
=========================
Convert stored RLBench demos (disk format) to a Zarr store whose schema
matches 3D FlowMatch Actor exactly — **no simulator / YARR replay needed**.

Reads directly from the on-disk episode layout produced by RLBench's demo
saver:

    {demo_path}/{task}/all_variations/episodes/episode{N}/
        low_dim_obs.pkl          — RLBench Demo object (poses, misc, …)
        {cam}_rgb/{frame}.png    — uint8 RGB images  (H, W, 3)
        {cam}_depth/{frame}.png  — encoded depth     (H, W, 3) RGB24
        variation_number.pkl     — int
        variation_descriptions.pkl — List[str]
        nerf_data/{frame}/images/{view}.png   — optional NeRF RGB
        nerf_data/{frame}/depths/{view}.png   — optional NeRF depth
        nerf_data/{frame}/poses/{view}.txt    — optional NeRF camera pose

Output zarr schema (all arrays length-N, one row per keyframe):

    rgb             (N, ncam, 3, H, W)        uint8
    depth           (N, ncam, 1, H, W)        float16  (metric metres)
    pcd             (N, ncam, 3, H, W)        float16  (world-frame, derived)
    proprioception  (N, 3, 1, 8)              float32  (history window)
    action          (N, chunk_size, 1, 8)     float32  [xyz qxyzw grip]
    extrinsics      (N, ncam, 4, 4)           float32
    intrinsics      (N, ncam, 3, 3)           float32
    task_id         (N,)                      uint8
    variation       (N,)                      uint8
    nerf_rgb_path   (N, nview)                object   (optional)
    nerf_dep_path   (N, nview)                object   (optional)
    nerf_cam_path   (N, nview)                object   (optional)
    nerf_next_rgb_path  (N, nview)            object   (optional)
    nerf_next_dep_path  (N, nview)            object   (optional)
    nerf_next_cam_path  (N, nview)            object   (optional)

    instructions.json  {task_name: {variation_id(str): [instruction, ...]}}

Usage
-----
    python scripts/convert_replay_to_zarr.py \\
        --demo_path  data/train_data \\
        --out_path   data/train_zarr \\
        --tasks      close_jar open_drawer \\
        --num_demos  100 \\
        --split      train
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr
from numcodecs import Blosc, VLenUTF8
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ---- project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.demo_loading_utils import keypoint_discovery


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK      = 64          # zarr chunk along leading axis
COMPRESSOR = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
NHIST      = 3           # gripper-pose history length (t-2, t-1, t)
NHAND      = 1           # arms (1 = single-arm)
DEPTH_SCALE = 2**24 - 1  # RLBench float_array_to_rgb_image scale factor


# ---------------------------------------------------------------------------
# Depth decoding  (mirrors hiveformer_to_zarr / rlbench_utils)
# ---------------------------------------------------------------------------

def _decode_depth(png_path: str, near: float, far: float) -> np.ndarray:
    """Read an RGB24-encoded RLBench depth image → metric float array (H, W)."""
    img = np.array(Image.open(png_path))        # (H, W, 3) uint8
    # Reconstruct 24-bit integer from RGB channels
    float_arr = (img[..., 0].astype(np.float32) * 65536
                 + img[..., 1].astype(np.float32) * 256
                 + img[..., 2].astype(np.float32)) / DEPTH_SCALE
    return (near + float_arr * (far - near)).astype(np.float32)


# ---------------------------------------------------------------------------
# Point cloud from depth + intrinsics + extrinsics
# ---------------------------------------------------------------------------

def _depth_to_pcd(depth: np.ndarray, intr: np.ndarray,
                  extr: np.ndarray) -> np.ndarray:
    """Back-project metric depth (H, W) → world-frame point cloud (3, H, W)."""
    H, W = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)                 # (H, W)
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    # Camera-frame homogeneous coords (4, H*W)
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0).reshape(4, -1)
    # Transform to world frame
    pts_world = (extr @ pts_cam)[:3]           # (3, H*W)
    return pts_world.reshape(3, H, W).astype(np.float32)


# ---------------------------------------------------------------------------
# Quaternion normalisation (matches helpers/utils.py convention)
# ---------------------------------------------------------------------------

def _normalise_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    q = q / n if n > 1e-8 else q
    if q[-1] < 0:
        q = -q
    return q


def _obs_to_pose8(obs) -> np.ndarray:
    """[xyz qxyzw grip] float32 from an RLBench Observation."""
    quat = _normalise_quat(np.array(obs.gripper_pose[3:], dtype=np.float32))
    return np.concatenate([
        np.array(obs.gripper_pose[:3], dtype=np.float32),
        quat,
        [float(obs.gripper_open)]
    ])


# ---------------------------------------------------------------------------
# NeRF path helpers
# ---------------------------------------------------------------------------

def _nerf_paths_for_frame(ep_folder: str, frame_idx: int):
    """
    Return (rgb_paths, dep_paths, cam_paths) as sorted string lists for the
    given frame index.  Returns ([], [], []) if nerf_data is absent.
    """
    nerf_root = Path(ep_folder) / "nerf_data" / str(frame_idx)
    if not nerf_root.exists():
        return [], [], []
    rgb_dir = nerf_root / "images"
    dep_dir = nerf_root / "depths"
    cam_dir = nerf_root / "poses"
    rgb_paths = sorted(str(p) for p in rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    dep_paths = sorted(str(p) for p in dep_dir.glob("*.png")) if dep_dir.exists() else []
    cam_paths = sorted(str(p) for p in cam_dir.glob("*.txt")) if cam_dir.exists() else []
    return rgb_paths, dep_paths, cam_paths


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------

def _append(zf: zarr.Group, field: str, data: np.ndarray,
            shape_per_sample: tuple, dtype):
    if field not in zf:
        kwargs = dict(
            shape=(0,) + shape_per_sample,
            chunks=(CHUNK,) + shape_per_sample,
            compressor=COMPRESSOR,
            dtype=dtype,
        )
        # zarr v2 requires an explicit object codec for variable-length string arrays
        if dtype is object or dtype == object:
            kwargs["object_codec"] = VLenUTF8()
            kwargs["compressor"] = None   # VLenUTF8 handles encoding
        zf.create_dataset(field, **kwargs)
    zf[field].append(data)


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(ep_folder: str, demo, key_frames: List[int],
                    cameras: List[str], task_id: int, variation: int,
                    zarr_file: zarr.Group, chunk_size: int,
                    im_size: int = 128) -> int:
    """
    Write every keyframe of one episode into zarr.

    Parameters
    ----------
    ep_folder   : path to episode directory on disk (contains *_rgb/, *_depth/ …)
    demo        : loaded low_dim_obs.pkl (rlbench.demo.Demo)
    key_frames  : list of keyframe indices **including** the initial frame 0
                  (i.e. key_frames[0] == 0, targets are key_frames[1:])
    cameras     : list of camera names
    task_id     : integer task index
    variation   : integer variation index
    zarr_file   : open zarr.Group in write mode
    chunk_size  : action chunk size T
    im_size     : image height = width in pixels

    Returns
    -------
    Number of rows appended.
    """
    ncam = len(cameras)

    # ---- Build full-episode pose array for history window ---------------
    all_poses = [_obs_to_pose8(obs) for obs in demo]

    # Keyframe target poses  (key_frames[0] is "before first keyframe" source)
    kf_poses = [_obs_to_pose8(demo[k]) for k in key_frames]

    rows_written = 0
    # Iterate over observation keyframes: source = key_frames[i], target = key_frames[i+1]
    for i, src_frame in enumerate(key_frames[:-1]):
        tgt_frame = key_frames[i + 1]

        obs     = demo[src_frame]
        obs_tp1 = demo[tgt_frame]

        # ---- Action chunk (T, NHAND, 8) ---------------------------------
        # Chunk = [target_i, target_{i+1}, …, target_{i+T-1}] (clamped)
        n_kf = len(key_frames) - 1           # total number of (src→tgt) pairs
        chunk_poses = [kf_poses[min(i + 1 + ci, n_kf)]
                       for ci in range(chunk_size)]
        action = np.stack(chunk_poses, axis=0)[:, np.newaxis, :]  # (T, 1, 8)

        # ---- Proprioception history (3, NHAND, 8) -----------------------
        h = [max(0, src_frame - 2), max(0, src_frame - 1), src_frame]
        prop = np.stack([all_poses[hi] for hi in h])[:, np.newaxis, :]   # (3,1,8)

        # ---- Camera observations ----------------------------------------
        rgb_arr = np.zeros((ncam, 3, im_size, im_size), dtype=np.uint8)
        dep_arr = np.zeros((ncam, 1, im_size, im_size), dtype=np.float16)
        pcd_arr = np.zeros((ncam, 3, im_size, im_size), dtype=np.float16)
        ext_arr = np.zeros((ncam, 4, 4),               dtype=np.float32)
        int_arr = np.zeros((ncam, 3, 3),               dtype=np.float32)

        for ci, cam in enumerate(cameras):
            # RGB: read PNG from disk, transpose (H,W,3)→(3,H,W)
            rgb_path = os.path.join(ep_folder, f"{cam}_rgb", f"{src_frame}.png")
            if os.path.exists(rgb_path):
                rgb_arr[ci] = np.array(Image.open(rgb_path)).transpose(2, 0, 1)

            # Depth: decode RGB24-encoded depth → metric float → (1, H, W)
            dep_path = os.path.join(ep_folder, f"{cam}_depth", f"{src_frame}.png")
            near = obs.misc.get(f"{cam}_camera_near", 0.1)
            far  = obs.misc.get(f"{cam}_camera_far",  10.0)
            if os.path.exists(dep_path):
                dep_m = _decode_depth(dep_path, near, far)   # (H, W) float32
                dep_arr[ci] = dep_m[np.newaxis].astype(np.float16)  # (1,H,W)

                # Point cloud: back-project depth using camera matrices
                extr = obs.misc.get(f"{cam}_camera_extrinsics", np.eye(4))
                intr = obs.misc.get(f"{cam}_camera_intrinsics", np.eye(3))
                ext_arr[ci] = extr.astype(np.float32)
                int_arr[ci] = intr.astype(np.float32)
                pcd_arr[ci] = _depth_to_pcd(dep_m, intr, extr).astype(np.float16)
            else:
                extr = obs.misc.get(f"{cam}_camera_extrinsics", np.eye(4))
                intr = obs.misc.get(f"{cam}_camera_intrinsics", np.eye(3))
                ext_arr[ci] = extr.astype(np.float32)
                int_arr[ci] = intr.astype(np.float32)

        # ---- NeRF paths (optional) --------------------------------------
        nerf_rgb_p, nerf_dep_p, nerf_cam_p = _nerf_paths_for_frame(ep_folder, src_frame)
        nn_rgb_p,   nn_dep_p,   nn_cam_p   = _nerf_paths_for_frame(ep_folder, tgt_frame)
        nview = len(nerf_rgb_p)

        # If the target frame has no NeRF data (e.g. last keyframe),
        # pad with empty strings so shapes stay consistent with (nview,).
        if nview > 0 and len(nn_rgb_p) == 0:
            nn_rgb_p = [""] * nview
            nn_dep_p = [""] * nview
            nn_cam_p = [""] * nview

        # ---- Write row --------------------------------------------------
        i_ = np.newaxis
        _append(zarr_file, "rgb",           rgb_arr[i_], (ncam, 3, im_size, im_size), "uint8")
        _append(zarr_file, "depth",         dep_arr[i_], (ncam, 1, im_size, im_size), "float16")
        _append(zarr_file, "pcd",           pcd_arr[i_], (ncam, 3, im_size, im_size), "float16")
        _append(zarr_file, "proprioception",prop[i_],    (NHIST, NHAND, 8),            "float32")
        _append(zarr_file, "action",        action[i_],  (chunk_size, NHAND, 8),       "float32")
        _append(zarr_file, "extrinsics",    ext_arr[i_], (ncam, 4, 4),                 "float32")
        _append(zarr_file, "intrinsics",    int_arr[i_], (ncam, 3, 3),                 "float32")
        _append(zarr_file, "task_id",  np.array([task_id],  dtype=np.uint8),  (), "uint8")
        _append(zarr_file, "variation", np.array([variation], dtype=np.uint8), (), "uint8")

        if nview > 0:
            pa = np.array(nerf_rgb_p, dtype=object)
            _append(zarr_file, "nerf_rgb_path",      pa[i_],                   (nview,), object)
            _append(zarr_file, "nerf_dep_path",      np.array(nerf_dep_p, dtype=object)[i_], (nview,), object)
            _append(zarr_file, "nerf_cam_path",      np.array(nerf_cam_p, dtype=object)[i_], (nview,), object)
            _append(zarr_file, "nerf_next_rgb_path", np.array(nn_rgb_p,   dtype=object)[i_], (nview,), object)
            _append(zarr_file, "nerf_next_dep_path", np.array(nn_dep_p,   dtype=object)[i_], (nview,), object)
            _append(zarr_file, "nerf_next_cam_path", np.array(nn_cam_p,   dtype=object)[i_], (nview,), object)

        rows_written += 1

    return rows_written


# ---------------------------------------------------------------------------
# Image size auto-detection
# ---------------------------------------------------------------------------

def _detect_im_size(demo_path: str, task: str, cam: str) -> int:
    """Read one PNG to detect actual image resolution."""
    ep0 = Path(demo_path) / task / "all_variations" / "episodes" / "episode0"
    candidates = sorted((ep0 / f"{cam}_rgb").glob("*.png"))
    if candidates:
        img = np.array(Image.open(candidates[0]))
        return img.shape[0]   # assume square
    return 128


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert RLBench demos (disk) → Zarr (3D FlowMatch Actor schema)")
    p.add_argument("--demo_path",       required=True, help="Root of stored RLBench demos")
    p.add_argument("--out_path",        required=True, help="Output directory for zarr + json")
    p.add_argument("--tasks",           required=True, nargs="+", help="Task names")
    p.add_argument("--num_demos",       type=int, default=100,
                   help="Max episodes per task (skips missing ones)")
    p.add_argument("--cameras",         nargs="+",
                   default=["front", "left_shoulder", "right_shoulder", "wrist"])
    p.add_argument("--chunk_size",      type=int, default=1,
                   help="Action chunk size T (must match action_chunk_size in config)")
    p.add_argument("--keypoint_method", default="heuristic",
                   choices=["heuristic", "random", "fixed_interval"])
    p.add_argument("--split",           default="train")
    p.add_argument("--overwrite",       action="store_true",
                   help="Overwrite existing zarr (default: skip if exists)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    out_zarr = os.path.join(args.out_path, f"{args.split}.zarr")
    out_json = os.path.join(args.out_path, "instructions.json")

    if os.path.exists(out_zarr) and not args.overwrite:
        print(f"[convert] Zarr already exists at {out_zarr}. Pass --overwrite to regenerate.")
        return

    # Auto-detect image size from first available episode
    im_size = _detect_im_size(args.demo_path, args.tasks[0], args.cameras[0])
    print(f"[convert] Detected image size: {im_size}×{im_size}")

    print(f"[convert] Output     → {out_zarr}")
    print(f"  Tasks      : {args.tasks}")
    print(f"  Demos/task : {args.num_demos}")
    print(f"  Cameras    : {args.cameras}")
    print(f"  chunk_size : {args.chunk_size}")

    task2id      = {task: t for t, task in enumerate(args.tasks)}
    instructions = {task: {} for task in args.tasks}
    total_rows   = 0

    with zarr.open_group(out_zarr, mode="w") as zarr_file:
        for task in args.tasks:
            tid      = task2id[task]
            ep_root  = Path(args.demo_path) / task / "all_variations" / "episodes"
            if not ep_root.exists():
                print(f"  [WARN] {ep_root} not found, skipping task {task}")
                continue

            episodes = sorted([d for d in ep_root.iterdir() if d.name.startswith("episode")])
            episodes = episodes[:args.num_demos]
            print(f"\n  Task: {task}  (id={tid})  — {len(episodes)} episodes")

            for ep_dir in tqdm(episodes, desc=task):
                ld_file = ep_dir / "low_dim_obs.pkl"
                var_file = ep_dir / "variation_number.pkl"
                desc_file = ep_dir / "variation_descriptions.pkl"

                if not ld_file.exists():
                    continue

                with open(ld_file, "rb") as f:
                    demo = pickle.load(f)

                variation = 0
                if var_file.exists():
                    with open(var_file, "rb") as f:
                        variation = int(pickle.load(f))

                descriptions = [task]
                if desc_file.exists():
                    with open(desc_file, "rb") as f:
                        descriptions = pickle.load(f)

                var_str = str(variation)
                if var_str not in instructions[task]:
                    instructions[task][var_str] = []
                for d in descriptions:
                    if d not in instructions[task][var_str]:
                        instructions[task][var_str].append(d)

                # Keyframe discovery — uses heuristic from helpers/demo_loading_utils
                kps = keypoint_discovery(demo, method=args.keypoint_method)
                if not kps:
                    continue
                # Prepend the initial frame (index 0) as the starting observation
                key_frames = [0] + kps

                rows = process_episode(
                    ep_folder=str(ep_dir),
                    demo=demo,
                    key_frames=key_frames,
                    cameras=args.cameras,
                    task_id=tid,
                    variation=variation,
                    zarr_file=zarr_file,
                    chunk_size=args.chunk_size,
                    im_size=im_size,
                )
                total_rows += rows

    # Save instructions
    with open(out_json, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"\n[convert] Instructions → {out_json}")

    # Sanity check
    with zarr.open_group(out_zarr, mode="r") as zf:
        n = len(zf["action"])
        print(f"[convert] Done. {n} rows written to {out_zarr}")
        mismatches = [k for k in zf.keys() if len(zf[k]) != n]
        if mismatches:
            print(f"  [WARN] Length mismatch in: {mismatches}")
        else:
            print("  All arrays consistent ✓")


if __name__ == "__main__":
    main()