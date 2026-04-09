"""
maniflow_bc/zarr_dataset.py
===========================
Zarr-backed dataset for ManiFlow BC training.

Replaces the YARR ReplayBuffer for training with a much faster random-access
store:
  - Pre-processed data stored as chunked, LZ4-compressed Zarr arrays.
  - LRU in-process memory cache (configurable GB limit).
  - Supports relative actions (SO(3) delta via quaternion multiply).
  - Compatible with PyTorch DataLoader: DistributedSampler, persistent_workers,
    prefetch_factor.

Disk schema (written by convert_replay_to_zarr.py):
  {split}.zarr/
    rgb            (N, ncam, 3, H, W)   uint8
    depth          (N, ncam, 1, H, W)   float16
    point_cloud    (N, ncam, 3, H, W)   float16
    extrinsics     (N, ncam, 4, 4)      float32
    intrinsics     (N, ncam, 3, 3)      float32
    low_dim_state  (N, 4)               float32
    action         (N, 8)               float32    [xyz qxyz_w grip]
    gripper_pose   (N, 7)               float32
    ignore_collisions (N, 1)            float32
    lang_goal_emb  (N, 1024)            float32
    lang_token_embs(N, 77, D)           float32
    task           (N,)                 str / bytes
    lang_goal      (N,)                 str / bytes
    nerf_rgb_path  (N, nview)           str / bytes
    nerf_dep_path  (N, nview)           str / bytes
    nerf_cam_path  (N, nview)           str / bytes
    nerf_next_rgb_path  (N, nview)      str / bytes
    nerf_next_dep_path  (N, nview)      str / bytes
    nerf_next_cam_path  (N, nview)      str / bytes

Relative-action convention (mirrors 3d_flowmatch_actor):
  - Current gripper pose is stored as `proprioception` (the anchor).
  - rel_pos  = act_pos - prop_pos          (Euclidean delta)
  - rel_quat = q_act * q_prop^{-1}         (right-multiply inverse)
  - gripper open/close bit is kept absolute.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

import pytorch3d.transforms as pytorch3d_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def _open_zarr(path: str, mem_gb: float = 8.0) -> zarr.Group:
    """Open a zarr store wrapped in an LRU memory cache."""
    store = DirectoryStore(path)
    cached = LRUStoreCache(store, max_size=int(mem_gb * 2 ** 30))
    return zarr.open_group(cached, mode="r")


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two xyzw quaternions: q1 * q2.
    Both tensors have shape (..., 4) in [x, y, z, w] order.
    """
    # Convert xyzw → wxyz for pytorch3d, multiply, convert back
    q1_wxyz = q1[..., [3, 0, 1, 2]]
    q2_wxyz = q2[..., [3, 0, 1, 2]]
    out_wxyz = pytorch3d_transforms.quaternion_multiply(q1_wxyz, q2_wxyz)
    return out_wxyz[..., [1, 2, 3, 0]]  # back to xyzw


def _quaternion_invert(q: torch.Tensor) -> torch.Tensor:
    """Invert a unit xyzw quaternion: q^{-1} = [-x,-y,-z, w]."""
    inv = q.clone()
    inv[..., :3] = -inv[..., :3]
    return inv


def to_relative_action(action: torch.Tensor,
                        anchor: torch.Tensor) -> torch.Tensor:
    """
    Compute the SE(3) delta between *action* and *anchor*.

    Args:
        action : (..., 8)  [x, y, z, qx, qy, qz, qw, grip]  target pose
        anchor : (..., 8)  [x, y, z, qx, qy, qz, qw, grip]  current pose

    Returns:
        delta  : (..., 8)  [dx, dy, dz, dqx, dqy, dqz, dqw, grip]
                           grip is kept absolute.
    """
    rel_pos  = action[..., :3] - anchor[..., :3]
    rel_quat = _quaternion_multiply(
        action[..., 3:7],
        _quaternion_invert(anchor[..., 3:7]),
    )
    grip = action[..., 7:8]
    return torch.cat([rel_pos, rel_quat, grip], dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManiFlowZarrDataset(Dataset):
    """
    Zarr-backed dataset for ManiFlow BC.

    Emits **exactly the same key names and shapes** as the YARR
    TaskUniformReplayBuffer so that ``PreprocessAgent`` and
    ``QAttentionManiFlowBCAgent`` receive identical batches regardless of
    which backend is active.

    Key conventions (mirrors YARR buffer / launch_utils.py):
      - Per-camera visual keys: ``{cam}_rgb``, ``{cam}_depth``,
        ``{cam}_point_cloud``, ``{cam}_camera_extrinsics``,
        ``{cam}_camera_intrinsics`` — each with an extra ``timesteps=1``
        leading dim, i.e. shape ``(1, C, H, W)`` for images.
      - NeRF paths stored as numpy object arrays in the same way the YARR
        buffer stores them: ``nerf_multi_view_rgb``, ``nerf_multi_view_depth``,
        ``nerf_multi_view_camera`` and their ``nerf_next_*`` counterparts,
        each with shape ``(num_views,)`` dtype=object (file-path strings).
      - ``lang_goal`` is a ``(1,)`` numpy object array (not a plain string),
        so ``PreprocessAgent`` can index it with ``replay_sample['lang_goal']``
        directly.

    Parameters
    ----------
    zarr_path : str
        Path to the ``{split}.zarr`` directory.
    cameras : list[str]
        Camera names in the same order they were stored during conversion
        (e.g. ``['front']`` or ``['front','wrist']``).  **Must match** the
        order used by ``convert_replay_to_zarr.py``.
    mem_gb : float
        LRU cache size in GiB (per worker process).
    relative_action : bool
        If True return SO(3) delta actions relative to current gripper pose.
    copies : int
        Virtual dataset length multiplier.
    """

    def __init__(
        self,
        zarr_path: str,
        cameras: List[str] = None,
        mem_gb: float = 8.0,
        relative_action: bool = False,
        copies: int = 10,
    ):
        super().__init__()
        self._zarr_path = zarr_path
        self._mem_gb = mem_gb
        self._relative_action = relative_action
        self._copies = copies
        self._cameras = cameras or ["front"]

        # Open store (LRU-cached)
        self._z = _open_zarr(zarr_path, mem_gb)

        self._n = len(self._z["action"])
        print(f"[ManiFlowZarrDataset] {zarr_path} — {self._n} transitions "
              f"(cameras={self._cameras}, copies={copies}, mem_gb={mem_gb})")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._copies * self._n

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        idx = idx % self._n   # wrap within true dataset

        def _get(key) -> torch.Tensor:
            return _to_tensor(self._z[key][idx])

        action       = _get("action").float()        # (8,)
        gripper_pose = _get("gripper_pose").float()  # (7,)

        if self._relative_action:
            anchor = torch.cat([gripper_pose, action[7:8]], dim=-1)  # (8,)
            action = to_relative_action(action, anchor)

        # --- Stacked camera arrays from Zarr: (ncam, C, H, W) -----------
        # Keep rgb in [0, 255] float32 — PreprocessAgent._norm_rgb_() does
        # (x / 255.0) * 2.0 - 1.0, so the input must stay in [0, 255].
        rgb_stacked   = _get("rgb").float()           # (ncam, 3, H, W)  [0, 255]
        dep_stacked   = _get("depth").float()         # (ncam, 1, H, W)
        pcd_stacked   = _get("point_cloud").float()   # (ncam, 3, H, W)
        ext_stacked   = _get("extrinsics").float()    # (ncam, 4, 4)
        int_stacked   = _get("intrinsics").float()    # (ncam, 3, 3)

        # --- Expand per-camera keys with timesteps=1 unsqueeze ----------
        # YARR buffer stores each camera frame with a leading timestep dim,
        # so all per-camera tensors have shape (timesteps, C, H, W).
        # We use timesteps=1 to match (PreprocessAgent does v[:, 0] to remove it).
        sample = {}
        for ci, cname in enumerate(self._cameras):
            # unsqueeze(0) → (1, C, H, W) for images, (1, 4, 4) for matrices
            sample[f"{cname}_rgb"]                = rgb_stacked[ci].unsqueeze(0)   # (1,3,H,W)
            sample[f"{cname}_depth"]              = dep_stacked[ci].unsqueeze(0)   # (1,1,H,W)
            sample[f"{cname}_point_cloud"]        = pcd_stacked[ci].unsqueeze(0)   # (1,3,H,W)
            sample[f"{cname}_camera_extrinsics"]  = ext_stacked[ci].unsqueeze(0)   # (1,4,4)
            sample[f"{cname}_camera_intrinsics"]  = int_stacked[ci].unsqueeze(0)   # (1,3,3)

        # --- Flat state / action / language tensors ----------------------
        sample["low_dim_state"]  = _get("low_dim_state").float()    # (4,)
        sample["gripper_pose"]   = gripper_pose                      # (7,)
        sample["action"]         = action                            # (8,)
        sample["ignore_collisions"] = _get("ignore_collisions").float()  # (1,)
        sample["lang_goal_emb"]  = _get("lang_goal_emb").float()    # (1024,)
        sample["lang_token_embs"] = _get("lang_token_embs").float() # (77, D)
        sample["task"]           = str(self._z["task"][idx].flat[0])

        # lang_goal: (1,) numpy object array — mirrors YARR ReplayElement shape
        sample["lang_goal"] = np.array([str(self._z["lang_goal"][idx].flat[0])],
                                        dtype=object)

        # --- NeRF paths: numpy object arrays of file-path strings --------
        # Shape (num_views,) dtype=object — exactly what QAttentionManiFlowBCAgent
        # expects when it indexes nerf_multi_view_rgb_path[i].
        sample["nerf_multi_view_rgb"]    = np.asarray(self._z["nerf_rgb_path"][idx],   dtype=object)
        sample["nerf_multi_view_depth"]  = np.asarray(self._z["nerf_dep_path"][idx],   dtype=object)
        sample["nerf_multi_view_camera"] = np.asarray(self._z["nerf_cam_path"][idx],   dtype=object)
        sample["nerf_next_multi_view_rgb"]    = np.asarray(self._z["nerf_next_rgb_path"][idx], dtype=object)
        sample["nerf_next_multi_view_depth"]  = np.asarray(self._z["nerf_next_dep_path"][idx], dtype=object)
        sample["nerf_next_multi_view_camera"] = np.asarray(self._z["nerf_next_cam_path"][idx], dtype=object)

        # demo flag (always True since Zarr stores demo transitions only)
        sample["demo"] = torch.tensor(True)

        return sample


# ---------------------------------------------------------------------------
# Collate (handles the string / list-of-strings fields)
# ---------------------------------------------------------------------------

def maniflow_collate_fn(batch: List[dict]) -> dict:
    """
    Collate a list of samples into a batch dict.

    - ``torch.Tensor`` fields are stacked → (B, ...).
    - ``numpy`` object arrays (NeRF paths, lang_goal) are stacked into a
      ``(B, num_views)`` numpy object array — exactly as PyTorchReplayBuffer
      returns them so downstream code (preprocess_data, PreprocessAgent,
      QAttentionManiFlowBCAgent) needs zero changes.
    - ``str`` fields (task) are collected into a plain Python list.
    """
    out = {}
    first = batch[0]

    for k, v in first.items():
        samples = [s[k] for s in batch]

        if isinstance(v, torch.Tensor):
            out[k] = torch.stack(samples, dim=0)          # (B, ...)

        elif isinstance(v, np.ndarray) and v.dtype == object:
            # Stack object arrays: list of (num_views,) → (B, num_views)
            out[k] = np.stack(samples, axis=0)            # (B, num_views)

        else:
            # Plain strings (task), booleans, etc.
            out[k] = samples

    return out
