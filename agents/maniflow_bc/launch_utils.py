"""
ManiFlow launch utilities
=========================
Mirrors agents/manigaussian_bc/launch_utils.py but:
  - Builds a VoxelFlowEncoder instead of PerceiverVoxelLangEncoder.
  - Wraps it in ManiFlowBCAgent / ManiFlowStackAgent.
  - Keeps the replay buffer definition identical (same observation/action
    elements) so existing data pipelines work unchanged.
  - The continuous action stored in the replay is the 8-DoF gripper pose:
    [x, y, z, qx, qy, qz, qw, gripper_open].
"""

import logging
import random
from typing import List, Optional

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo

from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_single_process import UniformReplayBufferSingleProcess

from helpers import demo_loading_utils, utils
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from helpers.language_model import create_language_model

from agents.maniflow_bc.voxel_flow_encoder import VoxelFlowEncoder
from agents.maniflow_bc.qattention_maniflow_agent import ManiFlowBCAgent
from agents.maniflow_bc.qattention_stack_agent import ManiFlowStackAgent
# --- Fast Zarr DataLoader (Recommendation 1 + 2 + 3) ---
from agents.maniflow_bc.zarr_dataset import ManiFlowZarrDataset, maniflow_collate_fn
from agents.maniflow_bc.gpu_preprocessor import ManiFlowGPUPreprocessor

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from omegaconf import DictConfig
from termcolor import colored, cprint
from lightning_fabric import Fabric


REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=None,
                  replay_size=3e5,
                  single_process=False,
                  cfg=None):
    """
    Create a replay buffer with the same structure as manigaussian_bc so that
    existing data-generation scripts work without modification.
    """
    if image_size is None:
        image_size = [128, 128]

    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = cfg.method.language_model_dim
    cprint(f"[create_replay] lang_emb_dim: {lang_emb_dim}", "green")

    num_view_for_nerf = cfg.rlbench.num_view_for_nerf

    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))  # low-dim state (e.g. gripper xyz, open/close) for auxiliary losses and non-visual baselines

    for cname in cameras:
        observation_elements.extend([
            # RGB image (normalized to [0, 1]) has shape (3, H, W) to be compatible with PyTorch ConvNets
            ObservationElement('%s_rgb' % cname,
                               (3, *image_size,), np.float32),
            # depth image has shape (1, H, W) to be compatible with PyTorch ConvNets
            ObservationElement('%s_depth' % cname,
                               (1, *image_size,), np.float32),
            # point cloud has shape (3, H, W) to be compatible with PyTorch ConvNets
            ObservationElement('%s_point_cloud' %
                               cname,  (3, *image_size),  np.float32),
            # camera extrinsics has shape (4, 4) to be compatible with PyTorch ConvNets
            ObservationElement('%s_camera_extrinsics' %
                               cname, (4, 4,), np.float32),
            # camera intrinsics has shape (3, 3) to be compatible with PyTorch ConvNets
            ObservationElement('%s_camera_intrinsics' %
                               cname, (3, 3,), np.float32),
        ])

    # NeRF multi-view observations (same as manigaussian_bc)
    for prefix in ('nerf_multi_view', 'nerf_next_multi_view'):
        observation_elements.extend([
            # (num_views,) array of (3, H, W) RGB images stored as objects since they can be large and variable-length
            ObservationElement(f'{prefix}_rgb',
                               (num_view_for_nerf,), np.object_),
            # (num_views,) array of (1, H, W) depth images stored as objects since they can be large and variable-length
            ObservationElement(f'{prefix}_depth',
                               (num_view_for_nerf,), np.object_),
            # (num_views,) array of dicts with 'extrinsics' (4, 4) and 'intrinsics' (3, 3) stored as objects since they can be large and variable-length
            ObservationElement(f'{prefix}_camera',
                               (num_view_for_nerf,), np.object_),
        ])
    # ---------------------------------------------------------------------------
    # Additional observation elements specific to ManiFlow (not in manigaussian_bc):
    #  - gripper pose (for flow regression target)
    #  - ignore collisions flag (for flow regression loss masking)
    #  - language goal embedding + token embeddings (for language-conditioned modeling)
    #  - task name (for task-uniform sampling and analysis)
    #  - language goal (for analysis; stored as object since it can be variable-length string)
    # ---------------------------------------------------------------------------
    observation_elements.extend([
        # gripper pose has shape (7,) to be compatible with PyTorch ConvNets
        ReplayElement('gripper_pose',
                      (gripper_pose_size,),               np.float32),
        # language goal embedding has shape (1024,) to be compatible with PyTorch ConvNets
        ReplayElement('lang_goal_emb',       (lang_feat_dim,
                                              ),                   np.float32),
        # language token embeddings has shape (77, 1024) to be compatible with PyTorch ConvNets
        ReplayElement('lang_token_embs',
                      (max_token_seq_len, lang_emb_dim,), np.float32),
        ReplayElement('task',  (), str),   # task is a string
        # language goal is an object (stored as object since it can be variable-length string)
        ReplayElement('lang_goal', (1,), object),
    ])

    # flag to indicate whether this transition is from a demo (vs. generated by the agent's own policy) for analysis purposes; stored as bool
    extra_replay_elements = [ReplayElement('demo', (), np.bool_)]

    if not single_process:
        replay_buffer = TaskUniformReplayBuffer(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),          # continuous 8-DoF
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    else:
        replay_buffer = UniformReplayBufferSingleProcess(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    return replay_buffer


# ---------------------------------------------------------------------------
# Internal helpers (identical to manigaussian_bc)
# ---------------------------------------------------------------------------

def _get_action(obs_tp1: Observation):
    """
    Returns the continuous 8-DoF action for the flow-matching head:
      [x, y, z, qx, qy, qz, qw, gripper_open]

    The quaternion is normalized and canonicalized (qw >= 0) so the flow
    regression target is consistent across the dataset.
    """
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    grip = float(obs_tp1.gripper_open)
    continuous_action = np.concatenate([
        obs_tp1.gripper_pose[:3],   # xyz
        quat,                       # normalized + canonical quat (qw >= 0)
        np.array([grip])
    ])
    return continuous_action    


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        description: str = '',
        language_model=None,
        device='cpu'):

    prev_action = None
    obs = inital_obs    # Initial observation is 0

    for k, keypoint in enumerate(episode_keypoints):    # demo[-1].nerf_multi_view_rgb is none 
        obs_tp1 = demo[keypoint]    # target obs is at the keypoint
        obs_tm1 = demo[max(0, keypoint - 1)]  # obs_tm1 is used for extracting the previous action and ignore_collisions flag; if keypoint is 0, we use the initial observation (demo[0]) for these fields since there is no previous step

        action = _get_action(obs_tp1)   # action is the continuous 8-DoF gripper pose at the target step (obs_tp1)
        ignore_collisions = int(obs_tm1.ignore_collisions)  # ignore_collisions flag is taken from obs_tm1 since it indicates whether the transition to obs_tp1 involved a collision, and we want to mask out the flow regression loss for this transition if there was a collision

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = utils.extract_obs(
            obs, t=k, prev_action=prev_action,
            cameras=cameras,
            episode_length=cfg.rlbench.episode_length,
            next_obs=obs_tp1 if not terminal else obs_tm1,
        )
        sentence_emb, token_embs = language_model.extract(description)
        obs_dict['lang_goal_emb'] = sentence_emb[0].float(
        ).detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float(
        ).detach().cpu().numpy()
        obs_dict['lang_goal'] = np.array([description], dtype=object)

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'gripper_pose':      obs_tp1.gripper_pose,
            'task':              task,
            'lang_goal':         np.array([description], dtype=object),
        }
        others.update(obs_dict)   # current-step observation (obs, not obs_tp1)
        # target-step fields overwrite any obs_dict clashes
        others.update(final_obs)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # Final step
    obs_dict_tp1 = utils.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action,
        cameras=cameras,
        episode_length=cfg.rlbench.episode_length,
        next_obs=obs_tp1,
    )
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float(
    ).detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float(
    ).detach().cpu().numpy()
    obs_dict_tp1['lang_goal'] = np.array([description], dtype=object)
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


# ---------------------------------------------------------------------------
# fill_replay / fill_multi_task_replay (unchanged logic, just name-updated)
# ---------------------------------------------------------------------------

def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                language_model=None,
                device='cpu',
                keypoint_method='heuristic'):

    logging.getLogger().setLevel(cfg.framework.logging_level)
    logging.debug('Filling %s replay ...' % task)

    for d_idx in range(num_demos):
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx,
        )[0]

        descs = demo._observations[0].misc['descriptions']
        episode_keypoints = demo_loading_utils.keypoint_discovery(
            demo, method=keypoint_method
        )

        if rank == 0:
            logging.info(
                f"Loading Demo({d_idx}) - found {len(episode_keypoints)} "
                f"keypoints - {task}"
            )

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            obs = demo[i]
            desc = descs[0]

            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break

            _add_keypoints_to_replay(
                cfg, task, replay, obs, demo, episode_keypoints, cameras,
                description=desc,
                language_model=language_model,
                device=device,
            )

    logging.debug('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           keypoint_method: str = 'heuristic',
                           fabric: Fabric = None):

    manager = Manager()
    store = manager.dict()

    if hasattr(replay, '_task_idxs'):
        del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)

    device = fabric.device if fabric is not None else None
    language_model = create_language_model(
        name=cfg.method.language_model, device=device
    )

    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device(
                'cuda:%s' % (e_idx % torch.cuda.device_count())
                if torch.cuda.is_available() else 'cpu'
            )
            p = Process(
                target=fill_replay,
                args=(cfg, obs_config, rank, replay, task, num_demos,
                      demo_augmentation, demo_augmentation_every_n,
                      cameras, language_model, model_device,
                      keypoint_method),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


# ---------------------------------------------------------------------------
# Fast Zarr-based DataLoader (Recommendation 1 + 3)
# ---------------------------------------------------------------------------

def create_zarr_loader(
        zarr_path: str,
        batch_size: int,
        cameras: List[str] = None,
        num_workers: int = 4,
        mem_gb: float = 8.0,
        relative_action: bool = False,
        copies: int = 10,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
) -> tuple:
    """
    Build a DataLoader backed by ``ManiFlowZarrDataset``.

    Applies all three DataLoader optimisations from 3d_flowmatch_actor:
      - ``prefetch_factor=4``    → keep 4 batches pre-loaded in worker RAM
      - ``persistent_workers``   → workers stay alive across epochs (no fork overhead)
      - ``pin_memory=True``      → faster H→D transfer for CUDA tensors
      - Seeded ``worker_init_fn``→ reproducible augmentation per worker
      - ``DistributedSampler``   → proper data sharding in DDP runs

    Parameters
    ----------
    zarr_path        : str        Path to ``{split}.zarr``.
    batch_size       : int        Batch size per GPU.
    cameras          : list[str]  Camera names (must match converter order).
    num_workers      : int        DataLoader worker processes (default 4).
    mem_gb           : float      LRU cache per worker process in GiB.
    relative_action  : bool       Return delta actions (SO(3) relative).
    copies           : int        Virtual dataset multiplier.
    distributed      : bool       Use DistributedSampler (for DDP).
    rank / world_size: int        DDP rank / world-size.
    seed             : int        Global RNG seed for reproducibility.

    Returns
    -------
    (loader, dataset, sampler)
    """
    dataset = ManiFlowZarrDataset(
        zarr_path=zarr_path,
        cameras=cameras or ["front"],
        mem_gb=mem_gb,
        relative_action=relative_action,
        copies=copies,
    )

    def _seed_worker(worker_id: int):
        """Seed numpy + random in each DataLoader worker for reproducibility."""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        collate_fn=maniflow_collate_fn,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        generator=g,
        prefetch_factor=4 if num_workers > 0 else None,   # Recommendation 3
        persistent_workers=num_workers > 0,               # Recommendation 3
    )
    return loader, dataset, sampler


# ---------------------------------------------------------------------------
# GPU Pre-processor factory (Recommendation 2)
# ---------------------------------------------------------------------------

def create_gpu_preprocessor(
        image_size: tuple = (128, 128),
        augment_prob: float = 0.8,
        crop_prob: float = 0.1,
        custom_imsize: Optional[int] = None,
) -> ManiFlowGPUPreprocessor:
    """
    Instantiate a ``ManiFlowGPUPreprocessor`` (GPU-side depth→PCD + kornia
    augmentation).

    The returned module should be moved to the training device:
        preprocessor = create_gpu_preprocessor(...).to(device)

    During training call:
        rgb_proc, pcd_proc = preprocessor(rgb, depth, extrinsics, intrinsics,
                                          augment=True)
    During evaluation:
        rgb_proc, pcd_proc = preprocessor(rgb, depth, extrinsics, intrinsics,
                                          augment=False)
    """
    return ManiFlowGPUPreprocessor(
        image_size=image_size,
        augment_prob=augment_prob,
        crop_prob=crop_prob,
        custom_imsize=custom_imsize,
    )


# ---------------------------------------------------------------------------
# create_agent
# ---------------------------------------------------------------------------

def create_agent(cfg: DictConfig) -> PreprocessAgent:
    """
    Build the ManiFlow agent stack from config.

    Key config keys (under cfg.method):
      voxel_sizes, final_dim, language_model_dim,
      # Transformer denoising head:
      embedding_dim            (default: 120)
      num_attn_heads           (default: 8)
      num_shared_attn_layers   (default: 4)
      voxel_token_downsample   (default: 5)
      denoise_timesteps        (default: 100)
      action_dim               (default: 8)
      # legacy MLP stubs (kept for YAML back-compat, not used functionally):
      flow_context_dim, flow_hidden_dim, flow_num_layers
    """
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    # Transformer head hyperparameters (read from cfg, fall back to defaults)
    embedding_dim = getattr(cfg.method, 'embedding_dim',          120)
    num_attn_heads = getattr(cfg.method, 'num_attn_heads',         8)
    num_shared_attn_layers = getattr(cfg.method, 'num_shared_attn_layers', 4)
    voxel_token_downsample = getattr(cfg.method, 'voxel_token_downsample', 5)
    num_fps_tokens = getattr(cfg.method, 'num_fps_tokens',         512)
    denoise_timesteps = getattr(cfg.method, 'denoise_timesteps',      100)
    action_dim = getattr(cfg.method, 'action_dim',             8)
    # Legacy stubs (ignored by new VoxelFlowEncoder but kept for YAML compat)
    flow_context_dim = getattr(cfg.method, 'flow_context_dim',  256)
    flow_hidden_dim = getattr(cfg.method, 'flow_hidden_dim',   512)
    flow_num_layers = getattr(cfg.method, 'flow_num_layers',   4)

    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):

        flow_encoder = VoxelFlowEncoder(
            voxel_size=vox_size,
            initial_dim=3 + 3 + 1 + 3,   # rgb + xyz + density + normal
            low_dim_size=4,
            im_channels=cfg.method.final_dim,
            lang_feat_dim=1024,             # CLIP sentence embedding
            lang_emb_dim=cfg.method.language_model_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            voxel_token_downsample=voxel_token_downsample,
            num_fps_tokens=num_fps_tokens,
            coordinate_bounds=list(depth_0bounds),
            denoise_timesteps=denoise_timesteps,
            activation=cfg.method.activation,
            lang_fusion_type=cfg.method.lang_fusion_type,
            # legacy stubs
            context_dim=flow_context_dim,
            flow_hidden_dim=flow_hidden_dim,
            flow_num_layers=flow_num_layers,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            cfg=cfg,
        )

        qattention_agent = ManiFlowBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            flow_encoder=flow_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth -
                                                   1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            denoise_timesteps=denoise_timesteps,
            action_dim=action_dim,
            pos_loss_weight=getattr(cfg.method, 'pos_loss_weight',  30.0),
            rot_loss_weight=getattr(cfg.method, 'rot_loss_weight',  10.0),
            grip_loss_weight=getattr(cfg.method, 'grip_loss_weight',  1.0),
            cfg=cfg.method,
        )
        qattention_agents.append(qattention_agent)

    flow_stack_agent = ManiFlowStackAgent(
        qattention_agents=qattention_agents,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(pose_agent=flow_stack_agent)
    return preprocess_agent
