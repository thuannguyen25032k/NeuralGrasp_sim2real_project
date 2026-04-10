import os
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from termcolor import cprint
from lightning_fabric import Fabric as _LFabric
import types
L = types.SimpleNamespace(Fabric=_LFabric)
from tqdm import tqdm


def run_seed(
            rank,
            cfg: DictConfig,
            obs_config: ObservationConfig,
            cams,
            multi_task,
            seed,
            world_size,
            fabric: L.Fabric = None,
            ) -> None:
    
    if fabric is not None:
        rank = fabric.global_rank
    else:
        dist.init_process_group("gloo",
                        rank=rank,
                        world_size=world_size)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    replay_path = os.path.join(cfg.replay.path, 'seed%d' % seed)
    
    if cfg.method.name == 'GNFACTOR_BC':
        from agents import gnfactor_bc
        replay_buffer = gnfactor_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg)

        gnfactor_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, 0,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method
        )

        agent = gnfactor_bc.launch_utils.create_agent(cfg)
        
    elif cfg.method.name == 'ManiGaussian_BC':
        from agents import manigaussian_bc
        replay_buffer = manigaussian_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg)
        
        if cfg.replay.use_disk and (os.path.exists(replay_path) and len(os.listdir(replay_path)) > 1):  # default: True
            logging.info(f"Found replay files in {replay_path}. Loading...")
            replay_files = [os.path.join(replay_path, f) for f in os.listdir(replay_path) if f.endswith('.replay')]

            # Load replay files in parallel using a thread pool (I/O bound: pickle reads)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing

            def _load_one(replay_file):
                with open(replay_file, 'rb') as f:
                    return pickle.load(f)

            num_io_workers = min(16, multiprocessing.cpu_count(), len(replay_files))
            loaded = [None] * len(replay_files)
            with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
                future_to_idx = {executor.submit(_load_one, f): i for i, f in enumerate(replay_files)}
                for future in tqdm(as_completed(future_to_idx), total=len(replay_files), desc="Loading replay files"):
                    idx = future_to_idx[future]
                    try:
                        loaded[idx] = future.result()
                    except Exception as e:
                        logging.error(f"Error loading replay file {replay_files[idx]}: {e}")

            for replay_data in tqdm(loaded, desc="Adding to replay buffer"):
                if replay_data is not None:
                    try:
                        replay_buffer._add(replay_data)
                    except Exception as e:
                        logging.error(f"Error adding replay data: {e}")
        else:
            manigaussian_bc.launch_utils.fill_multi_task_replay(
                cfg, obs_config, 0,
                replay_buffer, tasks, cfg.rlbench.demos,
                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                cams, cfg.rlbench.scene_bounds,
                cfg.method.voxel_sizes, cfg.method.bounds_offset,
                cfg.method.rotation_resolution, cfg.method.crop_augmentation,
                keypoint_method=cfg.method.keypoint_method,
                fabric=fabric,
            )

        agent = manigaussian_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'ManiFlow_BC':
        from agents import maniflow_bc
        from agents.maniflow_bc.launch_utils import create_zarr_loader

        # Canonical camera order used when the zarr was created (matches
        # helpers/utils.py extract_obs / gen_demonstrations ordering).
        _ALL_CAMS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        _requested = list(cams)
        camera_inds = [_ALL_CAMS.index(c) for c in _requested if c in _ALL_CAMS]
        cprint(f'[ManiFlow_BC] camera_inds={camera_inds} for cameras={_requested}', 'cyan')

        loader, _, sampler = create_zarr_loader(
            zarr_path=cfg.replay.zarr_path,
            instructions_path=cfg.replay.zarr_instructions,
            batch_size=cfg.replay.batch_size,
            cameras=_requested,
            camera_inds=camera_inds,
            chunk_size=cfg.method.action_chunk_size,
            num_workers=cfg.replay.zarr_num_workers,
            mem_gb=cfg.replay.zarr_mem_gb,
            copies=cfg.replay.zarr_copies,
            distributed=cfg.ddp.num_devices > 1,
            rank=fabric.global_rank,
            world_size=cfg.ddp.num_devices,
        )
        replay_buffer = None
        zarr_loader   = loader
        cprint(f'[ManiFlow_BC] Using Zarr DataLoader: {cfg.replay.zarr_path}', 'green')

        agent = maniflow_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_BC':
        from agents import peract_bc
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    # ManiFlow_BC always uses the Zarr loader; all other methods use YARR.
    if cfg.method.name == 'ManiFlow_BC':
        wrapped_replay = None
        # zarr_loader already set above
    else:
        wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
        zarr_loader = None

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')  # load from the last checkpoint

    logdir = os.path.join(cwd, 'seed%d' % seed)

    cprint(f'Project path: {weightsdir}', 'cyan')

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        zarr_loader=zarr_loader,
        train_device=rank,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        fabric=fabric)
    cprint('Starting training!!', 'green')
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()