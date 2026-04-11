"""
QAttentionStackAgent for ManiFlow.
Mirrors agents/manigaussian_bc/qattention_stack_agent.py but adapted for
the continuous-action ManiFlowBCAgent.
"""

from typing import List

import numpy as np
import torch
from yarr.agents.agent import Agent, ActResult, Summary
from termcolor import cprint

from agents.maniflow_bc.qattention_maniflow_agent import ManiFlowBCAgent

NAME = 'ManiFlowStackAgent'


class ManiFlowStackAgent(Agent):

    def __init__(self,
                 qattention_agents: List[ManiFlowBCAgent],
                 camera_names: List[str]):
        super().__init__()
        self._qattention_agents = qattention_agents
        self._camera_names      = camera_names

    def build(self, training: bool, device=None,
              use_ddp: bool = True, **kwargs) -> None:
        self._device = device if device is not None else torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, self._device, use_ddp, **kwargs)

    # ------------------------------------------------------------------
    def update(self, step: int, replay_sample: dict, **kwargs) -> dict:
        if (replay_sample['nerf_multi_view_rgb'] is None
                or replay_sample['nerf_multi_view_rgb'][0, 0] is None):
            cprint("ManiFlowStackAgent: no nerf rgb in sample", "red")

        total_losses = 0.
        for qa in self._qattention_agents:
            update_dict   = qa.update(step, replay_sample, **kwargs)
            replay_sample.update(update_dict)
            total_losses += update_dict['total_loss']
        return {'total_losses': total_losses}

    # ------------------------------------------------------------------
    def act(self, step: int, observation: dict,
            deterministic: bool = False) -> ActResult:
        """
        Run each layer agent in sequence.
        Returns a continuous 8-DoF action:
            [x, y, z, qx, qy, qz, qw, gripper_open]
        """
        observation_elements = {}
        infos                = {}
        final_act_results    = None

        for depth, qagent in enumerate(self._qattention_agents):
            act_results = qagent.act(step, observation, deterministic)
            final_act_results = act_results

            attention_coordinate = act_results.observation_elements[
                'attention_coordinate'
            ].cpu().numpy()
            observation_elements[
                'attention_coordinate_layer_%d' % depth
            ] = attention_coordinate[0]

            # Pass attention coordinate forward for multi-layer setups
            observation['attention_coordinate'] = (
                act_results.observation_elements['attention_coordinate']
            )
            # prev_layer_voxel_grid / prev_layer_bounds live in `info` to
            # avoid YARR's rollout_generator calling np.array() on them.
            observation['prev_layer_voxel_grid'] = (
                act_results.info['prev_layer_voxel_grid']
            )
            observation['prev_layer_bounds'] = (
                act_results.info['prev_layer_bounds']
            )
            infos.update(act_results.info)

        # final_act_results.action is a continuous 8-DoF numpy array
        # [x, y, z, qx, qy, qz, qw, gripper]
        # Append ignore_collisions (from live observation) to form the 9-DoF
        # action expected by RLBench's action_mode.
        # observation['ignore_collisions'] is shape (1,) numpy array from YARR.
        ic_raw = observation['ignore_collisions']
        ignore_collisions = float(ic_raw.item() if hasattr(ic_raw, 'item') else ic_raw)
        continuous_action = np.concatenate([
            final_act_results.action,
            [ignore_collisions],
        ])

        observation_elements.update(final_act_results.observation_elements)

        return ActResult(
            continuous_action,
            observation_elements=observation_elements,
            info=infos,
        )

    # ------------------------------------------------------------------
    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def update_wandb_summaries(self):
        summaries = {}
        for qa in self._qattention_agents:
            summaries.update(qa.update_wandb_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)

    def load_clip(self):
        for qa in self._qattention_agents:
            qa.load_clip()

    def unload_clip(self):
        for qa in self._qattention_agents:
            qa.unload_clip()
