import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.manigaussian_bc.qattention_manigaussian_bc_agent import QAttentionPerActBCAgent

# FIX: Point to the new modeling directory
from .modeling.noise_scheduler.rectified_flow import RFScheduler
from .modeling.policy.denoise_actor_3d import DenoiseActor

class ManiGaussianFlowBCAgent(QAttentionPerActBCAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.action_horizon = 16 
        self.action_dim = 3
        
        # 1. Instantiate the Denoising Actor
        self.denoise_actor = DenoiseActor(
            # We are leaving this empty for a second so we can see what shapes ManiGaussian feeds it
        )
        
        # 2. Instantiate the Flow Scheduler
        self.noise_scheduler = RFScheduler(
            noise_sampler="logit_normal"
        )

    def _q(self, obs, *args, **kwargs):
        """
        TRAINING FORWARD PASS
        """
        # =====================================================================
        # 1. EXTRACT 3D FEATURES (Using base ManiGaussian logic)
        # =====================================================================
        # This will likely crash first because we need to intercept the voxel 
        # grid before it goes into the old PerceiverIO transformer.
        # We will debug this exact shape in the next step.
        
        # =====================================================================
        # 2. FLOW MATCHING MATH
        # =====================================================================
        # Get the GT action sequence from your chunked replay buffer
        gt_actions = kwargs.get('actions') # Shape: (Batch, 16, 3)
        batch_size = gt_actions.shape[0]
        device = gt_actions.device
        
        # A. Sample random timesteps between 0 and 1
        timesteps = self.noise_scheduler.sample_noise_step(batch_size, device)
        
        # B. Generate pure Gaussian noise matching the action shape
        noise = torch.randn_like(gt_actions)
        
        # C. Corrupt the GT actions with the noise based on the timestep
        noisy_actions = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)
        
        # D. Predict the flow (velocity) using Olivier's Actor
        # (This will definitely throw a dimension mismatch error next, which is what we want!)
        pred_flow = self.denoise_actor(
            noisy_actions, 
            timesteps, 
            # Needs the 3D voxel features here
        )
        
        # E. Calculate Rectified Flow Loss (MSE between prediction and target vector)
        target_flow = self.noise_scheduler.prepare_target(noise, gt_actions)
        flow_loss = F.mse_loss(pred_flow, target_flow)
        
        # =====================================================================
        # 3. COMBINE LOSSES
        # =====================================================================
        # For now, let's just return the flow loss so we can isolate the bugs.
        # Once the shapes match, we will add the Gaussian Splatting rendering loss back in.
        
        return {
            'flow_loss': flow_loss,
            'total_loss': flow_loss # YARR expects 'total_loss' to backpropagate
        }
    def act(self, obs, lang_goal):
        """
        EVALUATION / INFERENCE FORWARD PASS
        """
        # 1. Extract features
        voxel_grid = self.extract_3d_features(obs)
        lang_embed = self.extract_lang_features(lang_goal)
        
        # 2. Generate pure random noise
        noisy_actions = torch.randn((1, self.action_horizon, self.action_dim)).to(obs.device)
        
        # 3. DENOISE: Step through the scheduler to predict the clean trajectory
        clean_actions = self.noise_scheduler.step(
            self.denoise_actor, 
            noisy_actions, 
            voxel_grid, 
            lang_embed
        )
        
        return clean_actions