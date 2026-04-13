"""
Rectified Flow scheduler for continuous action denoising.
Adapted from: https://github.com/cloneofsimo/minRF
"""

import torch
import numpy as np


class RFScheduler:
    """
    Rectified Flow (flow matching) noise scheduler.

    During training:  z_t = (1 - t) * x_clean + t * noise   (t in [0,1])
    Model predicts:   v = noise - x_clean  (the "velocity")
    Inference step:   z_{t'} = z_t - (t - t') * v_theta(z_t, t)

    Supports three noise timestep samplers:
      - 'logit_normal'  (default, recommended for RF)
      - 'uniform'
      - 'pi0'
    """

    def __init__(self, noise_sampler: str = "logit_normal",
                 noise_sampler_config: dict = None):
        self.noise_sampler = noise_sampler
        self.noise_sampler_config = noise_sampler_config or {
            "mean": 0.0, "std": 1.5}

    # ------------------------------------------------------------------
    # Inference setup
    # ------------------------------------------------------------------
    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        """Build a timestep grid in (0, 1] descending.

        FIX #5: When the sampler is 'logit_normal', use a logit-normal-warped
        grid rather than a uniform one.  During training, timesteps are drawn
        from logit-normal(mean, std), which concentrates mass around the
        midpoint of [0, 1].  A uniform inference grid therefore over-spends
        steps near t=0 and t=1 (where the velocity field is nearly zero) and
        under-spends near t=0.5 (where most of the trajectory curvature is).
        Warping the grid to match the training distribution halves the number
        of steps needed for the same trajectory quality.

        For 'uniform' and 'pi0' samplers the grid stays uniform (unchanged).
        """
        if self.noise_sampler == "logit_normal":
            mean = self.noise_sampler_config.get("mean", 0.0)
            std  = self.noise_sampler_config.get("std",  1.5)
            # Map a uniform grid through the logistic CDF to get a grid whose
            # density matches the logit-normal training distribution.
            # u ∈ (0, 1) linearly, then t = sigmoid(u_logit * std + mean)
            u = np.linspace(0.0, 1.0, num_inference_steps + 2)[1:-1]  # open interval
            u_logit = np.log(u / (1.0 - u + 1e-8))                    # logit transform
            t_warped = 1.0 / (1.0 + np.exp(-(u_logit * std + mean)))  # sigmoid back
            t_warped = np.clip(t_warped, 1e-4, 1.0 - 1e-4)
            # Sort descending (t=1 is pure noise, t=0 is clean)
            timesteps_np = np.sort(t_warped)[::-1].astype(np.float32)
        else:
            # Original uniform grid for other samplers
            timesteps_np = (
                np.arange(num_inference_steps + 1)[1:][::-1].astype(np.float32)
                / num_inference_steps
            )

        self.timesteps = torch.from_numpy(timesteps_np.copy()).to(device)
        # prev_t[i] = timesteps[i+1], with prev_t[-1] = 0
        self.prev_timesteps = torch.cat((
            self.timesteps[1:],
            torch.zeros(1, device=device, dtype=self.timesteps.dtype)
        ))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def sample_noise_step(self, num_noise: int, device) -> torch.Tensor:
        """Sample random timestep t in (0, 1) for training."""
        if self.noise_sampler == "uniform":
            return torch.zeros(num_noise, dtype=torch.float32, device=device).uniform_()
        elif self.noise_sampler == "logit_normal":
            s = torch.zeros(num_noise, dtype=torch.float32, device=device).normal_(
                mean=self.noise_sampler_config["mean"],
                std=self.noise_sampler_config["std"]
            )
            return torch.sigmoid(s)
        elif self.noise_sampler == "pi0":
            alpha, beta = 1.5, 1.0
            return torch.distributions.Beta(alpha, beta).sample(
                (num_noise,)
            ).to(device).clamp(max=0.999)
        else:
            raise NotImplementedError(
                f"Sampler '{self.noise_sampler}' not implemented.")

    def add_noise(self, original_samples: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate: z_t = (1-t)*x + t*noise."""
        t = timesteps.view([original_samples.size(0),
                            *([1] * (original_samples.dim() - 1))])
        return ((1.0 - t) * original_samples + t * noise).to(original_samples.dtype)

    def prepare_target(self, noise: torch.Tensor,
                       gt: torch.Tensor) -> torch.Tensor:
        """Training target = velocity = noise − x_clean."""
        return noise - gt

    # ------------------------------------------------------------------
    # Inference step
    # ------------------------------------------------------------------
    def step(self, model_output: torch.Tensor,
             timestep_ind: int,
             sample: torch.Tensor):
        """One Euler step:  z_{t'} = z_t - dt * v_theta."""
        t = self.timesteps[timestep_ind].to(model_output.device)
        prev_t = self.prev_timesteps[timestep_ind].to(model_output.device)
        dt = t - prev_t
        prev_sample = sample - dt * model_output
        return _StepOutput(prev_sample=prev_sample)


class _StepOutput:
    """Simple container matching the HuggingFace scheduler API."""

    def __init__(self, prev_sample: torch.Tensor):
        self.prev_sample = prev_sample
