import gymnasium as gym
import numpy as np

# class DummySplatSimEnv(gym.Env):
#     def __init__(self):
#         super().__init__()
#         # Define action space (e.g., 6DoF end-effector + 1DoF gripper)
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
#         # Define observation space (e.g., 256x256 RGB image)
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 256, 256), dtype=np.uint8)

#     def step(self, action):
#         # Apply action, get new dummy image, calculate reward
#         dummy_obs = self.observation_space.sample()
#         return dummy_obs, 0.0, False, False, {}

#     def reset(self, seed=None):
#         dummy_obs = self.observation_space.sample()
#         return dummy_obs, {}