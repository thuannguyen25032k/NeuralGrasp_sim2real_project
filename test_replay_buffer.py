# A quick test script to put in your ManiGaussian root folder
import numpy as np
from third_party.YARR.yarr.replay_buffer.uniform_replay_buffer import ManiGaussianFlowReplayBuffer
import os
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=0, world_size=1)
# 1. Instantiate your modified buffer (using dummy shapes for testing)
buffer = ManiGaussianFlowReplayBuffer(
    batch_size=4,
    timesteps=1,
    action_horizon=8, # Your new parameter!
    replay_capacity=1000,
    action_shape=(7,), # Example: 6DoF + 1 gripper
    action_dtype=np.float32,
    # ... mock the rest of your required parameters ...
)

# 2. Add some dummy transitions to simulate a short episode ending
for i in range(15):
    terminal = 1 if i == 14 else 0
    buffer.add(action=np.random.rand(7), reward=0, terminal=terminal, timeout=0)

# 3. Sample a batch and verify the shape
try:
    batch = buffer.sample_transition_batch(batch_size=4)
    print("SUCCESS! Action tensor shape:", batch['action'].shape) 
    # EXPECTED OUTPUT: (4, 8, 7) -> [Batch, Horizon, Action_Dim]
except Exception as e:
    print("CRASHED:", e)