import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ManiGaussianFlowDataset(Dataset):
    def __init__(self, data_dir, action_horizon=8):
        """
        data_dir: Path to the specific task folder, 
                  e.g., 'data/train_data/open_drawer'
        """
        self.episodes_dir = os.path.join(data_dir, "all_variations/episodes")
        # Find all episode folders (episode0, episode1, ...)
        self.episode_paths = [
            os.path.join(self.episodes_dir, d) 
            for d in os.listdir(self.episodes_dir) 
            if d.startswith("episode")
        ]
        self.action_horizon = action_horizon
        
        # Build an index of all available timesteps across all episodes
        self.samples = []
        for ep_path in self.episode_paths:
            rgb_dir = os.path.join(ep_path, "front_rgb")
            if not os.path.exists(rgb_dir): 
                continue
            
            # Count frames in this episode
            num_frames = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
            
            for t in range(num_frames):
                self.samples.append((ep_path, t, num_frames))
                
        print(f"Dataset initialized with {len(self.samples)} total transitions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_path, t, num_frames = self.samples[idx]

        # 1. VISUALS: Load as lightweight types (3DFA Optimization)
        # RGB -> uint8
        img_path = os.path.join(ep_path, "front_rgb", f"{t}.png")
        img_uint8 = cv2.imread(img_path)[..., ::-1].copy() # Convert BGR to RGB
        
        # Depth -> float16 (GPU later)
        depth_path = os.path.join(ep_path, "front_depth", f"{t}.png")
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_float16 = depth_img.astype(np.float16)

        # 2. ACTIONS: Load sequence with sliding window
        with open(os.path.join(ep_path, "low_dim_obs.pkl"), 'rb') as f:
            obs_list = pickle.load(f)
            
        available_steps = min(self.action_horizon, num_frames - t)
        future_obs = obs_list[t : t + available_steps]
        
        
        action_chunk = np.array([obs.joint_positions for obs in future_obs])

        # Padding: If we are too close to the end of the episode, repeat the last action
        if available_steps < self.action_horizon:
            pad_len = self.action_horizon - available_steps
            last_action = action_chunk[-1:]
            padding = np.repeat(last_action, pad_len, axis=0)
            action_chunk = np.concatenate([action_chunk, padding], axis=0)

        # 3. GAUSSIANS: Provide paths to the GT rendering data
        nerf_images_dir = os.path.join(ep_path, "nerf_data", str(t), "images")
        nerf_poses_dir = os.path.join(ep_path, "nerf_data", str(t), "poses")

        return {
            "image": img_uint8,
            "depth": depth_float16,
            "actions": action_chunk.astype(np.float32),
            # We return strings for directories so the GS branch can load them dynamically
            "nerf_images_dir": nerf_images_dir, 
            "nerf_poses_dir": nerf_poses_dir
        }

def create_dataloader(data_dir: str, batch_size:int = 4, action_horizon:int = 8, num_workers:int = 4):
    dataset = ManiGaussianFlowDataset(data_dir, action_horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if __name__ == "__main__":
    # Point to raw data folder
    test_data_dir = "/home/yosri/Desktop/NeuralGrasp_sim2real_project/data/train_data/open_drawer"
    
    # Create the dataloader
    dataloader = create_dataloader(test_data_dir, batch_size=4, action_horizon=8, num_workers=0)
    
    # Grab one batch
    batch = next(iter(dataloader))
    
    print("\n--- BATCH OUTPUT SUCCESS ---")
    print("Images Shape (uint8):", batch['image'].shape)
    print("Depth Shape (float16):", batch['depth'].shape)
    print("Actions Shape (float32):", batch['actions'].shape) # Should be [4, 8, 7]
    print("NeRF Path Example:", batch['nerf_images_dir'][0])