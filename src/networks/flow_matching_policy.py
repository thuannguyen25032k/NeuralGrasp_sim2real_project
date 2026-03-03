import torch
import torch.nn as nn
import torchvision.models as models

class SplatFlowNetwork(nn.Module):
    def __init__(self, action_dim=7, text_embed_dim=512):
        super().__init__()
        
        # 1. Image Encoder (The Vision component)
        # Using a pre-trained ResNet to extract features from the RGB image
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer, we just want the raw feature embeddings
        self.vision_encoder.fc = nn.Identity() 
        vision_feature_dim = 512
        
        # 2. Flow Matching Time Encoder
        # Flow models need to know the current 'time' (t) in the generation process
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )
        
        # 3. The Core Action Predictor
        # Combines the image, text, and time to predict the action vector field
        combined_dim = vision_feature_dim + text_embed_dim + 128 + action_dim
        
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, action_dim) # Outputs the 7 robot joint velocities
        )

    def forward(self, rgb_image, text_embedding, time_step, noisy_action):
        # Extract features from the image
        img_features = self.vision_encoder(rgb_image)
        
        # Encode the time step
        t_features = self.time_mlp(time_step)
        
        # Concatenate everything into one massive context vector
        combined_features = torch.cat([img_features, text_embedding, t_features, noisy_action], dim=1)
        
        # Predict the action
        action_vector = self.action_head(combined_features)
        return action_vector