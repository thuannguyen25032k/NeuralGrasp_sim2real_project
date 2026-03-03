import torch
import torch.nn as nn
import torch.optim as optim
from src.networks.flow_matching_policy import SplatFlowNetwork

def train_flow_policy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model = SplatFlowNetwork(action_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # --- DUMMY DATA FOR TESTING ---
    # In reality, this comes from your DataLoader parsing SplatSim datasets
    batch_size = 32
    dummy_images = torch.rand(batch_size, 3, 256, 256).to(device)
    dummy_text_embeds = torch.rand(batch_size, 512).to(device) # From the VLM
    true_actions = torch.rand(batch_size, 7).to(device) # The target x_1
    
    epochs = 100
    model.train()

    print("Starting Flow Matching Training Loop...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 2. Sample random noise (x_0) and a random time step (t) for the batch
        noise = torch.randn_like(true_actions).to(device)
        t = torch.rand(batch_size, 1).to(device)

        # 3. Interpolate to create the intermediate noisy action (x_t)
        # x_t = t * x_1 + (1 - t) * x_0
        x_t = t * true_actions + (1 - t) * noise

        # 4. Calculate the true direction we want the model to predict
        # u_t = x_1 - x_0
        target_velocity = true_actions - noise

        # 5. Forward Pass: Ask the model to predict the velocity
        # The model looks at the image, text, current time, and the noisy action
        predicted_velocity = model(dummy_images, dummy_text_embeds, t, x_t)

        # 6. Calculate Loss and Backpropagate
        loss = criterion(predicted_velocity, target_velocity)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | MSE Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_flow_policy()