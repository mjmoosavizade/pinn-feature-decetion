# train_pinn.py

import torch
import torch.nn as nn
from pinn_model import PINN, pde_residual_anisotropic
from data_preparation import load_images_from_directory, detect_keypoints, prepare_training_data, generate_collocation_points
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Hyperparameters
layers = [2, 50, 50, 50, 1]
lambda_pde = 1.0
learning_rate = 1e-3
num_epochs = 200
batch_size = 8192  # Adjust based on your GPU memory
colloc_batch_size = 8192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define k for the anisotropic diffusion PDE
k = 0.1  # Adjust as needed

def loss_function(model, x_data, u_data, x_colloc, k):
    # Data Loss
    u_pred = model(x_data)
    mse_data = nn.MSELoss()(u_pred, u_data)

    # PDE Loss
    residual = pde_residual_anisotropic(model, x_colloc, k)
    mse_pde = torch.mean(residual**2)

    # Total Loss
    loss = mse_data + lambda_pde * mse_pde
    return loss, mse_data.item(), mse_pde.item()

def main():
    # Load and prepare data
    selected_images_folder = 'data'  # Update this path
    images_list, image_names_list = load_images_from_directory(selected_images_folder, max_images=1)
    keypoints_list = detect_keypoints(images_list)  # Not needed for anisotropic diffusion
    x_data, u_data, keypoints, h, w = prepare_training_data(images_list, keypoints_list)
    x_colloc = generate_collocation_points(100000, lb=torch.tensor([0.0, 0.0]), ub=torch.tensor([1.0, 1.0]))

    # Move data to device
    x_data = x_data.to(device)
    u_data = u_data.to(device)
    x_colloc = x_colloc.to(device)

    # Initialize model and optimizer
    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision training

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample mini-batches
        idx_data = torch.randint(0, x_data.shape[0], (batch_size,))
        x_data_batch = x_data[idx_data]
        u_data_batch = u_data[idx_data]

        idx_colloc = torch.randint(0, x_colloc.shape[0], (colloc_batch_size,))
        x_colloc_batch = x_colloc[idx_colloc]

        # Compute loss with mixed precision
        with autocast():
            loss, mse_data, mse_pde = loss_function(model, x_data_batch, u_data_batch, x_colloc_batch, k)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item():.5f}, Data Loss: {mse_data:.5f}, PDE Loss: {mse_pde:.5f}')

    # Save the trained model
    torch.save(model.state_dict(), 'pinn_model.pth')
    print("Model saved to 'pinn_model.pth'.")

    # Visualization
    visualize_results(model, x_data, u_data, h, w, device)

def visualize_results(model, x_data, u_data, h, w, device):
    model.eval()
    with torch.no_grad():
        x_plot = x_data.cpu().numpy()
        u_plot = u_data.cpu().numpy()
        u_pred = model(x_data.to(device)).cpu().numpy()

        # Reshape for plotting using actual image dimensions
        u_plot_img = u_plot.reshape((h, w))
        u_pred_img = u_pred.reshape((h, w))

        # Plot original image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(u_plot_img, cmap='gray', extent=[0, 1, 0, 1])
        plt.title('Original Image')
        plt.axis('off')

        # Plot predicted image
        plt.subplot(1, 3, 2)
        plt.imshow(u_pred_img, cmap='gray', extent=[0, 1, 0, 1])
        plt.title('PINN Prediction')
        plt.axis('off')

        # Plot difference
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(u_plot_img - u_pred_img), cmap='hot', extent=[0, 1, 0, 1])
        plt.title('Difference')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('pinn_results.png')
        print("Visualization saved to 'pinn_results.png'.")

if __name__ == '__main__':
    main()
