import torch
from pinn_model import PINN
from data_preparation import load_images_from_directory, detect_keypoints, prepare_training_data
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def visualize_results(model, x_data, u_data, h, w, device):
    model.eval()
    with torch.no_grad():
        x_plot = x_data.cpu().numpy()
        u_plot = u_data.cpu().numpy()
        u_pred = model(x_data.to(device)).cpu().numpy()

        # Reshape for plotting using actual image dimensions
        u_plot_img = u_plot.reshape((h, w))
        u_pred_img = u_pred.reshape((h, w))

        # Compute error metrics
        mse = np.mean((u_plot - u_pred) ** 2)
        mae = np.mean(np.abs(u_plot - u_pred))
        max_pixel = 1.0  # Since images are normalized
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        ssim_value = ssim(u_plot_img, u_pred_img, data_range=u_plot_img.max() - u_plot_img.min())

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
        print(f"Structural Similarity Index Measure (SSIM): {ssim_value}")

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

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    layers = [2, 100, 100, 100, 100, 1]  # Ensure this matches your trained model
    model = PINN(layers).to(device)
    model.load_state_dict(torch.load('pinn_model.pth', map_location=device))
    print("Model loaded from 'pinn_model.pth'.")

    # Load and prepare data
    selected_images_folder = 'data'  # Update this path
    images_list, image_names_list = load_images_from_directory(selected_images_folder, max_images=1)
    keypoints_list = detect_keypoints(images_list)

    # Prepare training data and get h and w
    x_data, u_data, keypoints, h, w = prepare_training_data(images_list, keypoints_list)

    # Move data to device
    x_data = x_data.to(device)
    u_data = u_data.to(device)

    # Visualization
    visualize_results(model, x_data, u_data, h, w, device)

if __name__ == '__main__':
    main()
