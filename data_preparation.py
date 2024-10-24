# data_preparation.py

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pyDOE import lhs

def load_images_from_directory(directory_path, max_images=None):
    images = []
    image_names = []
    image_files = [f for f in sorted(os.listdir(directory_path)) if f.lower().endswith(('.jpg', '.png'))]

    # Limit the number of images
    if max_images is not None:
        image_files = image_files[:max_images]

    for file_name in tqdm(image_files, desc='Loading images'):
        file_path = os.path.join(directory_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
            image_names.append(file_name)
        else:
            print(f"Warning: Failed to load image '{file_path}'")

    return images, image_names

def detect_keypoints(images_list):
    # Initialize the FAST detector
    fast = cv2.FastFeatureDetector_create()

    keypoints_list = []

    for image in images_list:
        kp = fast.detect(image, None)
        keypoints = np.array([kp_.pt for kp_ in kp], dtype=np.float32)
        keypoints_list.append(keypoints)

    return keypoints_list

def normalize_coordinates(coords, image_shape):
    # Normalize coordinates to [0, 1]
    coords_norm = coords.copy()
    coords_norm[:, 0] /= image_shape[1]  # x coordinate
    coords_norm[:, 1] /= image_shape[0]  # y coordinate
    return coords_norm

def prepare_training_data(images_list, keypoints_list):
    # Assuming all images are the same size
    image_shape = images_list[0].shape
    h, w = image_shape  # Get height and width

    # Stack all images and keypoints
    all_x_data = []
    all_u_data = []
    all_keypoints = []

    for idx, image in enumerate(images_list):
        image_normalized = image.astype(np.float32) / 255.0

        # Get pixel coordinates
        h, w = image.shape
        xv, yv = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        coords = np.column_stack([xv.flatten(), yv.flatten()])
        intensities = image_normalized.flatten().reshape(-1, 1)

        all_x_data.append(coords)
        all_u_data.append(intensities)

        # Normalize keypoints
        keypoints_norm = normalize_coordinates(keypoints_list[idx], image_shape)
        all_keypoints.append(keypoints_norm)

    # Concatenate all data
    x_data = np.vstack(all_x_data)
    u_data = np.vstack(all_u_data)
    keypoints = np.vstack(all_keypoints)

    # Convert to tensors
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)
    keypoints = torch.tensor(keypoints, dtype=torch.float32)

    return x_data, u_data, keypoints, h, w

def generate_collocation_points(N_f, lb, ub):
    # N_f: Number of collocation points
    # lb: Lower bounds [x_min, y_min]
    # ub: Upper bounds [x_max, y_max]
    X_f = lhs(2, N_f)
    X_f = lb + (ub - lb) * torch.tensor(X_f, dtype=torch.float32)
    return X_f

if __name__ == '__main__':
    # Example usage
    selected_images_folder = 'data'
    images_list, image_names_list = load_images_from_directory(selected_images_folder, max_images=1)
    keypoints_list = detect_keypoints(images_list)
    x_data, u_data, keypoints = prepare_training_data(images_list, keypoints_list)
    x_colloc = generate_collocation_points(10000, lb=torch.tensor([0.0, 0.0]), ub=torch.tensor([1.0, 1.0]))
