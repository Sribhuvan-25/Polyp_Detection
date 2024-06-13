
# import pandas as pd
# import numpy as np
# import cv2
# from read_roi import read_roi_file
# import os
# import cv2
# import numpy as np
# from read_roi import read_roi_file
# import random

# def detect_features_in_channel(channel):
#     # Use goodFeaturesToTrack on the given channel
#     features = cv2.goodFeaturesToTrack(channel, maxCorners=200, qualityLevel=0.01, minDistance=50, blockSize=300)
#     if features is not None:
#         return np.int0(features)
#     else:
#         return np.array([])

# def filter_close_points(features, min_dist=100):
#     # Ensure features array is in the shape (n_points, 2)
#     if features.ndim == 3:  # In case features are in shape (n, 1, 2)
#         features = features.reshape(-1, 2)
    
#     # Initialize an empty list to hold filtered points
#     filtered_points = []

#     for feature in features:
#         if not filtered_points:  # If filtered_points is empty, add the first feature
#             filtered_points.append(feature)
#             continue

#         # Calculate distances from the current feature to all filtered points
#         dists = np.sqrt(np.sum((np.array(filtered_points) - feature) ** 2, axis=1))

#         # Check if all distances are greater than min_dist
#         if np.all(dists >= min_dist):
#             filtered_points.append(feature)

#     return np.array(filtered_points)

# def bounding_box(roi_file_path):
#     roi = read_roi_file(roi_file_path)
#     for box_info in roi.values():
#         if box_info['type'] == 'rectangle':
#             left = box_info['left']
#             top = box_info['top']
#             width = box_info['width']
#             height = box_info['height']
#     return left, top, width, height

# def extract_patch(image, center_x, center_y, patch_size):
#     half_width, half_height = patch_size[0] // 2, patch_size[1] // 2
#     patch = image[
#         center_y - half_height:center_y + half_height,
#         center_x - half_width:center_x + half_width
#     ]
#     return patch

# def extract_patches_from_abnormal(image_path, roi_file_path, patch_size=(100, 100)):
#     # Read the image
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Read ROI and get bounding box
#     left, top, width, height = bounding_box(roi_file_path)
    
#     # Extract the ROI region
#     roi_region = gray_image[top:top+height, left:left+width]
    
#     # Detect features in the ROI region
#     features = detect_features_in_channel(roi_region)
    
#     if len(features) > 0:
#         # Convert features to image coordinates
#         features[:, 0] += left
#         features[:, 1] += top

#         # Filter close points to get salient points
#         salient_points = filter_close_points(features)
        
#         if len(salient_points) > 0:
#             # Get the closest salient point to the center of the ROI
#             center_x, center_y = np.mean(salient_points, axis=0).astype(int)
#             patch = extract_patch(image, center_x, center_y, patch_size)
            
#             # Visualize patch on image
#             visualize_patch(image, center_x, center_y, patch_size)
            
#             return patch, image
    
#     # Fallback if no salient points found
#     center_x, center_y = left + width // 2, top + height // 2
#     patch = extract_patch(image, center_x, center_y, patch_size)
    
#     # Visualize patch on image
#     visualize_patch(image, center_x, center_y, patch_size)
    
#     return patch, image

# def extract_patches_from_normal(image_path, patch_size=(100, 100)):
#     # Read the image
#     image = cv2.imread(image_path)
#     h, w, _ = image.shape
    
#     # Define the middle region to avoid borders
#     middle_x_range = (w // 4, 3 * w // 4)
#     middle_y_range = (h // 4, 3 * h // 4)
    
#     patches = []
#     visualized_images = []
#     for _ in range(2):  # Extract two patches
#         center_x = random.randint(middle_x_range[0], middle_x_range[1])
#         center_y = random.randint(middle_y_range[0], middle_y_range[1])
#         patch = extract_patch(image, center_x, center_y, patch_size)
#         patches.append(patch)
        
#         # Visualize patch on image
#         visualize_patch(image, center_x, center_y, patch_size)
#         visualized_images.append(image.copy())
    
#     return patches, visualized_images

# def visualize_patch(image, center_x, center_y, patch_size):
#     half_width, half_height = patch_size[0] // 2, patch_size[1] // 2
#     top_left = (center_x - half_width, center_y - half_height)
#     bottom_right = (center_x + half_width, center_y + half_height)
#     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# def save_patches(patches, output_dir, base_name):
#     for i, patch in enumerate(patches):
#         patch_filename = f"{base_name}_patch_{i}.png"
#         patch_path = os.path.join(output_dir, patch_filename)
#         cv2.imwrite(patch_path, patch)

# def save_visualized_images(images, output_dir, base_name):
#     for i, img in enumerate(images):
#         visualized_filename = f"{base_name}_visualized_{i}.png"
#         visualized_path = os.path.join(output_dir, visualized_filename)
#         cv2.imwrite(visualized_path, img)

# def process_directory(input_dir, output_dir, visualized_dir, is_abnormal=False, patch_size=(100, 100)):
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.png'):
#             image_path = os.path.join(input_dir, filename)
#             base_name = os.path.splitext(filename)[0]

#             if is_abnormal:
#                 roi_file_path = os.path.join(input_dir, f"{base_name}.roi")
#                 patch, visualized_image = extract_patches_from_abnormal(image_path, roi_file_path, patch_size)
#                 save_patches([patch], output_dir, base_name)
#                 save_visualized_images([visualized_image], visualized_dir, base_name)
#             else:
#                 patches, visualized_images = extract_patches_from_normal(image_path, patch_size)
#                 save_patches(patches, output_dir, base_name)
#                 save_visualized_images(visualized_images, visualized_dir, base_name)

import pandas as pd
import numpy as np
import cv2
from read_roi import read_roi_file
import os
import random

def detect_features_in_channel(channel):
    features = cv2.goodFeaturesToTrack(channel, maxCorners=200, qualityLevel=0.01, minDistance=50, blockSize=300)
    if features is not None:
        return np.int0(features)
    else:
        return np.array([])

def filter_close_points(features, min_dist=100):
    if features.ndim == 3:  # In case features are in shape (n, 1, 2)
        features = features.reshape(-1, 2)
    
    filtered_points = []

    for feature in features:
        if not filtered_points:  # If filtered_points is empty, add the first feature
            filtered_points.append(feature)
            continue

        dists = np.sqrt(np.sum((np.array(filtered_points) - feature) ** 2, axis=1))

        if np.all(dists >= min_dist):
            filtered_points.append(feature)

    return np.array(filtered_points)

def bounding_box(roi_file_path):
    roi = read_roi_file(roi_file_path)
    for box_info in roi.values():
        if box_info['type'] == 'rectangle':
            left = box_info['left']
            top = box_info['top']
            width = box_info['width']
            height = box_info['height']
    return left, top, width, height

def extract_patch(image, center_x, center_y, patch_size):
    half_width, half_height = patch_size[0] // 2, patch_size[1] // 2
    patch = image[
        center_y - half_height:center_y + half_height,
        center_x - half_width:center_x + half_width
    ]
    return patch

def extract_patches_from_abnormal(image_path, roi_file_path, patch_size=(100, 100)):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    left, top, width, height = bounding_box(roi_file_path)
    roi_region = gray_image[top:top+height, left:left+width]
    
    features = detect_features_in_channel(roi_region)
    
    if len(features) > 0:
        features[:, 0] += left
        features[:, 1] += top

        salient_points = filter_close_points(features)
        
        if len(salient_points) > 0:
            center_x, center_y = np.mean(salient_points, axis=0).astype(int)
            patch = extract_patch(image, center_x, center_y, patch_size)
            visualize_patch(image, center_x, center_y, patch_size)
            return patch, image
    
    center_x, center_y = left + width // 2, top + height // 2
    patch = extract_patch(image, center_x, center_y, patch_size)
    visualize_patch(image, center_x, center_y, patch_size)
    
    return patch, image

def extract_patches_from_normal(image_path, patch_size=(100, 100)):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    middle_x_range = (w // 4, 3 * w // 4)
    middle_y_range = (h // 4, 3 * h // 4)
    
    patches = []
    visualized_images = []
    for _ in range(2):
        center_x = random.randint(middle_x_range[0], middle_x_range[1])
        center_y = random.randint(middle_y_range[0], middle_y_range[1])
        patch = extract_patch(image, center_x, center_y, patch_size)
        patches.append(patch)
        
        visualize_patch(image, center_x, center_y, patch_size)
        visualized_images.append(image.copy())
    
    return patches, visualized_images

def visualize_patch(image, center_x, center_y, patch_size):
    half_width, half_height = patch_size[0] // 2, patch_size[1] // 2
    top_left = (center_x - half_width, center_y - half_height)
    bottom_right = (center_x + half_width, center_y + half_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

def save_patches(patches, output_dir, base_name):
    for i, patch in enumerate(patches):
        patch_filename = f"{base_name}_patch_{i}.png"
        patch_path = os.path.join(output_dir, patch_filename)
        cv2.imwrite(patch_path, patch)

def save_visualized_images(images, output_dir, base_name):
    for i, img in enumerate(images):
        visualized_filename = f"{base_name}_visualized_{i}.png"
        visualized_path = os.path.join(output_dir, visualized_filename)
        cv2.imwrite(visualized_path, img)

def process_directory(input_dir, output_dir, visualized_dir, is_abnormal=False, patch_size=(100, 100)):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            if is_abnormal:
                roi_file_path = os.path.join(input_dir, f"{base_name}.roi")
                patch, visualized_image = extract_patches_from_abnormal(image_path, roi_file_path, patch_size)
                save_patches([patch], output_dir, base_name)
                save_visualized_images([visualized_image], visualized_dir, base_name)
            else:
                patches, visualized_images = extract_patches_from_normal(image_path, patch_size)
                save_patches(patches, output_dir, base_name)
                save_visualized_images(visualized_images, visualized_dir, base_name)

# Define the base directory
# base_dir = '/Users/sb/TReNDS_New/Data/Split_Data'

# # Process the training and testing directories
# for data_type in ['train', 'test']:
#     for class_type in ['normal', 'abnormal']:
#         input_dir = os.path.join(base_dir, data_type, class_type)
#         output_dir = os.path.join('Processed_Data', data_type, class_type)
#         visualized_dir = os.path.join('Visualized_Data', data_type, class_type)
#         os.makedirs(output_dir, exist_ok=True)
#         os.makedirs(visualized_dir, exist_ok=True)
#         process_directory(input_dir, output_dir, visualized_dir, is_abnormal=(class_type == 'abnormal'))

# print("Patch extraction and visualization complete.")
