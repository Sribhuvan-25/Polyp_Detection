import os
import random
import cv2
import numpy as np
from read_roi import read_roi_file
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def detect_features_in_channel(channel):
    features = cv2.goodFeaturesToTrack(channel, maxCorners=200, qualityLevel=0.01, minDistance=50, blockSize=300)
    if features is not None:
        return np.int0(features)
    else:
        return np.array([])

def filter_close_points(features, min_dist=100):
    if features.ndim == 3:
        features = features.reshape(-1, 2)
    
    filtered_points = []

    for feature in features:
        if not filtered_points:
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

def visualize_and_save(image, output_path, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def extract_and_save_patches_abnormal(folder, output_folder, visualization_folder, ideal_patch_size=(275, 300)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
        
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            visualization_image = image.copy()
            roi_path = os.path.splitext(image_path)[0] + ".roi"

            if os.path.exists(roi_path):
                blue, green, red = cv2.split(image)
                features_blue = detect_features_in_channel(blue)
                features_green = detect_features_in_channel(green)
                features_red = detect_features_in_channel(red)
                features_combined = np.vstack([f.reshape(-1, 2) for f in [features_blue, features_green, features_red] if f.size > 0])
                features_combined = np.unique(features_combined, axis=0)
                features_combined = filter_close_points(features_combined)

                if features_combined.size == 0:
                    continue  # Skip if no features found

                for point in features_combined:
                    cv2.circle(visualization_image, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)

                gt_left, gt_top, gt_width, gt_height = bounding_box(roi_path)
                cv2.rectangle(visualization_image, (gt_left, gt_top), (gt_left + gt_width, gt_top + gt_height), (0, 255, 0), 3)

                gt_center = np.array([[gt_left + gt_width / 2, gt_top + gt_height / 2]])
                distances = cdist(features_combined, gt_center, metric='euclidean')
                closest_point_index = np.argmin(distances)
                closest_point = features_combined[closest_point_index]

                sp_start_x = max(int(closest_point[0] - ideal_patch_size[1] / 2), 0)
                sp_start_y = max(int(closest_point[1] - ideal_patch_size[0] / 2), 0)
                sp_end_x = min(sp_start_x + ideal_patch_size[1], image.shape[1])
                sp_end_y = min(sp_start_y + ideal_patch_size[0], image.shape[0])

                cv2.rectangle(visualization_image, (sp_start_x, sp_start_y), (sp_end_x, sp_end_y), (0, 0, 0), 3)

                salient_patch = image[sp_start_y:sp_end_y, sp_start_x:sp_end_x]
                salient_patch_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_salient_patch_{sp_start_x}_{sp_start_y}.png"
                salient_patch_path = os.path.join(output_folder, salient_patch_name)

                cv2.imwrite(salient_patch_path, salient_patch)

                # Save visualization
                visualization_path = os.path.join(visualization_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_visualization.png")
                visualize_and_save(visualization_image, visualization_path, "Salient Patch and ROI")

def extract_and_save_patches_normal(folder, output_folder, visualization_folder, ideal_patch_size=(275, 300)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            visualization_image = image.copy()
            h, w, _ = image.shape

            middle_x_range = (w // 4, 3 * w // 4)
            middle_y_range = (h // 4, 3 * h // 4)

            patches = []
            for _ in range(2):
                center_x = random.randint(middle_x_range[0], middle_x_range[1])
                center_y = random.randint(middle_y_range[0], middle_y_range[1])
                sp_start_x = max(int(center_x - ideal_patch_size[1] / 2), 0)
                sp_start_y = max(int(center_y - ideal_patch_size[0] / 2), 0)
                sp_end_x = min(sp_start_x + ideal_patch_size[1], w)
                sp_end_y = min(sp_start_y + ideal_patch_size[0], h)

                patch = image[sp_start_y:sp_end_y, sp_start_x:sp_end_x]
                patches.append(patch)
                patch_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_normal_patch_{sp_start_x}_{sp_start_y}.png"
                patch_path = os.path.join(output_folder, patch_name)
                cv2.imwrite(patch_path, patch)

                cv2.rectangle(visualization_image, (sp_start_x, sp_start_y), (sp_end_x, sp_end_y), (0, 0, 0), 3)

            # Save visualization
            visualization_path = os.path.join(visualization_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_visualization.png")
            visualize_and_save(visualization_image, visualization_path, "Extracted Patches")

def process_dirs(train_dir, test_dir, output_dir, visualization_dir, ideal_patch_size=(275, 300)):
    for data_dir in [train_dir, test_dir]:
        for class_type in ['normal', 'abnormal']:
            input_dir = os.path.join(data_dir, class_type)
            output_sub_dir = os.path.join(output_dir, os.path.basename(data_dir), class_type)
            visualization_sub_dir = os.path.join(visualization_dir, os.path.basename(data_dir), class_type)
            os.makedirs(output_sub_dir, exist_ok=True)
            os.makedirs(visualization_sub_dir, exist_ok=True)

            if class_type == 'abnormal':
                extract_and_save_patches_abnormal(input_dir, output_sub_dir, visualization_sub_dir, ideal_patch_size)
            else:
                extract_and_save_patches_normal(input_dir, output_sub_dir, visualization_sub_dir, ideal_patch_size)
