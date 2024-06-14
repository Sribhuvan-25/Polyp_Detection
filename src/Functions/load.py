import random
import cv2
import os
import numpy as np
import shutil
import sys

def load_data(polyp_folder, non_polyp_folder, image_size=(224, 224)):
    images = []
    labels = []  # 1 for polyp, 0 for non-polyp
    
    # Load polyp images
    for filename in os.listdir(polyp_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(polyp_folder, filename))
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(1)
    
    # Load non-polyp images
    for filename in os.listdir(non_polyp_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(non_polyp_folder, filename))
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(0)
    
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    return np.array(images), np.array(labels)

def split_and_copy_images(source_dir, train_target_dir, test_target_dir, train_size):
    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    train_images = images[:train_size]
    test_images = images[train_size:]

    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(train_target_dir, image))
        if 'abnormal' in source_dir:
            roi_file = os.path.splitext(image)[0] + '.roi'
            if os.path.exists(os.path.join(source_dir, roi_file)):
                shutil.copy(os.path.join(source_dir, roi_file), os.path.join(train_target_dir, roi_file))
        
    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(test_target_dir, image))
        if 'abnormal' in source_dir:
            roi_file = os.path.splitext(image)[0] + '.roi'
            if os.path.exists(os.path.join(source_dir, roi_file)):
                shutil.copy(os.path.join(source_dir, roi_file), os.path.join(test_target_dir, roi_file))

    print('Data splitting complete.')
    

def load_train_test_images(polyp_folder, non_polyp_folder, image_size=(224, 224)):
    images = []
    labels = []  # 1 for polyp, 0 for non-polyp
    
    # Load polyp images
    for filename in os.listdir(polyp_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(polyp_folder, filename))
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(1)
    
    # Load non-polyp images
    for filename in os.listdir(non_polyp_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(non_polyp_folder, filename))
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(0)
    return images, labels

def load_train_test_patches(polyp_folder, non_polyp_folder, polyp_folder_testing, non_polyp_folder_testing, image_size=(224, 224)):
    train_images = []
    train_labels = []  # 1 for polyp, 0 for non-polyp

    test_images = []
    test_labels = []
    
    
    # Training
    # Load polyp images
    for filename in os.listdir(polyp_folder):
        image = cv2.imread(os.path.join(polyp_folder, filename))
        image = cv2.resize(image, image_size)
        train_images.append(image)
        train_labels.append(1)
    
    # Load non-polyp images
    for filename in os.listdir(non_polyp_folder):
        image = cv2.imread(os.path.join(non_polyp_folder, filename))
        image = cv2.resize(image, image_size)
        train_images.append(image)
        train_labels.append(0)
    
    # Testing
    for filename in os.listdir(polyp_folder_testing):
        image = cv2.imread(os.path.join(polyp_folder_testing, filename))
        image = cv2.resize(image, image_size)
        test_images.append(image)
        test_labels.append(0)
        
    for filename in os.listdir(non_polyp_folder_testing):
        image = cv2.imread(os.path.join(non_polyp_folder_testing, filename))
        image = cv2.resize(image, image_size)
        test_images.append(image)
        test_labels.append(0)
    
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)