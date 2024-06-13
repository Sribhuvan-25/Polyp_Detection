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
    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(images)
    
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(train_target_dir, image))
        
    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(test_target_dir, image))
        
    print('Data splitting complete.')

