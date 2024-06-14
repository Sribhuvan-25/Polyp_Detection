import os

import torch

pwd = os.path.abspath(os.path.dirname(__file__))
from Functions import load, patchExtraction_training, model_training
from setup import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Load Data
# images, labels = load.load_data(ABNORMAL_DIR, NORMAL_DIR)

# Split data for training and testing
load.split_and_copy_images(NORMAL_DIR, os.path.join(TRAIN_DIR, 'normal'), os.path.join(TEST_DIR, 'normal'), 410)
load.split_and_copy_images(ABNORMAL_DIR, os.path.join(TRAIN_DIR, 'abnormal'), os.path.join(TEST_DIR, 'abnormal'), 410)

# Process the normal and abnormal directories
patchExtraction_training.process_dirs(TRAIN_DIR, TEST_DIR, PROCESSED_PATCHES_DIR, VISUALIZATIONS_PATCHES_DIR)

# Loadding Images for training and testing
train_images, train_labels = load.load_train_test_images(TRAIN_ABNORMAL_IMAGES_DIR, TRAIN_NORMAL_IMAGES_DIR)
test_images, test_labels = load.load_train_test_images(TEST_ABNORMAL_IMAGES_DIR, TEST_NORMAL_IMAGES_DIR)

train_images, train_labels = load.load_train_test_images(TRAIN_ABNORMAL_IMAGES_DIR, TRAIN_NORMAL_IMAGES_DIR)
test_images, test_labels = load.load_train_test_images(TEST_ABNORMAL_IMAGES_DIR, TEST_NORMAL_IMAGES_DIR)

# Loading Patches for training and testing
train_patches, train_patch_labels, test_patches, test_patch_labels = load.load_train_test_patches(TRAIN_ABNORMAL_PATCHES_DIR, TRAIN_NORMAL_PATCHES_DIR, 
                                                                                                  TEST_ABNORMAL_PATCHES_DIR, TEST_NORMAL_PATCHES_DIR)



# Images
model_training.train_denseNetModel(train_images, train_labels, NUM_CLASSES, MODEL_IMAGES_PATH)
model_images = torch.load(MODEL_IMAGES_PATH)
accuracy, precision, recall = model_training.evaluate_model(model_images, test_images, test_labels, BATCH_SIZE, device)

# Images
model_training.train_denseNetModel(train_patches, train_patches, NUM_CLASSES, MODEL_PATCHES_PATH)
model_patches = torch.load(MODEL_PATCHES_PATH)
accuracy, precision, recall = model_training.evaluate_model(model_patches, test_images, test_labels, BATCH_SIZE, device)