import os
pwd = os.path.abspath(os.path.dirname(__file__))
from Functions import load, patchExtraction_training
from setup import *


# Load Data
images, labels = load.load_data(ABNORMAL_DIR, NORMAL_DIR)

# Split data for training and testing
load.split_and_copy_images(NORMAL_DIR, os.path.join(TRAIN_DIR, 'normal'), os.path.join(TEST_DIR, 'normal'), 410)
load.split_and_copy_images(ABNORMAL_DIR, os.path.join(TRAIN_DIR, 'abnormal'), os.path.join(TEST_DIR, 'abnormal'), 410)

# Process the normal and abnormal directories
patchExtraction_training.process_dirs(TRAIN_DIR, TEST_DIR, PROCESSED_PATCHES_DIR, VISUALIZATIONS_PATCHES_DIR)