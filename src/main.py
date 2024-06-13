import os
pwd = os.path.abspath(os.path.dirname(__file__))
from Functions import load, patchExtraction_training
from setup import *


# Load Data
images, labels = load.load_data(ABNORMAL_DIR, NORMAL_DIR)

# Split data for training and testing
load.split_and_copy_images(ABNORMAL_DIR, os.path.join(TRAIN_DIR, 'normal'), os.path.join(TEST_DIR, 'normal'), 410)
load.split_and_copy_images(NORMAL_DIR, os.path.join(TRAIN_DIR, 'abnormal'), os.path.join(TEST_DIR, 'abnormal'), 410)

for data_dir in [TRAIN_DIR, TEST_DIR]:
    for class_type in ['normal', 'abnormal']:
        input_dir = os.path.join(data_dir, class_type)
        print(input_dir)
        output_dir = os.path.join('Processed_Data', os.path.basename(data_dir), class_type)
        visualized_dir = os.path.join('Visualized_Data', os.path.basename(data_dir), class_type)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualized_dir, exist_ok=True)
        patchExtraction_training.process_directory(input_dir, output_dir, visualized_dir, is_abnormal=(class_type == 'abnormal'))

print("Patch extraction and visualization complete.")