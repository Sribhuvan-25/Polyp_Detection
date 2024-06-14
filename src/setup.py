import os
pwd = os.path.abspath(os.path.dirname(__file__))


DATA_DIR = 'src/Data'
ABNORMAL_DIR = os.path.join(DATA_DIR, 'abnormal')
NORMAL_DIR = os.path.join(DATA_DIR, 'normal')

OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'Split_Data')
TRAIN_DIR = os.path.join(OUTPUT_DATA_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DATA_DIR, 'test')

for directory in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'abnormal'), exist_ok=True)
    
PROCESSED_PATCHES_DIR = os.path.join(DATA_DIR, 'Processed_Patches')
VISUALIZATIONS_PATCHES_DIR = os.path.join(DATA_DIR, 'Visualizations_Extracted_Patches')
