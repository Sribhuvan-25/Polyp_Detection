import os


DATA_DIR = 'Data'
ABNORMAL_DATA = os.path.join(DATA_DIR, 'abnormal')
NORMAL_DATA = os.path.join(DATA_DIR, 'normal')


OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'Spit_Data')
TRAIN_DIR = os.path.join(OUTPUT_DATA_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DATA_DIR, 'test')

for directory in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'abnormal'), exist_ok=True)