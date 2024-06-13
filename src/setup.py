import os
pwd = os.path.abspath(os.path.dirname(__file__))


DATA_DIR = './Data'
ABNORMAL_DIR = os.path.join(DATA_DIR, 'abnormal')
NORMAL_DIR = os.path.join(DATA_DIR, 'normal')

OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'Spit_Data')
TRAIN_DIR = os.path.join(OUTPUT_DATA_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DATA_DIR, 'test')

for directory in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'abnormal'), exist_ok=True)