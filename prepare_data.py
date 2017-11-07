import utils as utils
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize


#### DIV2K data

# ## Training data
# PATH_TO_HR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_train_HR/'
# PATH_TO_LR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_train_LR_bicubic/X4/'
# SAVE_DIR = 'D:/PROJECT_DATA/NTIRE_2017_gray/tfrecords/'
# NAME = 'training_data'

# data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='train', output_data_dir=PATH_TO_HR)
# input_list, output_list = data_converter.get_files()
# data_converter.convert_to_tfrecord(input_list, output_list, num_files=8)

# ## Validation data
# PATH_TO_HR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_valid_HR/'
# PATH_TO_LR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_valid_LR_bicubic/X4/'
# SAVE_DIR = 'D:/PROJECT_DATA/NTIRE_2017_gray/tfrecords/'
# NAME = 'validation_data'

# data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='valid', output_data_dir=PATH_TO_HR)
# input_list, output_list = data_converter.get_files()
# data_converter.convert_to_tfrecord(input_list, output_list, num_files=1)

# ## Testing data
# PATH_TO_LR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_test_LR_bicubic/X4/'
# SAVE_DIR = 'D:/PROJECT_DATA/NTIRE_2017_gray/tfrecords/'
# NAME = 'testing_data'

# data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='test', output_data_dir=PATH_TO_LR)
# input_list, output_list = data_converter.get_files()
# data_converter.convert_to_tfrecord(input_list, output_list, num_files=1)

#### MRI data

## Training data
PATH_TO_HR = 'D:/PROJECT_DATA/axial_superresolution/raw_data/train_hr/'
PATH_TO_LR = 'D:/PROJECT_DATA/axial_superresolution/raw_data/train_lr_bicubic/'
SAVE_DIR = 'D:/PROJECT_DATA/axial_superresolution/tfrecords/'
NAME = 'training_data_bicubic'

data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='train', output_data_dir=PATH_TO_HR)
input_list, output_list = data_converter.get_files()
data_converter.convert_to_tfrecord(input_list, output_list, num_files=4)

## Validation data
PATH_TO_HR = 'D:/PROJECT_DATA/axial_superresolution/raw_data/valid_hr/'
PATH_TO_LR = 'D:/PROJECT_DATA/axial_superresolution/raw_data/valid_lr_bicubic/'
SAVE_DIR = 'D:/PROJECT_DATA/axial_superresolution/tfrecords/'
NAME = 'validation_data_bicubic'

data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='valid', output_data_dir=PATH_TO_HR)
input_list, output_list = data_converter.get_files()
data_converter.convert_to_tfrecord(input_list, output_list, num_files=1)

# ## Testing data
# PATH_TO_LR = 'D:/PROJECT_DATA/NTIRE_2017_gray/DIV2K_test_LR_bicubic/X4/'
# SAVE_DIR = 'D:/PROJECT_DATA/NTIRE_2017_gray/tfrecords/'
# NAME = 'testing_data'

# data_converter = utils.Data(PATH_TO_LR, SAVE_DIR, NAME, mode='test', output_data_dir=PATH_TO_LR)
# input_list, output_list = data_converter.get_files()
# data_converter.convert_to_tfrecord(input_list, output_list, num_files=1)
