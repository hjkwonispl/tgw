# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import warnings
import argparse

# Add path
sys.path.append('../')

import numpy as np

from PIL import Image
from shutil import copyfile
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import progress_bar

# Ignore warnings
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description='TGW for Age Estimation Training',
  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--aligned_data_path', default='../data/aligned', type=str, 
  help='database path')
parser.add_argument('--output_path', default='../data/output', type=str, 
  help='output path')
args = parser.parse_args()

# Use settings in [4] from following files
# https://github.com/GilLevi/AgeGenderDeepLearning/tree/master/Folds/train_val_txt_files_per_fold 
ADIENCE_5_FOLD_TXT_PATH = './adience_5_fold'

# Fold
FOLDS = ['0','1','2','3','4']

# Data directory 종류
DATA_TYPES = ['age_test', 'age_train', 'age_val']

def main():
  """
  This script do following actions

    1. Copy aligned images of Adience database into following format of 
        directories according to experimental scheme of [4]
       Each directory will have one of following name:
        `age_train_fold#', `age_val_fold#'. `age_test_fold#'
        where # is the number in [0, 1, 2, 3, 4] which represent folds in
        5-fold cross-validation setting 

    2. Add fold and age class information at the head of filename.
       We add `F#C@_' for fold and class information, where `#' is fold
       and `@' is age class of given image.

       e.g.) If `landmark_aligned_face.200.jpg' belong to fold = `0' and
             age class of face in the image is `1'

             Raw file name: landmark_aligned_face.200
             Output file name: F0C1_landmark_aligned_face.200
  """

  for data_type, fold in product(DATA_TYPES, FOLDS):

    # Load experimental settings in ref. [4]
    label_txt  = os.path.join(ADIENCE_5_FOLD_TXT_PATH, 
      ('test_fold_is_' + fold ), (data_type + '.txt')) 
    print('Process {} file'.format(label_txt))

    with open(label_txt) as label_txt_file:
        
      label_data_list = label_txt_file.readlines()

      for label_data in label_data_list:
      
        label_data = label_data.split()
    
        file_path = label_data[0]
        file_name = os.path.basename(label_data[0])

        # Get label
        label = label_data[1]

        # Create output directory
        dst_path = os.path.join(args.output_path, (data_type + '_fold' + fold))
        if not os.path.exists(dst_path):
          os.makedirs(dst_path)

        # Add `F#C@_' to each filename
        src_file_path = os.path.join(args.aligned_data_path, file_path)
        dst_file_path = \
          os.path.join(dst_path, ('F' + fold + 'C' + label + '_' + file_name))

        # Copy image into correspoding directory
        copyfile(src_file_path, dst_file_path)

if __name__ == '__main__':
  main()