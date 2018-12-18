# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import numpy as np
import warnings
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
warnings.filterwarnings("ignore")

class AdienceDataset(Dataset):
  
  def __init__(self, image_dirs, transforms=None):
    """
    Dataset for Adience Database

    Args:
      image_dir (List of string): Directory with all the images.
      transforms: transforms could be on the input images
    """
    self.image_dirs = image_dirs
    self.image_files = []
    self.targets = []
    self.transforms = None
    
    for image_dir in self.image_dirs:

      image_files = next(os.walk(image_dir))[2]
      
      for image_file in image_files:
        img_file = os.path.join(image_dir, image_file)
        self.image_files.append(img_file)
        self.targets.append(image_file[3])
        
    if (transforms != None):
      self.transforms = transforms

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = self.image_files[idx]
    
    targets = torch.LongTensor([int(self.targets[idx])])
    inputs = Image.open(img_name)

    if (self.transforms != None):
      inputs = self.transforms(inputs)
    
    sample = (inputs, targets)

    return sample