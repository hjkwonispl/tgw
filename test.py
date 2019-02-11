# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import argparse
import time
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models import ProposedNetwork, init_weights
from adience_dataset import AdienceDataset
from utils import progress_bar

# Argument parsing
# python train_adience.py -t ksize -f 3 --resume
parser = argparse.ArgumentParser(description='TGW for Age Estimation Training')
parser.add_argument('--gpu', type=str, default='2',
  help='GPU Number')
parser.add_argument('--log_path', type=str, default='./log',
  help='log file path')
parser.add_argument('--test_batch_size', type=int, default=1,
  help='validation batch size')
parser.add_argument('--data_path', type=str, default='./data/Adience',
  help='database path')
parser.add_argument('--num_workers', default=4, type=int, 
  help='number of queue workers for dataloader')
args = parser.parse_args()

# Set Training GPU Number
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
   
# Check if cuda is available
USE_CUDA = torch.cuda.is_available()

# List of test folds
FOLDS = ['0', '1', '2', '3', '4']

def main():

  if not os.path.exists(os.path.join(args.log_path)):
    os.makedirs(args.log_path)
    
  log_txt  = open(os.path.join(args.log_path, 'test_log.txt'), 'a')
  log_txt.write(dt.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
  
  for fold in FOLDS:

    print('# Test on fold: {}'.format(fold))
    log_txt.write("\n{}\t{}\t{}\n".format('type', 'fold', 'accuracy'))

    # Test data Preparation
    print('# Preparing test data')
    
    test_data_directory = \
      [os.path.join(args.data_path, 'age_{}_fold{}'.format('test', fold))]

    transform_test = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(227),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), 
        (0.2023, 0.1994, 0.2010)),
    ])

    test_set = AdienceDataset(image_dirs=test_data_directory, 
      transforms = transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, 
      batch_size= args.test_batch_size, 
      shuffle=False, 
      num_workers=args.num_workers)

    print('#- Test dataset = ', test_data_directory)

    # Load trained proposed network
    print('# Load trained proposed network')

    assert os.path.isdir('saved_models'), \
      '#- Error: no saved_models directory found!'
    net = ProposedNetwork('ksize')
    checkpoint = torch.load(
      './saved_models/proposed_network_fold{}'.format(fold))
    net = torch.nn.DataParallel(net).cuda()    
    net.load_state_dict(checkpoint['net'])

    if USE_CUDA:
      net.cuda()
      cudnn.benchmark = True

    # Test for one fold
    accuracy = test(test_loader, net)
    log_txt.write("{}\t{}\n".format(fold, accuracy))
                 
def test(test_loader, net):
  """
  Test of proposed network. Accuracy is given by %

  Args: 
    test_loader: instances of DataLoader with Adience DB
    net: instance of models.ProposedNetwork
  """
  net.eval()
  correct = 0
  accuracy = 0
  total = 0

  for batch_idx, (inputs, targets) in enumerate(test_loader):
  
    if USE_CUDA:
      inputs, targets = inputs.cuda(), targets.cuda()
    
    inputs = Variable(inputs, volatile=True)
    targets = Variable(targets.squeeze())
    outputs = net(inputs)
    
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    
    correct += predicted.eq(targets.data).cpu().sum()     
    accuracy = 100.*correct/total

    progress_bar(batch_idx, len(test_loader), 'Accuracy: %.3f%% (%d/%d)'
      % (accuracy, correct, total))
  
  return accuracy
    
if __name__ == "__main__": 
  main()
