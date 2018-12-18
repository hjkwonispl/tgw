# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import argparse
import shutil

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
parser = argparse.ArgumentParser(description='TGW for Age Estimation Training',
  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--resume', '-r', action='store_true', 
  help='resume from checkpoint')
parser.add_argument('--lr', default=1e-4, type=float, 
  help='initial learning rate')
parser.add_argument('--epochs', default=200, type=int, 
  help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, 
  help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch_size', default=8, type=int, 
  help='train batch size')
parser.add_argument('--val_batch_size', default=2, type=int, 
  help='validation batch size')
parser.add_argument('--num_workers', default=4, type=int, 
  help='number of queue workers for dataloader')
parser.add_argument('--data_path', default='./data/Adience', type=str, 
  help='database path')
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str, 
  help='checkpoint file path')
parser.add_argument('--fold', '-f', help='0/1/2/3/4')     
parser.add_argument('--gpu', default='2', type=str, help='GPU Number')
parser.add_argument('--type', '-t', 
  help='ksize: learns sampling grid (Best Method, abbreviation=Kernel SIZE) \
    \n all: learns sampling grids and parameters of Gabor wavelets \
    \n ~ksize: learns other parameters of Gabor wavelets except orientation \
    \n none: do not learn any parameter of Gabor wavelets w/ steering.')
args = parser.parse_args()

# Set Training GPU Number
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
   
# Check if cuda is available
USE_CUDA = torch.cuda.is_available()

def train(train_loader, net, criterion, optimizer):
  """
  Training for one epoch. Accuracy is given by %

  Args: 
    train_loader: instances of DataLoader with Adience DB
    net: instance of models.ProposedNetwork
    criterion: loss for network (usually cross-entropy)
    optimizer: weight optimizer for network
  """

  net.train()
  train_loss = 0
  correct = 0
  total = 0

  for batch_idx, (inputs, targets) in enumerate(train_loader):

    # for batch_normalization
    if (inputs.size()[0] < 2):
      continue

    if USE_CUDA:
      inputs, targets = inputs.cuda(), targets.cuda()
    
    optimizer.zero_grad()
    inputs, targets = Variable(inputs).cuda(), Variable(targets.squeeze())
    outputs = net(inputs)

    if (torch.sum((outputs.data != outputs.data))):
      print('NaN error')
      return
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.data[0]
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    # Print progressbar for monitoring via stdout
    progress_bar(batch_idx, len(train_loader), 
      'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def validate(val_loader, net, criterion):
  """
  Validation for one epoch. Accuracy is given by %

  Args: 
    val_loader: instances of DataLoader with Adience DB
    net: instance of models.ProposedNetwork
    criterion: loss for network (usually cross-entropy)
  """

  net.eval()
  val_loss = 0
  correct = 0
  total = 0

  for batch_idx, (inputs, targets) in enumerate(val_loader):

    if USE_CUDA:
      inputs, targets = inputs.cuda(), targets.cuda()

    inputs = Variable(inputs, volatile=True)
    targets = Variable(targets.squeeze())
    outputs = net(inputs)

    loss = criterion(outputs, targets)

    val_loss += loss.data[0]
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    # Print progressbar for monitoring via stdout
    progress_bar(batch_idx, len(val_loader), 
      'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

  cur_accuracy = 100.*correct/total
  
  return cur_accuracy

def main():
   
  best_accuracy = 0 
  checkpoint_str = 'ProposedNetwork_{}_fold{}'.format(args.type, args.fold)    

  # Create proposed network
  print('# Creating proposed network')

  net = ProposedNetwork(args.type)
  net = torch.nn.DataParallel(net).cuda()    

  if args.resume:
    print('#- Resume model from checkpoint')
    assert os.path.isdir('checkpoint'), \
      '#- Error: no checkpoint directory found!'

    checkpoint = torch.load(os.path.join(args.checkpoint_path, 
      checkpoint_str + '.checkpoint'))

    best_accuracy = checkpoint['acc']
    args.start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
  
  else:
    print('#- Initilize weights in proposed network')
    net.apply(init_weights)

  if USE_CUDA:
    net.cuda()
    cudnn.benchmark = True

  # Criterion, Optimizer  
  print('# Creating loss function and optimizer')

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
  if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])    

  # Training and validation data Preparation
  print('# Preparing trainning and validation data')

  train_data_path = os.path.join(args.data_path, 
    'age_{}_fold{}'.format('train', args.fold))
  val_data_path = os.path.join(args.data_path, 
    'age_{}_fold{}'.format('val', args.fold))
  
  transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
      (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
      (0.2023, 0.1994, 0.2010)),
  ])
  transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
      (0.2023, 0.1994, 0.2010)),
  ])

  train_set = AdienceDataset(image_dirs=[train_data_path], 
    transforms = transform_train)  # Give image_dirs as list format
  train_loader = torch.utils.data.DataLoader(train_set, 
    batch_size=args.train_batch_size, 
    shuffle=True, 
    num_workers=args.num_workers)
  val_set = AdienceDataset(image_dirs=[val_data_path], 
    transforms = transform_val) # Give image_dirs as list format
  val_loader = torch.utils.data.DataLoader(val_set, 
    batch_size=args.val_batch_size, 
    shuffle=False, 
    num_workers=args.num_workers)
  
  print('#- Trainining data = ', train_data_path)
  print('#- Validation data = ', val_data_path)

  # Training loop
  for epoch in range(args.start_epoch, args.epochs):
    print('\nEpoch:{}, Type:{}, Fold:{}'.format(epoch, args.type, args.fold))

    # Learning rate decay for every 30 epoches
    adjust_learning_rate(optimizer, epoch)

    # Training and validation for one epoch
    train(train_loader, net, criterion, optimizer)
    #train(val_loader, net, criterion, optimizer)
    cur_accuracy = validate(val_loader, net, criterion)

    # Remember best accuracy model and save checkpoint
    is_best = cur_accuracy > best_accuracy
    best_accuracy = max(cur_accuracy, best_accuracy)
    save_checkpoint({
      'net': net.state_dict(),
      'acc': best_accuracy,
      'epoch': epoch + 1,
      'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_str)

    print(' best accuracy = {}'.format(best_accuracy))     

def adjust_learning_rate(optimizer, epoch):
  """ Initial learning rate decayed by 10 every 30 epochs """

  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    
def save_checkpoint(state, is_best, checkpoint_str):
  """ Save checkpoint for each epoch and best accuracy model """

  print(' save checkpoint')
  torch.save(state, 
    os.path.join(args.checkpoint_path, checkpoint_str + '.checkpoint'))

  if is_best:
    print(' save best accurary model')

    shutil.copyfile(
      os.path.join(args.checkpoint_path, checkpoint_str + '.checkpoint'), 
      os.path.join(args.checkpoint_path, checkpoint_str + '.best_accuracy'))

if __name__ == '__main__':
  main()