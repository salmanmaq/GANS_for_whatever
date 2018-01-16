'''
Unsupervised feature learning using Deep Convolutional Generative Adversarial
Network

This code is using for unsupervised feature learning for small datasets
using the DCGAN.

Code adapted from: https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import argparse
import os
import shutil
import time

import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import dcgan
from miccaiSegDataLoader import miccaiSegDataset

parser = argparse.ArgumentParser(description='PyTorch DCGAN Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N',
            help='mini-batch size (default: 2)')
parser.add_argument('--image-size', default=64, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--nz', default=100, type=int,
            help='size of the latent z vector')
parser.add_argument('--ngf', default=64, type=int)
parser.add_argument('--ndf', default=64, type=int)
parser.add_argument('--lr', default=0.0002, type=float,
            help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', default=0.5, type=float,
            help='beta1 for adam (default: 0.5)')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='',
            help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
            help="path to netD (to continue training)")
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
parser.add_argument('--pre-trained', dest='pretrained', action='store_true',
            help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)

best_prec1 = np.inf
use_gpu = torch.cuda.is_available()

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.manual-seed is None:
        args.manual-seed = random.randint(1, 10000)
    random.seed(opt.manual-seed)
    torch.manual_seed(args.manual-seed)
    if use_gpu:
        torch.manual_seed_all(args.manual-seed)

    cudnn.benchmark = True

    if use_gpu:
        model.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    #     ]),
    # }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale((args.image-size, args.image-size)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Scale((args.image-size, args.image-size)),
            transforms.ToTensor(),
        ]),
    }

    data_dir = 'miccaiSeg'

    image_datasets = {x: miccaiSegDataset(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    if args.evaluate:
        validate(dataloaders['val'], model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        train(dataloaders['train'], model, criterion, optimizer, epoch)

        # Evaulate on validation set
        prec1 = validate(dataloaders['val'], model, criterion)
        prec1 = prec1.cpu().data.numpy()

        # Remember best prec1 and save checkpoint
        print(prec1)
        print(best_prec1)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, model, criterion, optimizer, epoch):
    '''
        Run one training epoch
    '''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()

    for i, input in enumerate(train_loader):
        # Measure Data loading time
        data_time.update(time.time() - end)
        target = input.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        if args.half:
            input_var = input_var.half()

        target_varR, target_varG, target_varB = splitInput(target)
        target_var = torch.autograd.Variable(target)
        target_varR = torch.autograd.Variable(target_varR).cuda()
        target_varG = torch.autograd.Variable(target_varG).cuda()
        target_varB = torch.autograd.Variable(target_varB).cuda()

        # Compute output
        outputR, outputG, outputB = model(input_var)
        lossR = criterion(outputR, target_varR)
        lossG = criterion(outputG, target_varG)
        lossB = criterion(outputB, target_varB)
        loss = lossR + lossG + lossB

        output = concatenateChannels(outputR, outputG, outputB)
        if i % args.print_freq == 0:
            displaySamples(target_var, output)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #output = output.float()
        loss = loss.float()

        # Measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    '''
        Run evaluation
    '''

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, input in enumerate(val_loader):
        target = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        if args.half:
            input_var = input_var.half()

        target_varR, target_varG, target_varB = splitInput(target)
        target_var = torch.autograd.Variable(target)
        target_varR = torch.autograd.Variable(target_varR, volatile=True).cuda()
        target_varG = torch.autograd.Variable(target_varG, volatile=True).cuda()
        target_varB = torch.autograd.Variable(target_varB, volatile=True).cuda()

        # Compute output
        outputR, outputG, outputB = model(input_var)
        lossR = criterion(outputR, target_varR)
        lossG = criterion(outputG, target_varG)
        lossB = criterion(outputB, target_varB)
        loss = lossR + lossG + lossB

        output = concatenateChannels(outputR, outputG, outputB)

        output = output.float()
        loss = loss.float()

        if i % args.print_freq == 0:
            displaySamples(target_var, output)

        # Measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses))

    return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
        Save the training model
    '''
    torch.save(state, filename)

class AverageMeter(object):
    '''
        Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    '''
        Sets the learning rate to the initial LR decayed by a factor of 10
        every 30 epochs
    '''

    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    '''
        Computes the precision@k for the specified values of k
    '''

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def displaySamples(input, output):
    ''' Display the original and the reconstructed image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, output image
    '''
    if use_gpu:
        input = input.cpu()
        output = output.cpu()

    unNorm = UnNormalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    input = input.data.numpy()
    input = np.transpose(np.squeeze(input[0,:,:,:]), (1,2,0))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    #input = unNorm(input)

    output = output.data.numpy()
    output = np.transpose(np.squeeze(output[0,:,:,:]), (1,2,0))
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    #output = unNorm(output)

    cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)

    cv2.imshow('Input Image', input)
    cv2.imshow('Reconstructed Image', output)
    cv2.waitKey(1)

def concatenateChannels(r, g, b):
    '''
        Concatenate the R, G, and B channels to form an RGB Image
        for channel-wise reconstruction experiments.

        Args:
            R-channel Tensor, G-channel Tensor, B-channel Tensor

        Retuns RGB PyTorch Tensor
    '''

    return torch.cat((r,g,b), 1)

def splitInput(input):
    '''
        Splits an Input Image PyTorch Tensor into it's constituent channels
        RGB.
    '''

    if use_gpu:
        input = input.cpu()

    input = input.numpy()

    # Iterate over each image tensor in a batch
    channelR = np.expand_dims(np.zeros((input.shape[2], input.shape[3])), axis=0)
    channelG = np.expand_dims(np.zeros((input.shape[2], input.shape[3])), axis=0)
    channelB = np.expand_dims(np.zeros((input.shape[2], input.shape[3])), axis=0)

    #print(channelR)

    for i in range(args.batch_size):

        img = input[i,:,:,:]

        channelR = np.concatenate((channelR, np.expand_dims(img[0,:,:], axis = 0)), 0)
        channelG = np.concatenate((channelG, np.expand_dims(img[1,:,:], axis = 0)), 0)
        channelB = np.concatenate((channelB, np.expand_dims(img[2,:,:], axis = 0)), 0)

    # Strip the extra empty channel added initially
    channelR = channelR[1:,:,:]
    channelG = channelG[1:,:,:]
    channelB = channelB[1:,:,:]

    # Convert back to torch tensors
    channelR = torch.from_numpy(channelR).float()
    channelG = torch.from_numpy(channelG).float()
    channelB = torch.from_numpy(channelB).float()

    return channelR, channelG, channelB

#TODO: Move auxillary code to utils

def imagetoLabelTesnor(input):
    '''
        Splits the input image into the three RGB channels, and then
        converts each channel into a one-hot tensor representing the
        pixel distribution in the image. Useful for channel-wise
        prediction.

        Args:
            seg: The segmented RGB image

        Output:
            The one-hot tensor representation of the image
    '''

    r, g, b = splitInput(input)


def segmentedImagetoLabel(seg):
    '''
        Gets the segmented image and returns the one-hot tensor representing
        the segmented distribution.

        Args:
            seg: The segmented RGB image

        Output:
            The one-hot tensor representation of the image
    '''

    if use_gpu:
        seg = seg.cpu()

    seg = seg.data.numpy()




class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Args:
            image (Image): Numpy ndarray of size (H, W, C) to be normalized.
        Returns:
            Numpy ndarray: Normalized image.
        """
        im = np.zeros_like(image)
        for i in range(image.shape[2]):
            im[i,:,:] = np.maximum(np.zeros_like(image[i,:,:]),
            np.minimum(np.ones_like(image[i,:,:]),
            (image[i,:,:] * self.mean[i]) + self.std[i]))
            # The normalize code -> t.sub_(m).div_(s)
        return im

if __name__ == '__main__':
    main()