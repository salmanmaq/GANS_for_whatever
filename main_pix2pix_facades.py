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
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import models.pix2pix as pix2pix
from facadesDataLoader import facadesDataset
import utils

# TODO: Incremental training over different network layers

parser = argparse.ArgumentParser(description='PyTorch DCGAN Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', default=16, type=int,
            help='mini-batch size (default: 16)')
parser.add_argument('--imageSize', default=224, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--nz', default=100, type=int,
            help='size of the latent z vector')
parser.add_argument('--ngf', default=64, type=int,
            help='Generator Feature Map depth (default: 64)')
parser.add_argument('--ndf', default=64, type=int,
            help='Discriminator Feature Map depth (default: 64)')
parser.add_argument('--lr', default=0.0002, type=float,
            help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', default=0.5, type=float,
            help='beta1 for adam (default: 0.5)')
parser.add_argument('--ngpu', type=int, default=1,
            help='number of GPUs to use')
parser.add_argument('--netG', default='',
            help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
            help="path to netD (to continue training)")
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)
parser.add_argument('--inputType', default='segM', type=str,
            help='Use either random noise or the segmentation mask as input')
parser.add_argument('--labelSmoothing', default='False', type=str,
            help='Use label smoothing or not')
parser.add_argument('--verbose', default='False', type=str,
            help='Prints certain messages which user can specify if true')
parser.add_argument('--useL1', default='False', type=str,
            help='Use L1 loss for generator or not')
parser.add_argument('--Lambda', default=10, type=int,
            help='Lambda parameter to weigh the L1 loss')

use_gpu = torch.cuda.is_available()

global args
args = parser.parse_args()
print(args)

if args.verbose == 'True':
    args.verbose = True
elif args.verbose == 'False':
    args.verbose = False

if args.labelSmoothing == 'True':
    args.labelSmoothing = True
elif args.labelSmoothing == 'False':
    args.labelSmoothing = False

if args.useL1 == 'True':
    args.useL1 = True
elif args.useL1 == 'False':
    args.useL1 = False

def main():

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.manual_seed)

    cudnn.benchmark = True

    # # Optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

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
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
    }

    data_dir = 'Dataset/facades'

    image_datasets = {x: facadesDataset(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nc = 3

    # Initialize the Generator Network
    netG = pix2pix._netG(args.verbose)
    netG.apply(pix2pix.weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print(netG)

    # Initialize the Discriminator Network
    netD = pix2pix._netD(ndf)
    netD.apply(pix2pix.weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print(netD)

    # Define loss function (criterion) and optimizer
    criterion = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
    noise = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
    fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(args.batchSize)

    if use_gpu:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        criterion_L1.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    # Define the optimizers for the Generator and the Discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr,
                            betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr,
                            betas=(args.beta1, 0.999))

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train(dataloaders['train'], netG, netD, criterion, criterion_L1, optimizerG,
              optimizerD, epoch, input, noise, fixed_noise, label,
              nz)

        # Save checkpoints
        #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.save_dir, epoch))
        #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.save_dir, epoch))

def train(train_loader, netG, netD, criterion, criterion_L1, optimizerG,
        optimizerD, epoch, input, noise, fixed_noise, label, nz):
    '''
        Run one training epoch
    '''

    for i, (img, gt) in enumerate(train_loader):

        # Generate smoothed labels
        if args.labelSmoothing:
            real_label = random.uniform(0.7, 1.2)
            fake_label = random.uniform(0.0, 0.3)
        else:
            real_label = 1
            fake_label = 0

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        gtForViz = gt
        if args.verbose:
            print('GT shape')
            print(gt.shape)
        if use_gpu:
            img = img.cuda()
            gt = gt.cuda()
        netD.zero_grad()
        real_cpu = img
        batch_size = real_cpu.size(0)
        if use_gpu:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        input.normal_(-1, 1)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        if args.verbose:
            print('Output size - real: ')
            print(output.data.shape)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        if args.inputType == 'segM':
            noise = gt
            if args.verbose:
                print('GT noise shape')
                print(gt.shape)
            if args.verbose:
                print('Noise Reshaped: ')
                print(noise.shape)
        else:
            noise = torch.randn(batch_size, 3, args.imageSize, args.imageSize)

        if use_gpu:
            noise = noise.cuda()
        #noiseForViz = noise.resize_(batch_size, nz, 1, 1)
        noise.normal_(-1, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        #fakeForViz = fake.detach().normal_(0, 1)
        fakeForViz = fake
        if args.verbose:
            print('Fake img size: ')
            print(fake.data.shape)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        if args.verbose:
            print('Output size - fake: ')
            print(output.data.shape)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        if args.useL1:
            errL1 = criterion_L1(input, fake.detach())
            errD = errD_real + errD_fake + args.Lambda * errL1
        else:
            errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        if args.useL1:
            errG = criterion(output, labelv) + args.Lambda * errL1
        else:
            errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        if args.useL1:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(train_loader),
                     errD.data[0], errG.data[0], errL1.data[0], D_x, D_G_z1, D_G_z2))
        else:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.epochs, i, len(train_loader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % args.print_freq == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % args.save_dir,
                    normalize=True)
            #fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (args.save_dir, epoch),
                    normalize=True)
            utils.displaySamples(real_cpu, fakeForViz, gtForViz, use_gpu)


if __name__ == '__main__':
    main()
