'''
DCGAN - pix2pix: Image-to-Image Translation with Conditional Adversarial Networks
Code apapted from:
https://github.com/pytorch/examples/blob/master/dcgan/main.py
and
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
and
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

https://arxiv.org/abs/1611.07004
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *

def weights_init(m):
    '''Randomly initialize Generator and Discriminator weights'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    '''The Generator Network based on UNet Architecture'''

    def __init__(self, verbose):
        super(_netG, self).__init__()
        self.verbose = verbose
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 3)

    def forward(self, x):
        if self.verbose:
            print('Network Forward pass:')
            print(x.data.shape)
        x1 = self.inc(x)
        if self.verbose:
            print(x1.data.shape)
        x2 = self.down1(x1)
        if self.verbose:
            print(x2.data.shape)
        x3 = self.down2(x2)
        if self.verbose:
            print(x2.data.shape)
        x4 = self.down3(x3)
        if self.verbose:
            print(x4.data.shape)
        x5 = self.down4(x4)
        if self.verbose:
            print(x5.data.shape)
        x = self.up1(x5, x4)
        if self.verbose:
            print(x.data.shape)
        x = self.up2(x, x3)
        if self.verbose:
            print(x.data.shape)
        x = self.up3(x, x2)
        if self.verbose:
            print(x.data.shape)
        x = self.up4(x, x1)
        if self.verbose:
            print(x.data.shape)
        x = self.outc(x)
        if self.verbose:
            print(x.data.shape)
        x = F.tanh(x)
        if self.verbose:
            print(x.data.shape)
        return x

class _netD(nn.Module):
    '''The Discriminator Network based on PatchGAN'''

    def __init__(self, ndf):
        super(_netD, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(3, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        sequence += [nn.Conv2d(1, 1, kernel_size=26, stride=13, padding=padw)]

        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input).view(-1,1).squeeze(1)
