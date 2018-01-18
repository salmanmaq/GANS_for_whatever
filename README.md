# Experiments with Generative Adversarial Networks (GANs)

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

## Dataset

Currently, the miccaiSeg dataset is in development, but would be open-sourced later.
You can modify the miccaiSegDataLoader class to use the code on your own dataset.

The Facades dataset can be downloaded from: https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
Courtesy of [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

'''
wget -c https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
'''

## Usage
```
usage: main.py [-h] [--workers WORKERS] [--epochs EPOCHS] [--batchSize BATCHSIZE]
               [--imageSize IMAGESIZE] [--nz NZ] [e--ngf NGF] [--ndf NDF]
               [--lr LR] [--beta1 BETA1] [--ngpu NGPU] [--netG NETG]
               [--netD NETD] [--print-freq] [--save-dir] [--verbose True/False]

optional arguments:
  -h, --help            Show this help message and exit
  --workers WORKERS     Number of data loading workers
  --batchSize BATCHSIZE
                        Input batch size
  --imageSize IMAGESIZE
                        The height / width of the input image to network
  --nz NZ               Size of the latent z vector
  --ngf NGF
  --ndf NDF
  --epochs EPOCHS       Number of epochs to train for
  --lr LR               Learning rate, default=0.0002
  --beta1 BETA1         Beta1 for adam. default=0.5
  --ngpu NGPU           Number of GPUs to use
  --netG NETG           Path to netG (to continue training)
  --netD NETD           Path to netD (to continue training)
  --print-freq          Frequency with which to print training statistics and save the generated samples
  --save-dir SAVE_DIR   Directory to save the model and generated samples
  --verbose             Prints outs relevant informational text for debugging such as tensor shapes

```
