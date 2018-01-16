# Experiments with Generative Adversarial Networks (GANs)

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

Code adapted from PyTorch DCGAN example (https://github.com/pytorch/examples/tree/master/dcgan)

## Dataset

Currently, the miccaiSeg dataset is in development, but would be open-sourced later.

## Usage
```
usage: main.py [-h] [--workers WORKERS] [--epochs EPOCHS] [--batchSize BATCHSIZE]
               [--imageSize IMAGESIZE] [--nz NZ] [e--ngf NGF] [--ndf NDF]
               [--lr LR] [--beta1 BETA1] [--ngpu NGPU] [--netG NETG]
               [--netD NETD] [--print-freq] [--save-dir SAVE_DIR]

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


```
