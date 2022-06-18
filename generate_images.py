import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from common import central_save_net_path, device, mkdir_if_not_exists
from net import VAE

device = device()

parser = ArgumentParser(description='Generate images from random latent vectors using the learned model.')
parser.add_argument('--nepoch', type=int, help="number of epochs to generate images", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=100)
args = parser.parse_args()

central_generated_images_path = mkdir_if_not_exists('./generated_images/central')

net = VAE(args.nz, device)
net.to(device)
net.eval()

fixed_z = torch.randn(12 * 12, args.nz).to(device)

for epoch in range(args.nepoch):
    print(f'generating epoch={epoch} images ...')
    net.load_state_dict(torch.load(os.path.join(central_save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}.pth')))
    y = net._decoder(fixed_z)
    images = y.view(-1, 1, 28, 28)
    save_image(images, os.path.join(central_generated_images_path, f'z{args.nz:02d}_e{epoch+1:04d}_.png'), nrow=12, pad_value=1)
    


