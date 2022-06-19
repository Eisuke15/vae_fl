import os
from argparse import ArgumentParser
from random import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from common import central_save_net_path, device, mnist_data_root
from net import VAE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=100)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = device(args.gpu_num)
print(device)

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = MNIST(root=mnist_data_root, train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

vae = VAE(args.nz, device)
vae.to(device)
optimizer = Adam(vae.parameters())

for epoch in range(args.nepoch):
    losses = []
    vae.train()

    for images, _ in train_dataloader:
        images = images.to(device)

        optimizer.zero_grad()

        KL_loss, reconstruction_loss = vae.loss(images)
        loss = KL_loss + reconstruction_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    print(f'epoch: {epoch + 1}  Train Lower Bound: {sum(losses)/len(losses)}')

    torch.save(vae.state_dict(), os.path.join(central_save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}.pth'))
