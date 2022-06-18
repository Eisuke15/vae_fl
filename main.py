import os
from argparse import ArgumentParser
from random import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from net import VAE
from common import device, mkdir_if_not_exists

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

root = mkdir_if_not_exists('./data')
save_net_path = mkdir_if_not_exists('./net/central')
dataset_train_valid = MNIST(root=root, train=True, download=True, transform=transform)
dataset_test = MNIST(root=root, train=False, download=True, transform=transform)
total_train_data_num = len(dataset_train_valid)
val_data_num = int(total_train_data_num * 0.3)

dataset_train, dataset_valid = random_split(dataset_train_valid, [total_train_data_num - val_data_num, val_data_num])
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)
test_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=2)

vae = VAE(args.nz, device)
vae.to(device)
optimizer = Adam(vae.parameters())

for epoch in range(args.nepoch):
    losses = []
    
    vae.train()

    for image, label in train_dataloader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        y, _ = vae(image)
        KL_loss, reconstruction_loss = vae.loss(image)
        loss = KL_loss + reconstruction_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    losses_val = []
    vae.eval()

    for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)

        y, _ = vae(image)
        KL_loss, reconstruction_loss = vae.loss(image)
        loss = KL_loss + reconstruction_loss

        losses_val.append(loss.detach())
    
    print(f'epoch: {epoch + 1}  Train Lower Bound: {sum(losses)/len(losses)}  Valid Lower Bound: {sum(losses_val)/len(losses_val)}')

    torch.save(vae.state_dict(), os.path.join(save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}.pth'))
