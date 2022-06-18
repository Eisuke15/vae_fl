from argparse import ArgumentParser
from random import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from net import VAE
from utils import device, mkdir_if_not_exists
from torch.utils.data.dataset import Subset


parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=25)
parser.add_argument('--pre-nepoch', type=int, help='number of epochs of pre-self train', default=20)
parser.add_argument('--bs', type=int, help="input batch size", default=64)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=100)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
args = parser.parse_args()

n_node = args.nnodes

device = device()
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

root = mkdir_if_not_exists('./data')

dataset_train = MNIST(root=root, train=True, download=True, transform=transform)
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=args.bs, shuffle=True, num_workers=2) for subset in subsets]

dataset_test = MNIST(root=root, train=False, download=True, transform=transform)
test_dataloader = DataLoader(dataset_train, batch_size=args.bs, shuffle=False, num_workers=2)

nets = [VAE(args.nz).to(device) for _ in range(n_node)]
local_model=[{} for i in range(n_node)]
update_model=[{} for i in range(n_node)]

optimizers = [Adam(net.parameters()) for net in nets]

# pre-self training
# train local model just by using the local data
for net in nets:
    net.train()

for epoch in range(args.pre_nepoch):
    for n in range(n_node):
        optimizer = optimizers[n]
        net = nets[n]
        for images, _ in test_dataloader[n]:
            images = images.to(device)

            optimizer.zero_grad()

            y, _ = net(images)
            KL_loss, reconstruction_loss = net.loss()
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

