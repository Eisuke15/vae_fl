import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import trange

from common import device, fl_save_net_path, mnist_data_root, transform, mkdir_if_not_exists
from net import VAE

device = device()

parser = ArgumentParser(description='Generate learning curve using test images.')
parser.add_argument('--nepoch', type=int, help="number of epochs to generate images", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=100)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
args = parser.parse_args()

n_node = args.nnodes
n_epoch = args.nepoch

test_dataset = MNIST(root=mnist_data_root, train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

losses = [np.empty(n_epoch) for _ in range(n_node)]

net = VAE(args.nz, device)
net.to(device)
net.eval()

for epoch in trange(n_epoch):
    for n in trange(n_node, leave=False):
        train_loss = []
        net.load_state_dict(torch.load(os.path.join(fl_save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}_n{n:02d}.pth')))
        for images, _ in test_dataloader:
            images = images.to(device)
            y, _ = net(images)
            loss = F.binary_cross_entropy(y, images, reduction="sum") / images.size()[0]
            train_loss.append(loss.item())
        losses[n][epoch] = np.mean(train_loss)


plt.title('learning curve')
x = np.arange(0, n_epoch)
for i in range (n_node):
    plt.plot(x, losses[i], label=i)
plt.legend()
plt.ylabel('Test Loss')
plt.xlabel('Epoch')

image_path = mkdir_if_not_exists('./learning_curve')
plt.savefig(os.path.join(image_path, f'nz{args.nz:02d}_nnode{n_node:02d}.png'), bbox_inches='tight')