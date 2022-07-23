import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import trange

from net import VAE

parser = ArgumentParser(description='Generate learning curve using test images.')
parser.add_argument('architecture', choices=["fl", "wafl"], help="architecture", default='fl')
parser.add_argument('--nepoch', type=int, help="number of epochs to generate images", default=1000)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

n_node = 10
n_epoch = args.nepoch

test_dataset = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

net = VAE(args.nz)
net.to(device)
net.eval()

if args.architecture == 'fl':
    fl_losses = []
    central_losses = []

    for epoch in trange(0, n_epoch+1, 10):
        test_losses = []
        net.load_state_dict(torch.load(f'nets/fl/e{epoch}_z{args.nz}.pth'))
        for images, _ in test_dataloader:
            images = images.to(device)
            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss
            test_losses.append(loss.item())
        fl_losses.append(np.mean(test_losses))

    for epoch in trange(0, n_epoch+1, 10):
        test_losses = []
        net.load_state_dict(torch.load(f'nets/central/e{epoch}_z{args.nz}.pth'))
        for images, _ in test_dataloader:
            images = images.to(device)
            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss
            test_losses.append(loss.item())
        central_losses.append(np.mean(test_losses))
    

    x = np.arange(0, n_epoch+1, 10)
    plt.plot(x, fl_losses, label="FL")
    plt.plot(x, central_losses, label="Central")
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend()
    file_path = f'graph/fl_nz{args.nz:02d}.png'
    plt.savefig(file_path, bbox_inches='tight')
    print(f'image saved {file_path}')

elif args.architecture == 'wafl':
    losses = [[] for _ in range(n_node)]

    for epoch in trange(0, n_epoch+1, 10):
        for n in trange(n_node, leave=False):
            test_losses = []
            net.load_state_dict(torch.load(f'nets/wafl/e{epoch}_z{args.nz}_n{n}.pth'))
            for images, _ in test_dataloader:
                images = images.to(device)
                KL_loss, reconstruction_loss = net.loss(images)
                loss = KL_loss + reconstruction_loss
                test_losses.append(loss.item())
            losses[n].append(np.mean(test_losses))

    x = np.arange(0, n_epoch+1, 10)
    for i in range (n_node):
        plt.plot(x, losses[i], label=i)
    plt.legend()
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    file_path = f'learning_curve/wafl/nz{args.nz:02d}.png'
    plt.savefig(file_path, bbox_inches='tight')
    print(f'image saved {file_path}')
