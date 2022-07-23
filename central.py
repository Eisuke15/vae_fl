from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm

from net import VAE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=1000)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=2)

dataset_val = MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
val_dataloader = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=2)

vae = VAE(args.nz)
vae.to(device)
optimizer = Adam(vae.parameters())

fixed_z = torch.randn(64, args.nz).to(device)

for epoch in range(args.nepoch+1):
    losses = []
    vae.train()

    for images, _ in tqdm(train_dataloader, leave=False):
        images = images.to(device)

        optimizer.zero_grad()

        KL_loss, reconstruction_loss = vae.loss(images)
        loss = KL_loss + reconstruction_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    print(f'epoch: {epoch}  Train Lower Bound: {np.mean(losses)}')

    if epoch % 10 == 0:
        torch.save(vae.state_dict(), f'nets/central/e{epoch}_z{args.nz}.pth')
        vae.eval()
        save_image(vae._decoder(fixed_z), f'images/central/e{epoch}_z{args.nz}.png')
