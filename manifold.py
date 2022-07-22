from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from tqdm import tqdm

from net import VAE

parser = ArgumentParser(description='plot latent space')
parser.add_argument('net_path', help="network file path")
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

valid_dataset = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

vae = VAE(args.nz).to(device)
vae.load_state_dict(torch.load(args.net_path))
vae.eval()

x0, l0 = valid_dataset[3]
x0 = x0.to(device)
x0 = x0.unsqueeze(0)
mean0, logvar0 = vae._encoder(x0)
z0 = vae._sample_z(mean0, logvar0)

x1, l1 = valid_dataset[0]
x1 = x1.to(device)
x1 = x1.unsqueeze(0)
mean1, logvar1 = vae._encoder(x1)
z1 = vae._sample_z(mean1, logvar1)

z_linear = torch.cat([z1 * (i * 0.1) + z0 * ((9 - i) * 0.1) for i in range(10)])
z_linear = z_linear.view((10, -1))
y = vae._decoder(z_linear)
save_image(y, f'graph/image_{l0}_{l1}.png', nrow=10, pad_value=1, padding=1)
