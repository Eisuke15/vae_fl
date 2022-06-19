import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import MNIST
from tqdm import tqdm, trange

from common import (central_save_net_path, device, mkdir_if_not_exists,
                    mnist_data_root, transform)
from net import VAE

parser = ArgumentParser(description='plot latent space')
parser.add_argument('--nepoch', type=int, help="number of epochs to plot latent space", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=2)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
parser.add_argument('--fl', action="store_true", help="Plot latent space for federated learning.")
args = parser.parse_args()

device=device()

dim_red = 'TSNE' # 'TSNE' または 'PCA'

valid_dataset = MNIST(root=mnist_data_root, train=False, download=True, transform=transform)

if not args.fl:
    net = VAE(args.nz)
    net.to(device)
    net.eval()
    for epoch in trange(args.nepoch, desc="epoch"):
        net.load_state_dict(torch.load(os.path.join(central_save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}.pth')))
        t_list = []
        z_list = []
        for x, t in tqdm(valid_dataset, leave=False, desc="batch"):
            t_list.append(t)
            x = x.to(device).unsqueeze(0)
            y, z = net(x)
            z_list.append(z.cpu().detach().numpy()[0])

        if args.nz == 2:
            z_list = np.array(z_list).T
        else:
            if dim_red == 'TSNE':
                from sklearn.manifold import TSNE
                z_list = TSNE(n_components=2).fit_transform(z_list).T
            elif dim_red == 'PCA':
                from sklearn.decomposition import PCA
                z_list = PCA(n_components=2).fit(np.array(z_list).T).components_
            else:
                raise ValueError()

        colors = ['khaki', 'lightgreen', 'cornflowerblue', 'violet', 'sienna', 'darkturquoise', 'slateblue', 'orange', 'darkcyan', 'tomato']
        plt.figure(figsize=(8,8))
        plt.scatter(z_list[0], z_list[1], s=0.7, c=[colors[t] for t in t_list])

        # 凡例を追加
        for i in range(10):
            plt.scatter([],[], c=colors[i], label=i)
        plt.legend()
        image_path = mkdir_if_not_exists('./latent_space/central')
        plt.savefig(os.path.join(image_path, f'nz{args.nz:02d}_e{epoch+1:04d}.png'), bbox_inches='tight')
        plt.close()
