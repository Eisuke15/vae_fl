import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
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

dim_red = 'TSNE' # 'TSNE' または 'PCA'

valid_dataset = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=2)

net = VAE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(args.net_path))
t_list = []
z_list = []
for x, t in tqdm(valid_dataloader, leave=False, desc="batch"):
    t_list += t.tolist()
    x = x.to(device)
    mean, logvar = net._encoder(x)
    z = net._sample_z(mean, logvar)
    z_list += z.tolist()

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
filename = os.path.basename(args.net_path).split('.')[0] + '.png'
filepath = f'graph/latent_space_{filename}'
plt.savefig(filepath, bbox_inches='tight')
plt.close()
print(f'image saved {filepath}')
