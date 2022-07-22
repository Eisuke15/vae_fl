from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm, trange

from net import VAE

parser = ArgumentParser()
parser.add_argument('-e', '--nepoch', type=int, help="number of epochs to train for", default=1000)
parser.add_argument('-p', '--pre-nepoch', type=int, help='number of epochs of pre-self train', default=100)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2) for subset in subsets]

nets = [VAE(args.nz).to(device) for _ in range(n_node)]

optimizers = [Adam(net.parameters()) for net in nets]

fixed_z = torch.randn(64, args.nz).to(device)

# pre-self training
# train local model just by using the local data
for epoch in trange(args.pre_nepoch, desc="pre-self training epoch"):
    for node_num, (net, optimizer, dataloader) in tqdm(enumerate(zip(nets, optimizers, train_loaders)), leave=False, total=n_node, desc="node"):
        net.train()
        for images, _ in tqdm(dataloader, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

global_model = VAE(args.nz).to(device).state_dict()

for epoch in trange(args.nepoch+1, desc="federated learning epoch"):
    new_global_model = global_model.copy()

    # aggregate models
    for net in nets:
        parameters = net.state_dict()
        for key in new_global_model:
            new_global_model[key] = new_global_model[key] + (parameters[key] - global_model[key]) / n_node

    global_model = new_global_model

    for node_num, (net, optimizer, dataloader) in tqdm(enumerate(zip(nets, optimizers, train_loaders)), leave=False, total=n_node, desc="node"):
        # firstly, send global model to each nodes
        net.load_state_dict(new_global_model)

        # then update each models
        for images, _ in tqdm(dataloader, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

    if epoch%10 == 0:
        torch.save(global_model, f'nets/fl/e{epoch}_z{args.nz}.pth')
        global_vae = VAE(args.nz).to(device)
        global_vae.load_state_dict(global_model)
        global_vae.eval()
        save_image(global_vae._decoder(fixed_z), f'images/fl/e{epoch}_z{args.nz}.png')
