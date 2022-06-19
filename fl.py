import os
from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm, trange

from common import device, fl_save_net_path, mnist_data_root
from net import VAE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=25)
parser.add_argument('-p', '--pre-nepoch', type=int, help='number of epochs of pre-self train', default=20)
parser.add_argument('--bs', type=int, help="input batch size", default=64)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=100)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = args.nnodes

device = device(args.gpu_num)
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root=mnist_data_root, train=True, download=True, transform=transform)
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=args.bs, shuffle=True, num_workers=2) for subset in subsets]

nets = [VAE(args.nz, device).to(device) for _ in range(n_node)]

optimizers = [Adam(net.parameters()) for net in nets]

# pre-self training
# train local model just by using the local data
for net in nets:
    net.train()

for epoch in trange(args.pre_nepoch, desc="pre-self training epoch"):
    for node_num, (net, optimizer, dataloader) in tqdm(enumerate(zip(nets, optimizers, train_loaders)), leave=False, total=n_node, desc="node"):
        for images, _ in tqdm(dataloader, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()


global_model = VAE(args.nz, device).to(device).state_dict()

for epoch in trange(args.nepoch, desc="federated learning epoch"):
    new_global_model = global_model.copy()

    # aggregate models
    for net in nets:
        parameters = net.state_dict()
        for key in new_global_model:
            new_global_model[key] += (parameters[key] - global_model[key]) / n_node

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
        
        torch.save(net.state_dict(), os.path.join(fl_save_net_path, f'mnist_vae_nz{args.nz:02d}_e{epoch+1:04d}_n{node_num}.pth'))
