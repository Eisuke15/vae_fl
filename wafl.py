import json
from argparse import ArgumentParser

import numpy as np
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
parser.add_argument('--bs', type=int, help="input batch size", default=256)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

filename=f'./contact_pattern/rwp_n10_a0500_r100_p10_s01.json'
# filename=f'./contact_pattern/cse_n10_c10_b02_tt10_tp5_s01.json'
print(f'Loading ... {filename}')
with open(filename) as f :
    contact_list=json.load(f)

indices=torch.load('./noniid_filter/filter_r100_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=args.bs, shuffle=True, num_workers=2) for subset in subsets]

nets = [VAE(args.nz).to(device) for _ in range(n_node)]

optimizers = [Adam(net.parameters()) for net in nets]

fixed_z = torch.randn(64, args.nz).to(device)

# pre-self training
# train local model just by using the local data
for net in nets:
    net.train()

for epoch in range(args.pre_nepoch):
    for node_num, (net, optimizer, dataloader) in enumerate(zip(nets, optimizers, train_loaders)):
        train_losses = []
        for images, _ in tqdm(dataloader, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print(f'epoch: {epoch + 1} node: {node_num} Train Lower Bound: {np.mean(train_losses)}')

        if epoch == args.pre_nepoch - 1:
            save_image(net._decoder(fixed_z), f'images/wafl/first_z{args.nz}_n{node_num}.png')


train_losses_series_seq = [[] for _ in range(n_node)]

for epoch in range(args.nepoch+1):
    contact = contact_list[epoch]
    update_models = [net.state_dict() for net in nets]
    local_models = [net.state_dict() for net in nets]

    # exchange models
    for i, (local_model, update_model) in enumerate(zip(local_models, update_models)):
        neighbors = contact[str(i)]
        if neighbors:
            for key in local_model:
                update_model[key] = sum([local_models[neighbor][key] for neighbor in neighbors] + [local_model[key]])/(len(neighbors) + 1)


    for node_num, (net, update_model, optimizer, dataloader, train_losses_series) in enumerate(zip(nets, update_models, optimizers, train_loaders, train_losses_series_seq)):
        # load updated models
        net.load_state_dict(update_model)

        if epoch%10 == 0:
            save_image(net._decoder(fixed_z), f'images/wafl/e{epoch}_z{args.nz}_n{node_num}_before.png')

        train_losses = []
        for images, _ in tqdm(dataloader, leave=False, desc=f"node {node_num}"):
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = net.loss(images)
            loss = KL_loss + reconstruction_loss

            # skip train when no neighbor
            if contact[str(node_num)]:
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

        train_losses_series.append(np.mean(train_losses))
        print(f'epoch: {epoch}  Lower Bound: {np.mean(train_losses)}')

        if epoch%10 == 0:
            torch.save(net.state_dict(), f'nets/wafl/e{epoch}_z{args.nz}_n{node_num}.pth')
            save_image(net._decoder(fixed_z), f'images/wafl/e{epoch}_z{args.nz}_n{node_num}_after.png')

np.save('learning_curve/wafl/losses.npy', np.array(train_losses_series_seq))
