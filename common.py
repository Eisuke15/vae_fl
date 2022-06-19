import os

import torch
from torchvision import transforms
import os


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def device(device_num = 0):
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


mnist_data_root = mkdir_if_not_exists('./data')
central_net_dir = mkdir_if_not_exists('./net/central')
fl_net_dir = mkdir_if_not_exists('./net/fl')

def central_net_path(nz, epoch):
    return os.path.join(central_net_dir, f'nz{nz:03d}/e{epoch+1:04d}.pth')

def fl_net_path(nz, epoch, node):
    return os.path.join(fl_net_dir,  f'nz{nz:03d}/e{epoch+1:04d}_n{node:02d}.pth')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])
