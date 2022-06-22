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

def central_path(base, nz, epoch):
    central_dir = mkdir_if_not_exists(f'./{base}/central/nz{nz:03d}')
    return os.path.join(central_dir, f'e{epoch+1:04d}.' + ('pth' if base == 'net' else 'png'))

def fl_path(base, nz, epoch, node):
    fl_dir = mkdir_if_not_exists(f'./{base}/fl/nz{nz:03d}')
    return os.path.join(fl_dir,  f'e{epoch+1:04d}_n{node:02d}.' + ('pth' if base == 'net' else 'png'))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])
