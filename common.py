import os

import torch
from torchvision import transforms


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def device(device_num = 0):
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


mnist_data_root = mkdir_if_not_exists('./data')
central_save_net_path = mkdir_if_not_exists('./net/central')
fl_save_net_path = mkdir_if_not_exists('./net/fl')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])
