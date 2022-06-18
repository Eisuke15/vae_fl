import os
import torch


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def device(device_num):
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")