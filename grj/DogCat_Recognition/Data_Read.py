import os

import torch
from torch.utils.data import dataset
from torchvision.io import read_image
import re

label_dict = {"Cat": 0, "Dog": 1}


# create class for data process
class DogCatData(dataset.Dataset):
    def __init__(self, data_root):
        self.Data_path = data_root
        self.Data_list = os.listdir(data_root)

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, index):
        data = read_image(os.path.join(self.Data_path, self.Data_list[index]))
        data = min_max_normalization(data)
        path = self.Data_list[index]
        label_str = re.findall(r'^(\w+)_', path)[0]
        label = label_dict[label_str]
        return data, label


def min_max_normalization(data):
    max_data = torch.max(data)
    min_data = torch.min(data)
    data_normalized = (data - min_data) / (max_data - min_data)
    return data_normalized