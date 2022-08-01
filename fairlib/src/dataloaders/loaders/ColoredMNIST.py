import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os
import torch
from torchvision import transforms


class MNISTDataset(BaseDataset):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))])

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "colored_MNIST_{}.pt".format(self.split))

        data = torch.load(self.data_dir)

        self.X = [self.transform(_img) for _img in data[0]]
        self.y = data[1]
        self.protected_label = data[2]