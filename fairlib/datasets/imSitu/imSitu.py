from fairlib.src.utils import seed_everything
import numpy as np
import os
from fairlib.datasets.utils.download import download
from fairlib.datasets.utils.unzip import unzip


from PIL import Image
import torch
from torchvision import datasets as tv_datasets


class imSitu:

    _NAME = "imSitu"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size, download_fuldata=False):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.download_fuldata = download_fuldata



    def download_files(self):
        # Download subset ids
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/verb_classification/data/train.data", 
            dest_folder = self.dest_folder
            )
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/verb_classification/data/val.data", 
            dest_folder = self.dest_folder
            )
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/verb_classification/data/test.data", 
            dest_folder = self.dest_folder
            )
        
        if self.download_fuldata:
            download(
                "https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar",
                dest_folder=self.dest_folder)
            from pathlib import  Path
            unzip(
                file_path=Path(dest_folder=self.dest_folder)/"of500_images_resized.tar",
                dest_path=Path(dest_folder=self.dest_folder)/"imSitu"
                )

    def prepare_data(self):
        self.download_files()