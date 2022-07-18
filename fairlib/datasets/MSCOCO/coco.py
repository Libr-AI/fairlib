from fairlib.src.utils import seed_everything
import numpy as np
import os
from fairlib.datasets.utils.download import download
from fairlib.datasets.utils.unzip import unzip

from PIL import Image
import torch
from torchvision import datasets as tv_datasets


# TODO: download the dataset with gstuil APIs
# TODO: download files and unzip files in parallel
class COCO:

    _NAME = "MSCOCO"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size, download_fuldata=False):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.download_coco = download_fuldata



    def download_files(self):
        # Download subset ids
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/object_multilabel/data/train.data", 
            dest_folder = self.dest_folder
            )
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/object_multilabel/data/val.data", 
            dest_folder = self.dest_folder
            )
        download(
            url= r"https://raw.githubusercontent.com/uvavision/Balanced-Datasets-Are-Not-Enough/master/object_multilabel/data/test.data", 
            dest_folder = self.dest_folder
            )
        
        if self.download_coco:
            from pathlib import Path
            # Download coco 2014
            download(
                "http://images.cocodataset.org/zips/train2014.zip",
                dest_folder=self.dest_folder)
            unzip(
                file_path=Path(dest_folder=self.dest_folder)/"train2014.tar",
                dest_path=Path(dest_folder=self.dest_folder)/"train2014"
                )
            download(
                "http://images.cocodataset.org/zips/val2014.zip",
                dest_folder=self.dest_folder)
            unzip(
                file_path=Path(dest_folder=self.dest_folder)/"val2014.tar",
                dest_path=Path(dest_folder=self.dest_folder)/"val2014"
                )
            download(
                "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                dest_folder=self.dest_folder)
            unzip(
                file_path=Path(dest_folder=self.dest_folder)/"annotations_trainval2014.tar",
                dest_path=Path(dest_folder=self.dest_folder)/"annotations_trainval2014"
                )

    def prepare_data(self):
        self.download_files()