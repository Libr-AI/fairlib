# wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy -P data/deepmoji
# wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy -P data/deepmoji
# wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy -P data/deepmoji
# wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy -P data/deepmoji

from fairlib.datasets.utils.download import download
from fairlib.src.utils import seed_everything
import numpy as np
import os

class Moji:

    _NAME = "Moji"
    _SPLITS = ["pos_pos", "pos_neg", "neg_pos", "neg_neg"]

    def __init__(self, dest_folder):
        self.dest_folder = dest_folder


    def read_data_file(self, input_file: str):
        vecs = np.load(input_file)
        
        seed_everything(2020)
        np.random.shuffle(vecs)

        return vecs[:40000], vecs[40000:42000], vecs[42000:44000]

    def download_files(self):

        for split in self._SPLITS:
            download(
                url = "https://storage.googleapis.com/ai2i/nullspace/deepmoji/{}.npy".format(split), 
                dest_folder = self.dest_folder
                )

    def data_splits(self):
        for split in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
            train, dev, test = self.read_data_file(self.dest_folder + '/' + split + '.npy')
            for split_dir, data in zip(['train', 'dev', 'test'], [train, dev, test]):
                os.makedirs(self.dest_folder + '/' + split_dir, exist_ok=True)
                np.save(self.dest_folder + '/' + split_dir + '/' + split + '.npy', data)

    def prepare_data(self):
        self.download_files()
        self.data_splits()