from fairlib.src.utils import seed_everything
import numpy as np
import os

from PIL import Image
import torch
from torchvision import datasets as tv_datasets

def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([
            arr,
            np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([
            np.zeros((h, w, 1), dtype=dtype),
            arr,
            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr

def process_colored_MNIST(loaded_data, ratio=0.2):
    im_list, target_class, protected_class = [], [], []
    # Iterate over all images
    for idx, (im, label) in enumerate(loaded_data):
        if idx % 5000 == 0:
            print(f'Converting image {idx}/{len(loaded_data)}')
        im_array = np.array(im)

        # Assign a binary label y to the image based on the digit
        if label < 5:
            red_prob = ratio
        else:
            red_prob = 1-ratio
        
        color_red = (np.random.uniform() < red_prob)

        colored_array = color_grayscale_arr(im_array, red=color_red)

        im_list.append(Image.fromarray(colored_array))
        target_class.append(int(label))
        protected_class.append(int(color_red))
    
    return (im_list, target_class, protected_class)

class MNIST:

    _NAME = "coloredMNIST"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size



    def download_files(self):
        self.train_mnist = tv_datasets.mnist.MNIST(root=self.dest_folder, train=True, download=True)
        self.test_mnist = tv_datasets.mnist.MNIST(root=self.dest_folder, train=False, download=True)


    def processing(self):
        num_train = len(self.train_mnist)
        indices = list(range(num_train))
        seed_everything(2020)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[10000:], indices[:10000]

        colored_MNIST_train = process_colored_MNIST(torch.utils.data.Subset(self.train_mnist, train_idx), ratio = 0.8)
        colored_MNIST_dev = process_colored_MNIST(torch.utils.data.Subset(self.train_mnist, valid_idx), ratio = 0.5)
        colored_MNIST_test = process_colored_MNIST(self.test_mnist, ratio = 0.5)

        torch.save(colored_MNIST_train, os.path.join(self.dest_folder, "colored_MNIST_train.pt"))
        torch.save(colored_MNIST_dev, os.path.join(self.dest_folder, "colored_MNIST_dev.pt"))
        torch.save(colored_MNIST_test, os.path.join(self.dest_folder, "colored_MNIST_test.pt"))


    def prepare_data(self):
        self.download_files()
        self.processing()