from .MNIST import MNIST

def init_data_class(dest_folder, batch_size):
    return MNIST(dest_folder = dest_folder, batch_size=batch_size)
