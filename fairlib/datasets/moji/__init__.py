from .processed import Moji

def init_data_class(dest_folder, batch_size):
    return Moji(dest_folder = dest_folder)
