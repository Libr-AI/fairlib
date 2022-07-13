from .bios import Bios

def init_data_class(dest_folder, batch_size):
    return Bios(dest_folder = dest_folder, batch_size=batch_size)
