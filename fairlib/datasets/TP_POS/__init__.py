from .pos import POS

def init_data_class(dest_folder, batch_size):
    return POS(dest_folder = dest_folder, batch_size=batch_size)
