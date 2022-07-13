from .COMPAS import COMPAS

def init_data_class(dest_folder, batch_size):
    return COMPAS(dest_folder = dest_folder, batch_size=batch_size)
