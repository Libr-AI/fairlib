from .imSitu import imSitu

def init_data_class(dest_folder, batch_size):
    return imSitu(dest_folder = dest_folder, batch_size=batch_size)
