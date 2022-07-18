from .coco import COCO

def init_data_class(dest_folder, batch_size):
    return COCO(dest_folder = dest_folder, batch_size=batch_size)
