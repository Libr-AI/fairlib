from . import moji
from . import bios

name2class = {
    "Moji":moji,
    "moji":moji,
    "bios":bios,
    "Bios":bios,
}

def prepare_dataset(name, dest_folder,batch_size=64):
    if name in name2class.keys():
        data_class = name2class[name]
        initialized_class = data_class.init_data_class(dest_folder=dest_folder, batch_size=batch_size)
        initialized_class.prepare_data()