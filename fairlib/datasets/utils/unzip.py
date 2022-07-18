import zipfile
from pathlib import Path
import os

def unzip(file_path, dest_path):

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)  # create folder if it does not exist

    with zipfile.ZipFile(file_path,"r") as zip_ref:
        zip_ref.extractall(dest_path)