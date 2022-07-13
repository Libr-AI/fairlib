from fairlib.datasets.utils.download import download
from fairlib.datasets.utils.bert_encoding import BERT_encoder
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path

professions = ["accountant","architect","attorney","chiropractor","comedian","composer","dentist","dietitian","dj","filmmaker","interior_designer","journalist","model","nurse","painter","paralegal","pastor","personal_trainer","photographer","physician","poet","professor","psychologist","rapper","software_engineer","surgeon","teacher","yoga_teacher"]

professions2id = {
    j:i for i,j in enumerate(professions)
}

gender2id = {
    "m":0,
    "f":1
}

class Bios:

    _NAME = "Bios"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.encoder = BERT_encoder(self.batch_size)

    def download_files(self):

        for split in self._SPLITS:
            download(
                url = "https://storage.googleapis.com/ai2i/nullspace/biasbios/{}.pickle".format(split), 
                dest_folder = self.dest_folder
                )

    def bert_encoding(self):
        for split in self._SPLITS:
            split_df = pd.DataFrame(pd.read_pickle(Path(self.dest_folder)/"{}.pickle".format(split)))

            text_data = list(split_df["hard_text"])
            avg_data, cls_data = self.encoder.encode(text_data)
            split_df["bert_avg_SE"] = list(avg_data)
            split_df["bert_cls_SE"] = list(cls_data)
            split_df["gender_class"] = split_df["g"]
            split_df["profession_class"] = split_df["p"].map(professions2id)

            split_df.to_pickle(Path(self.dest_folder) / "bios_{}_df.pkl".format(split))

    def prepare_data(self):
        self.download_files()
        self.bert_encoding()