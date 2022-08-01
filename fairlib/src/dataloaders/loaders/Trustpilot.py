import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

class TrustpilotDataset(BaseDataset):
    embedding_type = "avg_embedding"
    text_type = "text"

    def load_data(self):
        self.filename = "trustpilot_{}.pkl".format(self.split)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["target_label"].astype(np.float64) #Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_label"].astype(np.int32) # Gender
        elif self.args.protected_task == "age":
            self.protected_label = data["age_label"].astype(np.int32) # Economy
        elif self.args.protected_task == "country":
            self.protected_label = data["country_label"].astype(np.int32) # Economy
        else:
            raise NotImplementedError