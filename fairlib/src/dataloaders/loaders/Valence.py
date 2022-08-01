import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd


class ValenceDataset(BaseDataset):
    embedding_type = "cls"
    CV_fold = 0
    text_type = "text"
    def load_data(self):
        self.filename = "valence_arousal_{}_{}.pickle".format(self.split, self.CV_fold)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["valence"].astype(np.float64)
        self.protected_label = data["gender"].astype(np.int32)