import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

class FCL_BiosDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "hard_text"

    selected_classes = [
        "surgeon", "nurse", "photographer", "teacher", 
        "dentist", "psychologist", "software_engineer", "model"]

    prof2id = {j:i for i,j in enumerate(selected_classes)}

    def load_data(self):
        self.filename = "bios_{}_df.pkl".format(self.split)

        self.args.protected_task = "gender"

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        data = data[data["p"].isin(self.selected_classes[:self.args.num_classes])]

        if self.args.protected_task in ["economy", "both"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["p"].map(self.prof2id).astype(np.float64) #Profession


        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32) # Economy
        else:
            self.protected_label = data["intersection_class"].astype(np.int32) # Intersection