import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

class BiosDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "hard_text"

    def load_data(self):
        self.filename = "bios_{}_df.pkl".format(self.split)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        # if self.args.protected_task in ["economy", "both"] and self.args.full_label:
        if self.args.protected_task in ["economy", "both", "intersection"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            _input_ids, _token_type_ids, _attention_mask = self.args.text_encoder.encoder(list(data[self.text_type]))
            self.X = _input_ids
            self.addition_values["input_ids"] = _input_ids
            self.addition_values['attention_mask'] = _attention_mask
        else:
            raise NotImplementedError

        self.y = data["profession_class"].astype(np.float64) #Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32) # Economy
        elif self.args.protected_task == "intersection":
            self.protected_label = np.array(
                [2*_e+_g for _e,_g in zip(list(data["economy_class"]), list(data["gender_class"]))]
                ).astype(np.int32) # Intersection
        else:
            self.protected_label = data["intersection_class"].astype(np.int32) # Intersection