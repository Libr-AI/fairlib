import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os

class COMPASDataset(BaseDataset):

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "COMPAS_{}.pkl".format(self.split))

        data = pd.read_pickle(self.data_dir)

        self.X = data.drop(['sex', 'race', 'is_recid'], axis=1).to_numpy().astype(np.float32)
        self.y = list(data["is_recid"])

        if self.args.protected_task == "gender":
            self.protected_label =np.array(list(data["sex"])).astype(np.int32) # Gender
        elif self.args.protected_task == "race":
            self.protected_label = np.array(list(data["race"])).astype(np.int32) # Race
        elif self.args.protected_task == "intersection":
            self.protected_label = np.array(
                [_r+_s*3 for _r,_s in zip(list(data["race"]), list(data["sex"]))]
                ).astype(np.int32) # Intersectional