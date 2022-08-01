import numpy as np
from ..utils import BaseDataset
from pathlib import Path

class DeepMojiDataset(BaseDataset):

    p_aae = 0.5 # distribution of the main label, proportion of the AAE
    n = 100000 # target size

    def load_data(self):
        # stereotyping, 0.5 is balanced 
        if self.split == "train":
            self.ratio = 0.8 
        else:
            self.ratio = 0.5 # stereotyping, 0.5 is balanced 

        self.data_dir = Path(self.args.data_dir) / self.split

        n_1 = int(self.n * self.p_aae * self.ratio) # happy AAE 
        n_2 = int(self.n * (1-self.p_aae) * (1-self.ratio)) # happy SAE
        n_3 = int(self.n * self.p_aae * (1-self.ratio)) # unhappy AAE
        n_4 = int(self.n * (1-self.p_aae) * self.ratio) # unhappy SAE


        for file, label, protected, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                    [1, 1, 0, 0],
                                                                    [1, 0, 1, 0], 
                                                                    [n_1, n_2, n_3, n_4]
                                                                    ):
            data = np.load('{}/{}.npy'.format(self.data_dir, file))
            data = list(data[:class_n])
            self.X = self.X + data
            self.y = self.y + [label]*len(data)
            self.protected_label = self.protected_label + [protected]*len(data)