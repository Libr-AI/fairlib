import imp
import logging
from typing import Dict

import numpy as np

import torch
import torch.utils.data as data

from ..utils import BaseDataset
from ..utils import full_label_data
from pathlib import Path
import pandas as pd

class TestDataset(BaseDataset):

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
            if self.split == "train":
                _data_dir = "D:\\Project\\Fair_NLP_Classification\\data\\moji_text"
                data_df = pd.read_pickle('{}/{}_df.pkl'.format(_data_dir, file))
                mapped_data_df = pd.read_pickle('{}/mapped_{}_df.pkl'.format(_data_dir, file))
                data = data_df["encoding"].to_list()[:class_n]+mapped_data_df["encoding"].to_list()[:class_n]

            else:
                data = np.load('{}/{}.npy'.format(self.data_dir, file))
                data = list(data[:class_n])
            self.X = self.X + data
            self.y = self.y + [label]*len(data)
            self.protected_label = self.protected_label + [protected]*len(data)

class SampleDataset(BaseDataset):

    p_aae = 0.5 # distribution of the main label, proportion of the AAE
    n = 5000 # target size

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

class BiosDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "hard_text"

    def load_data(self):
        self.filename = "bios_{}_df.pkl".format(self.split)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        if self.args.protected_task in ["economy", "both"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["profession_class"].astype(np.float64) #Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32) # Economy
        else:
            self.protected_label = data["intersection_class"].astype(np.int32) # Intersection

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